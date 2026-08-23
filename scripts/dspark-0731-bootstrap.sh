#!/bin/bash
# Bootstrap entrypoint for ghcr.io/anemll/dspark-vllm-gx10:0.1.1 serving
# DeepSeek-V4-Flash-0731 (official or the Keys abliterated derivative) across
# two Sparks.
#
# Ported from the `command:` preamble of
# github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
# (docker-compose.dspark.yml @ 489af95, 2026-08-22). Two jobs, in this order:
#
#   1. Encoder. The image predates 0731: it ships the preview
#      `vllm/tokenizers/deepseek_v4{,_encoding}.py`, whose encoder and
#      reasoning-effort mapping do NOT match 0731's `reasoning_content`,
#      reasoning-effort and tool-argument semantics. The checkpoint carries its
#      own encoder at `encoding/encoding_dsv4.py`; install it into vLLM before
#      import, on BOTH ranks, then fix the wrapper's effort mapping (`low` was
#      folded into `high` pre-0731). Weight loading succeeds either way, so a
#      missing patch shows up only as wrong encodings at request time.
#   2. Hotfixes. Upstream's `patches/` suite, vendored verbatim under
#      ai-models/dspark-0731/patches/ and mounted at /opt/dspark-patches (see
#      that directory's PROVENANCE.md). These are what the recipe's serving
#      behaviour actually depends on — #22 long-context decode, #26 prefix
#      cache, #27 prefill concurrency (reads DSPARK_MAX_INFLIGHT_PREFILLS),
#      #43 decode fairness, #55 tool truncation, #79 spin-wait, #109 empty
#      encoder output, the six v0.27 perf backports, and the
#      stops-in-reasoning guard.
#
# 2026-08-22 (upstream 489af95): the chain is now fail-closed on both sides.
# Upstream used to run the shell train with `|| true`; PR #103 turned each
# multi-hunk backport into a stage-then-atomic-commit transaction and aborts
# before `exec vllm` when one does not apply, and PR #108 did the same for the
# Python patchers. This script's `run_soft` treatment of the perf backports was
# tracking upstream's old `|| true` — that rationale is gone, so they are hard
# now too. Under a transactional patcher a refusal means the image drifted,
# which is exactly what must not silently serve.
#
# Where this lane is still stricter than upstream: a missing patches directory
# is fatal. `extra_mounts` is symmetric with no per-node override, so
# "stale/absent on the worker" is the likely failure here and it would
# otherwise leave the two ranks patched differently after a clean boot.
#
# The one place this lane is still SOFTER: the reasoning-effort mapping rewrite
# below warns and leaves the file unpatched when it recognises neither the
# pre-0731 nor the post-0731 shape, where upstream asserts. Deliberate — a
# future image bump that already ships the fix would otherwise be an opaque
# boot failure. Everything else in this script aborts.
#
# Verified idempotent (2026-08-22): a second run over an already-patched layer
# reports skip/already-applied for every step and exits 0. That matters under
# `restart: always` — fail-closed plus non-idempotent would turn one crash into
# a permanent boot failure.
#
# Verifying the suite against a new image without serving — run the WHOLE chain
# in upstream order, not per-script `--status`. Four patchers now write the same
# file (#109 empty-encoder, #27, #43 and grammar-advance all touch
# v1/core/sched/scheduler.py), so only a sequential run proves each anchor
# survives the previous edit:
#   docker run --entrypoint bash \
#     -v ~/utils/lmswitch/ai-models/dspark-0731/patches:/opt/dspark-patches:ro \
#     -v <checkpoint>/encoding:/model/encoding:ro \
#     ghcr.io/anemll/dspark-vllm-gx10:0.1.1 /dspark-bootstrap.sh   # exec stubbed
#   (`bash <script> --status` exists on the 8 multi-hunk .sh patches only;
#   hotfix-gb10-spin-wait.sh has no such flag and errors on it.)
#
# Runs as the container ENTRYPOINT, so lmswitch's `vllm serve` arguments arrive
# as "$@" (see ai-models/scripts/laguna-bootstrap.sh for the same pattern).
set -e

# The image's own PATH/CUDA exports, which an ENTRYPOINT override skips.
export PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

PATCHES="${DSPARK_PATCHES_DIR:-/opt/dspark-patches}"
VLLM_TOKENIZERS="${VLLM_TOKENIZERS_DIR:-/usr/local/lib/python3.12/dist-packages/vllm/tokenizers}"
# lmswitch bind-mounts the checkpoint at /model, so the compose file's HF-cache
# glob never matches here; the recipe pins DSPARK_ENCODING_FILE instead.
ENCODING_SOURCE="${DSPARK_ENCODING_FILE:-/model/encoding/encoding_dsv4.py}"

die() { echo "[bootstrap] FATAL: $*" >&2; exit 1; }

# Reason: an empty bind mount is indistinguishable from a correct one at boot,
# and the only symptom is a rank serving unpatched vLLM. Fail here instead.
[ -d "$PATCHES" ] || die "patches dir $PATCHES missing (bind mount not present on this node?)"
[ -f "$PATCHES/hotfix-nvfp4-ds-mla-issue22.sh" ] || die "patches dir $PATCHES is empty or incomplete"

# Runs a hotfix. Every one of them is fail-closed as of upstream 489af95: the
# shell backports commit atomically (nothing is written unless every hunk
# validated), so a nonzero exit means the target drifted, not that a partial
# edit landed. Serving on a drifted rank is the failure mode this guards.
run_hard() {
  local f="$PATCHES/$1"
  [ -f "$f" ] || die "$1 not found in $PATCHES"
  case "$1" in
    *.sh) bash "$f" || die "$1 failed" ;;
    *.py) python3 "$f" || die "$1 failed" ;;
  esac
}

# ---------- 1. encoder ----------
if [ ! -d "$VLLM_TOKENIZERS" ]; then
  die "vllm tokenizers dir not found at $VLLM_TOKENIZERS -- image layout changed"
elif [ ! -f "$ENCODING_SOURCE" ]; then
  die "encoder not found at $ENCODING_SOURCE (checkpoint mount or DSPARK_ENCODING_FILE wrong)"
else
  cp "$ENCODING_SOURCE" "$VLLM_TOKENIZERS/deepseek_v4_encoding.py"
  echo "[bootstrap] installed 0731 encoder from $ENCODING_SOURCE"
  VLLM_TOKENIZERS="$VLLM_TOKENIZERS" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["VLLM_TOKENIZERS"]) / "deepseek_v4.py"
source = path.read_text()
# Pre-0731 wrappers mapped every non-max effort to "high", so `low` was
# unreachable. 0731 distinguishes low/high/max.
old = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"'''
new = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort == "high":
                reasoning_effort = "high"
            else:
                reasoning_effort = "low"'''
# Reason: idempotent + tolerant on purpose. The upstream compose asserts here,
# which turns a future image bump that already ships the fix into an opaque
# boot failure.
if new in source:
    print("[bootstrap] reasoning-effort mapping already 0731-correct")
elif old in source:
    path.write_text(source.replace(old, new))
    print("[bootstrap] patched reasoning-effort mapping for 0731")
else:
    print("[bootstrap] WARNING: reasoning-effort mapping not recognised -- left unpatched")
PY
  # #21: encode_arguments_to_dsml must accept dict tool arguments. Operates on
  # the encoder file just installed, so it belongs to the encoder chain.
  run_hard hotfix-encoding-dsv4-issue21.py
fi

# ---------- 2. request-semantics hotfixes ----------
# GPU-resident thinking_token_budget (issue #31 v2). Upstream made this opt-in
# on 2026-08-22 (issue #66) and defaults it OFF: leaving it applied reproduced
# a decode cliff on omit-field traffic, which is ~all traffic on this lane
# (DEFAULT_THINKING sets the effort, clients do not send the budget field).
# Gating costs this lane nothing — the patch's two Triton kernels only fire for
# a request that carries an explicit budget; without the patch such a request
# gets HTTP 400 instead. Set the env to 1 (and recreate) if a client needs it.
if [ "${DSPARK_ENABLE_ISSUE31_GPU_HOTFIX:-0}" = "1" ]; then
  run_hard hotfix-dsv4-issue31-v2-thinking-budget-gpu.py
fi
# #55: a tool call cut by max_tokens reports finish_reason=length and drops
# non-JSON args, instead of poisoning the next turn with a 400.
run_hard hotfix-dsv4-issue55-tool-truncation.py

# ---------- 3. in-image kernel/scheduler patches ----------
# #22: nvfp4_ds_mla long-context decode dispatches to the fast fp8_kv path.
# Load-bearing for this recipe's whole KV shape.
if [ "${DSPARK_SKIP_ISSUE22_HOTFIX:-0}" != "1" ]; then
  run_hard hotfix-nvfp4-ds-mla-issue22.sh
fi
# #79: vLLM shm SpinCondition.busy_loop_s 1s -> 2ms; decode IPC always lands
# inside the old window, so Grace P-cores spun instead of yielding.
if [ "${DSPARK_SKIP_SPIN_WAIT_HOTFIX:-0}" != "1" ]; then
  run_hard hotfix-gb10-spin-wait.sh
fi
# The six v0.27 perf backports + grammar advance, 35 hunks across the kernel
# and scheduler paths. Upstream gates all of them behind one switch and, since
# PR #103, aborts the boot if any one fails to apply (each is a whole-script
# transaction: stage, validate every anchor, then atomic rename + verify, or
# roll back byte-exactly).
if [ "${DSPARK_SKIP_HOTFIX:-0}" != "1" ]; then
  for hf in hotfix-dsv4-mtp-buffer-50312.sh \
            hotfix-dsv4-adaptive-topk-50004.sh \
            hotfix-dsv4-skip-topk-49486.sh \
            hotfix-dsv4-dense-prefill-indexer-48407.sh \
            hotfix-dsv4-skip-empty-c128-48957.sh \
            hotfix-dsv4-flashmla-workspace-50298.sh \
            hotfix-dsv4-grammar-advance.sh; do
    run_hard "$hf"
  done
fi

# ---------- 4. scheduler / cache hotfixes (never skipped upstream) ----------
# #109: an encoder-only EC producer step that sampled nothing must return one
# distinct empty row per scheduled request, not a fabricated token 0, and the
# scheduler must finish such a request once its prompt is consumed. Without
# both halves the scheduler appends token 0, advances grammar/stop state,
# counts fake output against spec-decode accounting, and can loop on empty
# output. Backport of vLLM 7ca49fb; runs first because #27/#43 patch the same
# scheduler file after it.
run_hard hotfix-vllm-empty-encoder-output.py
# #27: cap in-flight chunked prefills at DSPARK_MAX_INFLIGHT_PREFILLS (1-3).
# Anemll 0.1.1 rejects --max-num-partial-prefills, so this env is the only way
# to set it; without the patch the env is a no-op and long prefills serialize.
# Since #105 the cap is parsed once in Scheduler.__init__ instead of on every
# waiting-admission iteration, so a malformed value warns and falls back at
# construction rather than raising ValueError once traffic reaches the queue.
run_hard hotfix-dsv4-issue27-partial-prefill-concurrency.py
# #43: bounded decode service during mixed prefill steps (+ optional per-step
# scheduler diagnostics via DSPARK_ISSUE43_SCHED_DIAG=1).
run_hard hotfix-dsv4-issue43-decode-fairness-and-diag.py
# #26/#36 v2: hybrid coordinator takes the min hit length across KV groups
# again; pairs with VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096.
run_hard hotfix-dsv4-issue26-hybrid-swa-min.py
# Client stop strings stay dormant until </think>.
if [ "${DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX:-0}" != "1" ]; then
  run_hard hotfix-dsv4-suppress-stops-in-reasoning.py
fi
# #52/#53: opt-in, default stock rendering (upstream default since 2026-08-18).
# Fail-closed by design — an ON boot that cannot patch must not serve.
if [ "${DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX:-0}" = "1" ]; then
  run_hard hotfix-dsv4-assistant-final-continuation.py
fi
# Upstream also ships hotfix-vllm-redact-api-key-log.sh, which they apply only
# when VLLM_API_KEY/DSPARK_API_KEYS is set (vLLM's log_non_default_args prints
# every --api-key value verbatim at startup). This lane serves unauthenticated
# on the CX7 fabric and lmswitch has no --api-key plumbing, so it is vendored
# but never invoked. If auth is ever added here, apply it AND `--status` it
# fail-closed before exec, the way upstream's entrypoint does.

echo "[bootstrap] hotfix suite applied from $PATCHES"

# ---------- 5. default thinking mode ----------
# Upstream's DEFAULT_THINKING knob (docker-compose.dspark.yml `case` block),
# ported so the recipe sets one env instead of hand-writing template kwargs.
# Their shipped default is `max`. Request-level chat_template_kwargs always
# win, and with the #31 patch above a client can hard-cap reasoning per request
# with `thinking_token_budget` instead of turning thinking down globally.
#
# `max` is not free: the checkpoint's max directive is "do not stop reasoning
# until no error remains undiscovered" — upstream measured ~12.5k reasoning
# tokens on a moderate prompt. A harness that caps max_tokens at a few hundred
# gets content: null / finish_reason: length. Drop to low/off in the recipe if
# that bites.
if [ -n "${DEFAULT_THINKING:-}" ]; then
  case "$DEFAULT_THINKING" in
    off)  THINK_KWARGS='{"thinking":false}' ;;
    low)  THINK_KWARGS='{"thinking":true,"reasoning_effort":"low"}' ;;
    high) THINK_KWARGS='{"thinking":true,"reasoning_effort":"high"}' ;;
    max)  THINK_KWARGS='{"thinking":true,"reasoning_effort":"max"}' ;;
    *) die "DEFAULT_THINKING must be one of: off, low, high, max (got: $DEFAULT_THINKING)" ;;
  esac
  # An explicit flag in the recipe's extra_args wins — vLLM would otherwise see
  # the option twice and the effective config would be ambiguous.
  for arg in "$@"; do
    case "$arg" in
      --default-chat-template-kwargs*) THINK_KWARGS="" ;;
    esac
  done
  if [ -n "$THINK_KWARGS" ]; then
    echo "[bootstrap] DEFAULT_THINKING=$DEFAULT_THINKING -> $THINK_KWARGS"
    set -- "$@" "--default-chat-template-kwargs=$THINK_KWARGS"
  else
    echo "[bootstrap] DEFAULT_THINKING=$DEFAULT_THINKING ignored (recipe passes --default-chat-template-kwargs)"
  fi
fi

echo "[bootstrap] starting vLLM ..."
exec /usr/local/bin/vllm serve "$@"
