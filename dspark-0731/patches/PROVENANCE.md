# DSpark 0731 runtime hotfixes — vendored copy

Source: https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
Commit: `489af95` (tip of `main`, 2026-08-22)
Previous vendored commit: `21da90f1d2000661618fdb21dcd70d0afdddc7bf` (2026-08-20)
Path in source: `patches/`

Copied **verbatim**, no local edits. Each script locates vLLM inside the image
on its own (`/usr/local/lib/python3.12/dist-packages/vllm`, overridable via
`VLLM_ROOT`), so they need no path rewriting for the lmswitch lane — they are
mounted as a single directory at `/opt/dspark-patches` and invoked by
`ai-models/scripts/dspark-0731-bootstrap.sh` in upstream's compose order.

## What changed at 489af95

- **New: `hotfix-vllm-empty-encoder-output.py` (issue #109).** Encoder-only EC
  producer steps that sampled nothing fabricated token ID 0; the paired
  scheduler branch that finishes such a request was missing, so it could loop
  on empty output. Backport of vLLM `7ca49fb`, patching both `v1/outputs.py`
  and `v1/core/sched/scheduler.py`. Invoked, fail-closed, immediately before
  the #27 patcher — upstream's order, and it must run before #27/#43 touch the
  same scheduler file.
- **All seven multi-hunk DSV4 shell backports plus issue22 are now
  transactional** (PR #103): every target and anchor is validated against an
  in-memory staged view, the whole 35-hunk set is published through
  same-directory atomic renames with modes preserved and committed bytes
  verified, and any failure rolls every target back byte-exactly. Upstream now
  aborts before `exec vllm` on a nonzero exit instead of `|| true`; the
  bootstrap here does the same (they used to be `run_soft`).
- **Python patchers fail closed** (PR #108): the encoder copy, reasoning-effort
  rewrite, #21, #55, #27, #43, #26 and suppress-stops steps were chained with
  bare semicolons upstream, so a failure could be masked. Accordingly
  `hotfix-encoding-dsv4-issue21.py` now exits 1 on anchor drift (was: warn +
  exit 0) and `hotfix-dsv4-suppress-stops-in-reasoning.py` exits 1 on a missing
  target.
- **`hotfix-dsv4-issue27-partial-prefill-concurrency.py` parses the cap once**
  (issue #105): `DSPARK_MAX_INFLIGHT_PREFILLS` is read and validated in
  `Scheduler.__init__` instead of on every waiting-admission iteration, so a
  malformed value (`two`, `2.0`, `1x`) warns and falls back to
  `SchedulerConfig.max_num_partial_prefills` at construction rather than
  raising `ValueError` once traffic reaches the waiting queue. It now needs a
  **second anchor** (the `self._inflight_prefills: set[Request] = set()` block
  with its two-line comment) — re-check this one first after any image bump.

## Vendored but NOT invoked by this lane

- `hotfix-dsv4-assistant-final-continuation.py` — upstream made it opt-in
  (`DSPARK_ENABLE_ASSISTANT_FINAL_HOTFIX=0`) on 2026-08-18 after a gated-ON
  boot failed closed. Set that env to `1` in the recipe to turn it on; the
  bootstrap then runs it fail-closed like upstream.
- `hotfix-dsv4-issue31-v2-thinking-budget-gpu.py` — GPU-resident
  `thinking_token_budget`. Upstream flipped it to opt-in
  (`DSPARK_ENABLE_ISSUE31_GPU_HOTFIX=0`) on 2026-08-22 (issue #66): leaving it
  applied reproduced a decode cliff on omit-field traffic, which is what this
  lane serves. Without it a client sending `thinking_token_budget` gets HTTP
  400; requests that omit the field take the same path either way, so the two
  Triton kernels it installs are not otherwise load-bearing here.
- `hotfix-vllm-redact-api-key-log.sh` — new at 489af95, and upstream applies it
  only when `VLLM_API_KEY` / `DSPARK_API_KEYS` is configured (vLLM's
  `log_non_default_args()` prints every `--api-key` value verbatim at startup).
  This lane serves unauthenticated on the CX7 fabric and lmswitch has no
  `--api-key` plumbing, so it is never invoked. If auth is added, apply it and
  then `--status` it, both fail-closed before exec, as upstream's entrypoint
  does.

## Not vendored

Build-time only, used by upstream's `build-dspark-vllm-runtime.sh` for the
historical Stage-C image, never by the Anemll 0.1.1 runtime lane:
`keys-concurrency.patch`, `official-main-b12x-nvfp4-python.patch`,
`fix-nvfp4-ds-mla-long-context.patch`.

## Refresh procedure

Re-copy `patches/hotfix-*.{py,sh}` from a newer upstream commit, update the
commit hashes above, then **run the whole bootstrap chain in order** against the
pinned image (exec line stubbed) — not per-script `--status`. Four patchers now
write `v1/core/sched/scheduler.py` (#109, #27, #43, grammar-advance), so only a
sequential run proves each anchor survives the previous edit. `--status` exists
on the 8 multi-hunk `.sh` patches only; `hotfix-gb10-spin-wait.sh` errors on it.

Then re-sync to gigabyte and **checksum-compare the directory across both
nodes**: `extra_mounts` is symmetric with no per-node override, so a stale copy
on one node is the failure the fail-closed bootstrap exists to catch.
