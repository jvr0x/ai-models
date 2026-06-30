# ai-models

> YAML recipes for [lmswitch](https://github.com/jvr0x/lmswitch)

Pre-built, production-tested YAML configurations for local LLMs. Drop any of these into `ai-models/<name>.yaml` and `lmswitch` will detect, parse, and serve them immediately.

## Runtime support

| Runtime | Backend | File format | Docker needed? |
|---------|---------|-------------|----------------|
| `llama` | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | `.gguf` | No |
| `vllm` | [vLLM](https://github.com/vllm-project/vllm) | safetensors / FP8 / NVFP4 | Yes |

## Quick start

```bash
# 1. Clone lmswitch (includes this repo as a submodule)
git clone --recurse-submodules https://github.com/jvr0x/lmswitch.git
cd lmswitch

# 2. Install
uv tool install -e .
lmswitch init

# 3. Pick a recipe
cp ai-models/qwen3.6-35b.yaml ai-models/my-model.yaml
# Edit: model path, port, ctx, display_name

# 4. Launch
lmswitch
# Type the model number to toggle on
```

## Available recipes

| File | Runtime | Model | Size | Notes |
|------|---------|-------|------|-------|
| `qwen3.6-35b.yaml` | llama | Qwen3.6-35B | 20.8G | GGUF, default GGUF profile |
| `qwen3.6-35b-q8.yaml` | llama | Qwen3.6-35B Q8 | — | Q8_0 quant |
| `qwen3.6-35b-heretic-dflash-aeon7.yaml` | vllm | Qwen3.6-35B A3B heretic | 23G | NVFP4 + D-Flash speculative decoding |
| `qwen3.6-35b-nvfp4-nvidia-aeon7.yaml` | vllm | Qwen3.6-35B-A3B NVFP4 | — | Official NVIDIA quant |
| `qwen3.6-27b-nvfp4-aeon7.yaml` | vllm | Qwen3.6-27B NVFP4 | — | — |
| `qwen3.6-distill.yaml` | llama | Qwen3.6 Distill | — | Distilled variant |
| `qwen3.6-distill-q8.yaml` | llama | Qwen3.6 Distill Q8 | — | Q8 quant |
| `qwen3.6-kimi-distill.yaml` | llama | Kimi Distill | — | — |
| `qwen3.6-mtp.yaml` | llama | Qwen3.6 MTP | — | Multi-token prediction |
| `qwen3-coder-next.yaml` | llama | Qwen3 Coder Next | — | — |
| `qwen3-vl-8b.yaml` | llama | Qwen3-VL-8B | — | Multimodal vision-language |
| `qwen3-vl-30b.yaml` | llama | Qwen3-VL-30B | — | Multimodal vision-language |
| `qwopus3.6-coder.yaml` | llama | Qwopus3.6 Coder | — | — |
| `gemma4-31b.yaml` | llama | Gemma4 31B | — | — |
| `gemma4-coder.yaml` | llama | Gemma4 Coder | — | — |
| `gemma-4-12b-it.yaml` | llama | Gemma-4-12B Instruct | — | Multimodal (mmproj required) |
| `gemma-4-26b-a4b-it.yaml` | llama | Gemma-4-26B-it | — | — |
| `gpt-oss-20b.yaml` | llama | GPT-OSS-20B | — | — |
| `ornith-35b-q8.yaml` | llama | Ornith-35B Q8 | — | — |
| `ornith-35b-nvfp4-aeon7.yaml` | vllm | Ornith-35B NVFP4 | — | Docker/vLLM |
| `step-3.7-flash.yaml` | llama | Step-3.7 Flash | — | — |
| `nemotron.yaml` | llama | NVIDIA Nemotron | — | — |
| `nemotron3-nano-fp8.yaml` | vllm | Nemotron3-Nano FP8 | — | Docker/vLLM |
| `nemotron3-nano-omni-nvfp4.yaml` | vllm | Nemotron3-Nano Omni | — | Docker/vLLM, multimodal |
| `nex-n2-pro.yaml` | llama | Nex-N2-Pro 397B-A17B | 90G | Large MoE |
| `deepseek-flash-mini.yaml` | llama | DeepSeek Flash Mini | — | — |
| `deepseek-flash-spark.yaml` | llama | DeepSeek Flash Spark | — | — |
| `diffusiongemma-26b.yaml` | llama | DiffusionGemma-26B | — | — |
| `kimi-linear.yaml` | llama | Kimi Linear | — | — |
| `qwen-agentworld-stock.yaml` | vllm | Qwen-AgentWorld-35B-A3B | — | Stock vLLM image |

> **Note**: Size column shows on-disk weight size in GB unless noted. Some vLLM recipes use MoE models (e.g., "35B A3B" = 35B total, 3B active).

## Recipe format

Each YAML file maps to a single model config. The key is:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `runtime` | string | no | `llama` (default) or `vllm` |
| `model` | string | yes | Path relative to `$LMSWITCH_MODELS_DIR` (default `~/models`) |
| `port` | int | yes | OpenAI-compatible server port |
| `ctx` | int | no | Context length (default: 65536) |
| `display_name` | string | no | Human-readable label in tables / synced configs |
| `ready_timeout` | int | no | Seconds to wait for readiness (default: 300 llama / 600 vllm) |
| `force` | bool | no | Bypass pre-load RAM guard |
| `restart` | string | no | `on-failure` for systemd-managed auto-restart |

### Extra keys by runtime

**`llama` (GGUF)**

| Key | Default | Description |
|-----|---------|-------------|
| `gpu_layers` | 99 | `--n-gpu-layers` — offload layers to GPU |
| `threads` | 12 | CPU threads |
| `batch` | 1024 | `--batch-size` |
| `ubatch` | 512 | `--ubatch-size` |
| `fit` | `off` | Auto memory-fit; `off` skips it, `none` omits flag |
| `mmproj` | — | Multimodal projector path (for vision models) |
| `llama_bin` | auto | Path to `llama-server` binary |
| `extra_args` | — | Pass-through flags to `llama-server` |

**`vllm` (Docker)**

| Key | Default | Description |
|-----|---------|-------------|
| `gpu_memory_utilization` | 0.85 | `--gpu-memory-utilization` |
| `image` | `vllm/vllm-openai:latest` | Docker image |
| `entrypoint` | — | Override container entrypoint |
| `env` | — | Environment variables (mapping or list) |
| `trust_remote_code` | false | `--trust-remote-code` |
| `max_num_seqs` | 32 | `--max-num-seqs` |
| `extra_args` | — | Pass-through flags to `vllm serve` |
| `extra_mounts` | — | Additional `-v` bind mounts |
| `tool_call_parser` | — | Auto-enable tool calling (e.g. `qwen3_coder`) |
| `reasoning_parser` | — | Enable reasoning (e.g. `qwen3`) |

> **Full reference**: See [`examples/llama-gguf.yaml`](https://github.com/jvr0x/lmswitch/tree/main/examples/llama-gguf.yaml) and [`examples/vllm.yaml`](https://github.com/jvr0x/lmswitch/tree/main/examples/vllm.yaml) in the [lmswitch](https://github.com/jvr0x/lmswitch) repo.

## Contributing recipes

1. Fork [lmswitch](https://github.com/jvr0x/lmswitch)
2. Add your YAML to the root of `ai-models/`
3. Run `lmswitch add <name>` to validate syntax, or test locally with `lmswitch list`
4. Open a PR — include model source, quantization details, and any special flags

## License

[MIT](LICENSE) © 2026 jvr0x
