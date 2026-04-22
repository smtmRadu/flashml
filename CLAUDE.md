# FlashML

This is the project-level `CLAUDE.md` for the repository.

For now, it only documents the `flashml.llm` area because that is the part currently in focus. It should not be treated as full-project guidance yet.

The current scope is limited to:

- `flashml/llm/vllm_chat_openai_entrypoint.py`
- `flashml/llm/vllm_configs.py`
- `flashml/llm/llm_merging.py`
- `flashml/llm/llm_quantization.py`

As this file grows later, add new sections for other parts of the repo instead of silently changing the meaning of the existing LLM notes.

## What These Files Do

- `vllm_chat_openai_entrypoint.py` runs local batch chat inference through `python -m vllm.entrypoints.openai.run_batch`.
- `vllm_configs.py` stores reusable vLLM config dictionaries as uppercase constants ending in `_VLLM_CONFIG`.
- `llm_merging.py` merges LoRA adapters into base models, with separate handling for Unsloth checkpoints.
- `llm_quantization.py` quantizes merged fp16 checkpoints, mainly to BitsAndBytes 4-bit or GPTQ variants.

## Working Assumptions

- The vLLM path here is Linux-only. `vllm_chat_openai_entrypoint` raises on non-Linux platforms.
- The chat entrypoint is batch-oriented, not a long-lived server wrapper.
- Config dictionaries are passed directly into CLI flags. Key spelling matters.
- Several flows assume local Hugging Face-style model folders with files such as `config.json` and `adapter_config.json`.
- Temporary and offload folders are expected in the working directory, especially `./offload_flashml`.

## vLLM Entrypoint Rules

- Preserve the current contract: `messages` may be a single conversation (`list[dict]`), a batch of conversations (`list[list[dict]]`), and may contain `None` items that must round-trip back to `None` outputs.
- Keep the current batch JSONL shape compatible with `/v1/chat/completions`.
- Keep per-request generation controls in the request body:
  - `max_completion_tokens`
  - `temperature`
  - `top_p`
  - `top_k`
  - `reasoning_effort`
- Keep model and server startup settings in `vllm_config`, then remove request-only keys before building the CLI command.
- When `format` is provided, this code currently expects a Pydantic model class and derives a tool schema via `model_json_schema()`.
- Be careful with CLI flag emission:
  - Every config key becomes `--{key}`
  - Empty string or `None` is used as a flag-without-value pattern
  - Lists and tuples are expanded into repeated CLI values
- Do not silently remove the temp file cleanup or the `CalledProcessError` logging.

## vLLM Config Rules

- Preserve the existing naming convention: uppercase constant names ending in `_VLLM_CONFIG`.
- Prefer adding a new config constant instead of mutating an unrelated preset.
- Keep request-time fields in these dictionaries only if the entrypoint expects them and strips them before CLI invocation.
- Do not normalize key spelling casually. This file currently mixes keys such as:
  - hyphenated CLI-style keys: `tensor-parallel-size`, `gpu-memory-utilization`
  - underscored keys: `config_format`
  - dotted keys: `limit-mm-per-prompt.video`, `limit_mm_per_prompt.image`
- Before changing any key format, verify it against the exact vLLM or model loader path used by this module.
- Keep comments that capture model-specific caveats if they reflect real behavior, especially around async scheduling, reasoning parsers, or vision preprocessing.

## Merge Rules

- `merge_model()` is the public entrypoint. Keep it as the main path.
- The merge flow depends on `adapter_config.json` and reads `base_model_name_or_path` from it.
- Unsloth adapters must continue through `_merge_unsloth_model()`.
- Non-Unsloth adapters use Transformers + PEFT merge-and-unload.
- Keep the current output naming convention derived from `adapter_path`:
  - `_fp16`
  - `_bnb_4bit`
  - `_gguf_Q8_0`
- Preserve model-family-specific warnings and guardrails unless the underlying limitation is confirmed fixed:
  - Gemma extra processor and preprocessor files
  - Ministral JSON file reminder
- Do not remove cleanup of `./offload_flashml`.

## Quantization Rules

- `quantize_model()` assumes the source model is already merged and stored locally.
- Keep the Mistral special case that routes `model_type == "mistral3"` into `_quantize_mistral_model()`.
- Supported public quantization names currently include:
  - `bnb_4bit`
  - `gptq_2bit`
  - `gptq_3bit`
  - `gptq_4bit`
  - `gptq_8bit`
- GPTQ requires a calibration dataset. Do not remove that validation.
- Preserve the backend checks around `optimum`, `gptqmodel`, and `auto-gptq`. Those checks are protecting a fragile dependency path.
- Keep output folder naming derived from the source folder suffix, for example replacing `_fp16` with `_bnb_4bit` or `_gptq_4bit`.
- For Mistral quantization, preserve the tokenizer file copy behavior unless the save path is reworked end-to-end.

## Safe Editing Guidance

- If you change function signatures here, also check `flashml/llm/__init__.py`.
- Prefer small, model-specific edits over broad refactors. These files encode many compatibility workarounds.
- If you change a preset or loader behavior because of one model family, confirm it does not break:
  - Qwen text reasoning models
  - Qwen VL models
  - GPT-OSS presets
  - Gemma or Ministral special handling

## Things Worth Calling Out

- `vllm_chat_openai_entrypoint` currently has a loose and partially inaccurate type hint for `messages`; behavior matters more than the annotation.
- These files prioritize practical local workflows over abstraction purity.
- Some comments and strings are blunt or informal. Only clean them up if you are already touching the surrounding logic and can preserve behavior.
