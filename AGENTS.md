# Project
Audio question answering benchmark code for MMAU-style multiple-choice evaluation.
Main flows compare direct audio QA against caption-based QA.
Models are thin adapters around Qwen audio/omni models and a DeepSeek text agent.
Outputs are JSON result files under `outputs/<model_name>/` and optional HTML reports.

# conda 环境

- 默认：conda activate grpo
- 如果明确使用vllm：请conda activate vllm

# Hot Files
`run_eval.py`
作用: Main CLI entry for loading local MMAU data, selecting model, running evaluation, and saving timestamped results.
常改内容: `AUDIO_ROOT`, `JSON_PATH`, CLI choices/defaults, `build_model()`, `max_samples`, caption path validation, output naming.

`run.sh`
作用: Experiment scratchpad for CUDA device selection, nohup runs, result comparison commands, and historical command variants.
常改内容: GPU id, `--benchmark_mode`, `--audio_model`, `--text_model`, `--caption_path`, `--max_samples`, log/output paths. Treat old `--model` and `--prompt_type` examples as stale unless `run_eval.py` adds them back.

`benchmark/mmau_eval.py`
作用: Core evaluation loop. Builds prompts, calls `model.infer(conversation)`, extracts A/B/C/D, computes overall/task accuracy, writes JSON results.
常改内容: `extract_choice()` parsing rules, supported `prompt_type` values, result schema, task buckets, caption/audio branch logic, debug prints inside the sample loop.

`models/aqa/qwen_omni.py`
作用: Qwen2.5-Omni audio/multimodal adapter. Loads model with bf16, auto device map, flash attention, disables talker, converts Qwen conversations to processor inputs.
常改内容: HF model name, dtype/device map, `attn_implementation`, `thinker_max_new_tokens`, sampling flags, system prompt, prompt format, `debug` output.

`models/aqa/qwen2_audio.py`
作用: Qwen2-Audio AQA adapter using librosa waveform loading and `AutoProcessor`.
常改内容: HF model name, sampling rate handling, `max_new_tokens`, `do_sample`, system prompt, answer-only instruction, audio content key compatibility.

`models/text/qwen_omni.py`
作用: Text-only/caption QA wrapper around Qwen2.5-Omni using the same Qwen conversation format.
常改内容: generation token limit, flash attention/device settings, debug raw conversation, sampling flags.

`models/text/deepseek.py`
作用: Caption QA text adapter. Converts list-style multimodal conversations into plain text and calls `llm_processor.DeepSeekAgent`.
常改内容: external `llm_processor` dependency, prompt serialization, temperature, `max_tokens`.

`prompts/base_prompt.py`
作用: Stateless prompt interface. Prompt classes only transform inputs into Qwen-style conversation lists.
常改内容: Usually none; keep IO and model-specific logic out of prompt classes.

`prompts/aqa_prompt.py`
作用: Direct audio QA prompt with system prompt, few-shot example, audio item, and A/B/C/D answer instruction.
常改内容: system prompt wording, few-shot block, choice formatting, final answer constraint.

`prompts/caption_only_prompt.py`
作用: Caption-only QA prompt. Uses caption text and choices, no audio input.
常改内容: anti-hallucination wording, answer format, whether output should be answer text or A/B/C/D.

`prompts/caption_aqa_prompt.py`
作用: Historical two-step audio prompt that asks for description, reasoning, and answer. Currently not active in `benchmark/mmau_eval.py`.
常改内容: Re-enable only by adding a matching `prompt_type` branch and CLI support.

`conf/qwen25_omni.yaml`
作用: Caption generation config for Qwen2.5-Omni.
常改内容: `hf_tag`, `max_new_tokens`, captioning prompt.

`conf/qwen3_captioner.yaml`
作用: Caption generation config for Qwen3-Omni captioner.
常改内容: `hf_tag`, high `max_new_tokens`, prompt text.

`analyze_results.py`
作用: Single-result HTML report generator with overall, task, sub-category, and error-case tables.
常改内容: metrics grouping, top error limit, HTML output path/style, result schema field names.

`compare_results.py`
作用: Two-result HTML comparison report for accuracy deltas by task and sub-category.
常改内容: input result paths, experiment labels, sort key, output directory, HTML table details.

`check.py`
作用: Quick distribution checker for ground-truth or model choices in JSON result/data files.
常改内容: default file paths, key name such as `gt_choice` or `model_choice`, debug missing ids.

`tests/test_qwen_omni_audio_limit.py`
作用: Manual stress/debug script for long-audio memory and latency with Qwen2.5-Omni.
常改内容: `MODEL_NAME`, `GPU_ID`, `BASE_AUDIO`, `TEST_LENGTHS`, truncation, memory prints. Check syntax before use; this file has had experimental edits.

`tests/test_pipeline.py` and top-level `test_*.py`
作用: Manual smoke tests for model/prompt wiring, often older than current package paths.
常改内容: imports after moving `models/*` into `models/aqa` and `models/text`, dataset/audio roots, sample id, debug mode.

`scripts/download_mmau.py`, `scripts/mmau_to_wavs.py`, `scripts/mmau_to_wavs_qwen.py`
作用: Dataset preparation and audio export helpers.
常改内容: source dataset path/name, output wav directory, sample selection, Qwen-compatible conversion details.

# Workflow
Prefer git history plus targeted reads over broad scans; do not read `.log` files.
Current main command for direct audio QA:
`python run_eval.py --benchmark_mode aqa --audio_model QwenOmni --max_samples 1000`
Current main command for caption QA:
`python run_eval.py --benchmark_mode caption_qa --text_model deepseek --caption_path data/mmau-test-mini-captions.jsonl --max_samples 1000`
Use `Qwen2Audio` only through `--audio_model` in AQA mode; use `deepseek` or `QwenOmni` through `--text_model` in caption QA mode.
`benchmark_mode` is also passed as `prompt_type` into `evaluate()`: supported active values are `aqa` and `caption_qa`.
Dataset records are built from `JSON_PATH` and audio paths are derived as `Path(AUDIO_ROOT) / f"{id}.wav"`.
Caption QA requires JSONL records with `id` and `caption`; missing captions become empty strings.
Result JSON fields expected by analysis tools include `correct`, `question`, `choices`, `gt_answer`, `gt_choice`, `model_raw`, `model_choice`, `model_answer_text`, `task`, `difficulty`, and `sub-category`.
When changing answer format, update both prompt wording and `extract_choice()` or metrics will silently undercount correct answers.
Primary debug locations: `model.infer(..., debug=True)`, prompt `build()` output, `extract_choice()`, per-sample result dict in `benchmark/mmau_eval.py`, and distribution checks in `check.py`.
For model loading issues, inspect dtype/device map/flash attention in model adapters before changing evaluation code.
For path issues, inspect `AUDIO_ROOT`, `JSON_PATH`, `caption_path`, and old absolute paths in tests or `run.sh`.
For dependency issues, note that DeepSeek uses external `llm_processor.DeepSeekAgent`.
Avoid committing generated outputs, logs, large data, or HTML reports unless explicitly requested.
Keep prompt classes stateless and IO-free; keep dataset loading in `run_eval.py`; keep scoring/output schema in `benchmark/mmau_eval.py`.
