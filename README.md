# Foresight: Planning Future Perception in Streaming VLMs without Retraining

[![arXiv](https://img.shields.io/badge/arXiv-2610.03123-b31b1b.svg)](https://arxiv.org/abs/2610.03123)
[![Project Page](https://img.shields.io/badge/Project-Page-1f9d55.svg)](https://thenaivekid.github.io/foresight/)

Official code for **Foresight**, a training-free method that makes streaming vision-language
models (video LLMs) proactive and asynchronous.
**Paper:** [arXiv:2610.03123](https://arxiv.org/abs/2610.03123) · **Project page:** https://thenaivekid.github.io/foresight/

The system runs perception, prefill, and reasoning as three concurrent threads over a
**single** vision-language model and a **single** linear KV cache. The encoder streams and
encodes frames; the ingester prefills them into the shared cache; the controller reads that
cache each tick through an MVCC snapshot and emits a control JSON (frame-rate steer plus an
optional answer). Because the controller never blocks the encoder, the model can decide
*when* to speak while perception continues.

```
vision_stream ──frames──► input_ingester ──shared KV cache──► controller
```

## Layout

```
foresight/      Core system
  run.py              Entrypoint: builds config, loads backend, wires the three threads
  config.py           AsyncOmniConfig dataclass (all runtime options)
  backend.py          Vision-language model backend
  manager.py          Shared KV cache + MVCC snapshots
  vision_stream.py    Frame capture and encoding thread
  input_ingester.py   Prefill thread
  controller.py       Reasoning / emission thread
  compaction.py       KV cache compaction
  class_pruner.py     Pruning strategies
  clip_pruner.py
  dsh_pruner.py
  proactivity.py      Trigger / gating logic
  event_identity.py   Event de-duplication
  prompts.py          Prompt templates
  trigger_prompts.py
  writer.py           Output serialisation
  util.py             Logging, profiling, video clock, seeding
  tests/              Unit tests

evaluation/        Evaluation harness
  bench_eval.py       Benchmark driver
  evaluate.py         Evaluation entrypoint
  score_bench.py      Scoring
  metrics.py          Metric definitions
  gates.py            Gate fitting and threshold selection
  dataset.py          Dataset loading
  judge_offline.py    LLM-judged content scoring
  benchdata/          StreamingBench and OVO-Bench loaders
  ...                 Ablation, sweep, and reporting scripts

requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

`foresight` uses flat imports, so run it from inside that directory:

```bash
cd foresight
python run.py --video_path /path/to/video.mp4
```

Every field of `AsyncOmniConfig` in `config.py` is exposed as a command-line flag, so
`python run.py --help` lists the full option set.

## Data

Evaluation uses the OmniPro benchmark, which is distributed separately. Download it and
point `--benchmark` at its `benchmark.json`. The loaders in `evaluation/benchdata/` also
support StreamingBench and OVO-Bench.

## Evaluate

```bash
cd evaluation
python bench_eval.py --benchmark /path/to/benchmark.json --out results/
python score_bench.py results/
```

`judge_offline.py` performs content scoring and expects an API key in the environment
(`OPENAI_API_KEY` or `GEMINI_API_KEY`). It is optional — timing metrics do not require it.

## Tests

```bash
python -m pytest foresight/tests -q
```

`foresight/tests/conftest.py` puts the core package and the evaluation harness on `sys.path`;
the tests require `torch` and, for the pruner tests, a CUDA device.

## Notes

- Absolute paths from the development environment have been replaced with placeholders of
  the form `/path/to/...`. Set them to your own locations before running.
- This release covers the vision-only pipeline.
- `wandb` is imported lazily in `evaluation/utils.py` for optional run logging and is not a
  required dependency.

## Citation

```bibtex
@article{neupane2026foresight,
  title   = {Foresight: Planning Future Perception in Streaming VLMs without Retraining},
  author  = {Neupane, Ashok Prasad and Bartaula, Dipan and Belbase, Ankit and Adhikari, Saugat and
             Ghimire, Samip and Poudel, Saroj and Bhattarai, Binod and Paudel, Danda Pani},
  journal = {arXiv preprint arXiv:2610.03123},
  year    = {2026}
}
```
