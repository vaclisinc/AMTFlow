# AMTFlow: notes for coding agents

AMTFlow transcribes multi-instrument music in five steps:

1. Detect the instruments (tagger).
2. Map them to families.
3. Route each family to MuScriptor instrument groups.
4. Decode with MuScriptor under that hard constraint.
5. Write notes as JSON and MIDI.

Read `README.md` and `docs/usage.md` for usage. This file lists what you need to change the code safely.
If an untracked `AGENTS.local.md` exists, read it too. It holds machine-specific
paths: data, model caches, job runners.

## Commands

```bash
uv sync                                  # add --extra separation for BS-RoFormer configs
uv run pytest -q                         # CPU only, no downloads, about 5 s
uv run amtflow transcribe a.wav --out out/            # needs a GPU and HF access to MuScriptor
uv run amtflow run --config CONFIG --manifest m.jsonl --out DIR [--midi]
```

## Where things are

| task | file |
|---|---|
| CLI commands and flags | `src/amtflow/cli.py` |
| per-clip flow, stem rules, batch runner, resume | `src/amtflow/pipeline.py` |
| config loading (`extends`, built-in names), families, routing | `src/amtflow/families.py` |
| taggers (MATPAC++, probe heads, ALMs, vote, ...) | `src/amtflow/taggers.py` |
| MuScriptor wrapper (chunking, prelude forcing, window constraints) | `src/amtflow/transcribers.py` |
| separators, stem pools | `src/amtflow/separators.py` |
| frozen encoders for probe taggers | `src/amtflow/encoders/` |
| JSON → MIDI / piano-roll | `src/amtflow/render.py` |
| benchmark scoring | `src/amtflow/evaluate.py` (GuitarSet), `multtipop.py`, `tagging/` (OpenMIC) |
| config keys | `docs/configuration.md` |
| adding a model | `docs/extending.md` |

## Invariants

- A new model is a registry entry and a config value, never a new pipeline script or
  CLI command.
- `configs/paper.yaml` is the pipeline of the MIDI-REPA paper; every built-in config extends it. Do
  not change its behavior. Put new recipes in a new config, or change
  `default.yaml` only with benchmark evidence in `results/`.
- MuScriptor's constraint is hard: a missed family can never be transcribed. When
  tagging, prefer recall.
- An empty instrument list means *unconstrained*, not "nothing". Use
  `skip_untagged: true` to skip such stems.
- Outputs are written atomically and skipped when they already exist. Keep runs
  resumable.
- Tests must run on CPU without network or model weights. Fake the model (see
  `tests/test_transcribers.py`).
- Weights are fetched at pinned revisions (Hub) or with a SHA-256 (`assets.py`). Do
  not depend on local paths or unpinned downloads.
- Do not commit audio, model weights or run outputs. `results/<name>/` holds only small
  CSV/JSON evidence, configs and a README with the protocol.

## Changing results

A claim in a README must be backed by a bundle in `results/`, holding per-item scores,
the exact configs and a reproduce section. Compare configurations with paired
statistics over the same items (see `results/multtipop-yt-v2-001/summarize.py`).
