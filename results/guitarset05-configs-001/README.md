# GuitarSet player 05: AMTFlow configurations

Note-level transcription accuracy of AMTFlow configurations on the 30 GuitarSet
player-05 accompaniment (`comp`) recordings (`audio_mono-mic`), first 30 s each
(808.9 s of audio, 5,961 reference notes).

Protocol (`amtflow eval`, `src/amtflow/evaluate.py`): references are the six
`note_midi` annotations of each recording, with fractional pitch, restricted to notes
intersecting the 30 s window and clipped to it. Matching is `mir_eval` one-to-one at
50 ms onset / 50 cents, plus offset ratio 0.2 (minimum 50 ms) for the offset metric.
Micro scores sum matches over clips; macro scores average clip F1. Drum notes are
excluded from estimates. One run per configuration on one NVIDIA RTX PRO 6000
(Blackwell), with no retries and no clips dropped.

| config | guitar constraint | onset P | onset R | onset F1 micro | macro | +offset F1 micro | transcribe s / audio-min |
|---|---|---:|---:|---:|---:|---:|---:|
| `paper` (large, batch 1, prelude) | clean_electric_guitar | 0.819 | 0.880 | 0.849 | 0.865 | 0.417 | 32.2 |
| `fast` (medium, batch 6) | clean_electric_guitar | 0.801 | 0.878 | 0.838 | 0.856 | 0.417 | 4.9 |
| `fast`, tagger replaced by fixed `guitar` | clean_electric_guitar | 0.801 | 0.878 | 0.838 | 0.856 | 0.417 | 4.9 |
| `fast`, guitar family = 3 guitars | acoustic / clean / distorted | 0.810 | 0.901 | 0.853 | 0.870 | 0.432 | 13.4* |
| `stems` (BS-RoFormer) + fast decoding | acoustic / clean / distorted | 0.810 | 0.903 | 0.854 | 0.870 | 0.433 | 4.9 (+4.5 separation) |
| stems + pYIN (monophonic baseline) | — | 0.345 | 0.090 | 0.143 | 0.152 | 0.036 | 0.1 (+4.2 separation) |

\* This run shared its GPU with other jobs, so its timing is inflated. It is not a
property of the config.

Files: `metrics.csv` holds every column, including wall-clock seconds per audio minute
(model loading included). `per_clip/<config>.csv` has per-recording scores and timing.
`configs/<config>.yaml` is the exact config of each run; `extends` names a built-in config.

## Findings

- The MATPAC++ tagger routed all 30 recordings to `guitar`, so replacing it with a
  fixed `guitar` family changed nothing (identical outputs).
- Stem separation does not help on solo guitar. Its +1.6-point onset-F1 gain over
  `fast` comes entirely from constraining the guitar family to all three guitar types:
  the mixture with that constraint scores the same (0.853 vs 0.854). MuScriptor then
  picks `acoustic_guitar`, which is correct for GuitarSet.
- One instrument per family (the paper's routing: guitar → clean electric guitar)
  costs about 1.5 onset-F1 points on acoustic guitar. The top-N routing of
  `configs/default.yaml` lets MuScriptor choose among the family's instruments
  instead.
- `paper` (MuScriptor large, prelude forcing) beats `fast` (medium, batched) by 1.1
  onset-F1 points at 6.5x the transcription time. Offset F1 is unchanged.
- Monophonic pYIN on separated stems finds 9% of the notes at 35% precision. It
  cannot follow strummed chords.

## Caveats

This bundle covers solo acoustic guitar only. It does not measure multi-instrument
mixtures or tagger recall; see `results/multtipop-yt-v2-001` for those.

## Reproduce

```bash
# GuitarSet from Zenodo (record 3371780): audio_mono-mic/ and annotation/
amtflow eval guitarset-manifest --root GuitarSet --player 05 --kind comp --seconds 30 \
    --manifest runs/guitarset05.jsonl
amtflow run --config results/guitarset05-configs-001/configs/paper.yaml \
    --manifest runs/guitarset05.jsonl --out runs/gs-paper
amtflow eval score --manifest runs/guitarset05.jsonl --out runs/gs-paper
```
