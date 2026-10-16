# MulTTiPop: mixture vs stems, tagging and routing

This study compares ways of driving MuScriptor (medium, batch 1, prelude forcing) on
real pop recordings:

- no instrument constraint;
- instrument-family constraints from the paper's tagger (MATPAC++ ∪ Music Flamingo),
  or oracle families taken from the reference;
- one instrument per family vs top-N routing inside a family;
- transcribing the mixture vs BS-RoFormer SW-Fixed stems.

Arm R4 was the default until a voice family was added; the current default is compared
with it in `results/multtipop-voice-001`.

## Data

[MulTTiPop](https://huggingface.co/datasets/gclef-cmu/multtipop) (Pruyne et al., 2026)
consists of pop segments with aligned multitrack MIDI from the Lakh MIDI Dataset. The
release gives YouTube ids and timestamps, not audio. We downloaded the full tracks and
cut them sample-accurately with `amtflow multtipop cut`, which applies two corrections
(`adjusted.json`, `alignment_audit.csv`):

- **Negative start.** A metadata `start < 0` means the authors' segment begins before
  the video. We pad |start| s of leading silence (17 segments).
- **Per-video offsets.** For 10 videos, the upload available to us is shifted
  0.06–0.97 s against the authors' audio (15 test segments). The shifts were measured
  by sliding our unconditioned MuScriptor transcription against the authors' published
  MuScriptor output. They are stored in `YOUTUBE_OFFSETS` in `src/amtflow/multtipop.py`.

166 of 169 dev segments and 392 of 403 test segments were downloadable. 26 reference
MIDIs (6 dev, 20 test) appear offset by about one beat from the authors' own audio. Our
transcription and the authors' published output show the same lag on these segments.
They are kept as released, for comparability.

## Metric

The paper's onset F1 (`mir_eval`, 50 ms onset, pitch within 50 cents, offsets
ignored), computed per segment and averaged over segments. Notes match only within the
same instrument class:

- exact: GM program;
- group: MuScriptor group;
- family: AMTFlow family;
- agnostic: pitched notes, instrument ignored.

YourMT3+ writes its singing-voice class on GM program 65 (alto sax) in a track named
"Singing Voice". Those notes are read as voice (`VOICE_TRACK_NAMES` in
`src/amtflow/multtipop.py`), not as a wind instrument.

References are Lakh MIDI arrangements whose instrument choices are not verified
against the recordings. Exact and group scores therefore measure agreement with the
arranger, so family and agnostic are the primary metrics. 95% CIs come from a paired
bootstrap over segments (10k resamples, seed 0; `paired_bootstrap.csv`). There is one
run per arm. Published outputs (`pub:*`) were transcribed by the dataset authors from
their own audio and are scored on the same segments.

## Results (onset F1 ×100)

| arm | dev exact | dev family | dev agnostic | test exact | test group | test family | test agnostic |
|---|---:|---:|---:|---:|---:|---:|---:|
| published MT3 | 11.5 | 18.9 | 30.5 | 10.7 | 13.6 | 15.5 | 28.1 |
| published YourMT3+ | 11.6 | 21.6 | 30.9 | 11.5 | 15.9 | 18.9 | 29.3 |
| published MuScriptor medium (no tags) | 20.1 | 35.5 | 44.7 | 17.2 | 25.3 | 29.4 | 43.0 |
| R1 mixture, no constraint | 20.0 | 35.1 | 44.7 | 17.6 | 25.6 | 29.7 | 42.5 |
| R2 mixture, paper tagger | 18.0 | 37.9 | 45.1 | 16.9 | 23.6 | 32.8 | 43.3 |
| R3 mixture, oracle families | 18.5 | 40.4 | 45.4 | 17.4 | 25.1 | 35.6 | 44.0 |
| **R4 mixture, paper tagger + top-N (`default`)** | **22.2** | **39.9** | **45.7** | **18.9** | **26.7** | **33.7** | **43.1** |
| R4b R4, batched decoding | 21.8 | 38.9 | 45.1 | | | | |
| R5 stems, paper tagger on `other` | 18.5 | 35.8 | 42.0 | | | | |
| R6 stems, oracle on `other` | 18.6 | 36.7 | 42.3 | 17.0 | 26.7 | 32.9 | 40.8 |
| R7 stems, top-N on `other` | 17.8 | 35.3 | 41.3 | 17.2 | 26.1 | 32.1 | 40.5 |
| R8 stems, all stems + `other` top-1 | 18.8 | 35.5 | 41.7 | | | | |

`summary.csv` holds every space, precision/recall and the `_rich` subsets (segments
with at least two non-piano/guitar/bass/voice groups of ≥ 10 notes). The
`<split>-<arm>/multtipop_<split>.csv` files hold per-segment scores, and
`<split>-<arm>/config.yaml` the exact config of each arm. Arms were decided on dev;
only the arms needed for the conclusions were run on test.

## Findings

- **Unconditioned runs reproduce the published output.** R1 vs published MuScriptor,
  agnostic: dev +0.0 [−0.6, +0.7], test −0.5 [−1.0, +0.1].
- **Conditioning improves instrument assignment, not note detection.** R4 vs published
  MuScriptor on test: exact +1.7 [+1.1, +2.4], family +4.3 [+3.4, +5.1], agnostic
  +0.2 [−0.6, +0.9].
- **The tagger helps** (R2 − R1, test family +3.0 [+2.1, +4.0]).
- **Top-N routing removes the cost of one instrument per family** (R4 − R2: exact +2.1
  [+1.1, +3.0], family +0.9 [+0.3, +1.6]). Over R1, R4 recovers about two thirds of
  the oracle-family gain on test (+4.0 vs +5.9).
- **The mixture beats stems** (R7 − R4 test: family −1.6 [−2.6, −0.6], agnostic −2.6
  [−3.7, −1.6]; with oracle tags, R6 − R3 family −2.7 [−3.7, −1.8]). Stems raise
  recall slightly but lose precision, because separation residue becomes false notes.
- **Batched decoding costs accuracy** (R4b − R4, dev family −0.9 [−1.7, −0.2]).

## Reproduce

```bash
# MULTTIPOP: hf/ = the HF dataset; site/midi/ = reference and published MIDIs from
# https://gclef-cmu.org/multtipop; full YouTube audio as <ytid>.<ext> in FULL/
amtflow multtipop cut --root MULTTIPOP --full-audio FULL --audio-dir AUDIO
amtflow multtipop manifest --root MULTTIPOP --split test --audio-dir AUDIO --manifest runs/test.jsonl
amtflow run --config results/multtipop-yt-v2-001/test-r4-mix-paper-topn/config.yaml \
    --manifest runs/test.jsonl --out runs/test-r4-mix-paper-topn
amtflow multtipop score --root MULTTIPOP --split test --estimates runs/test-r4-mix-paper-topn
uv run python results/multtipop-yt-v2-001/summarize.py MULTTIPOP runs OUT
```

The oracle arms read the reference MIDI path from the manifest's `reference` field,
which `amtflow multtipop manifest` writes.
