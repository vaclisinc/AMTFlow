# MulTTiPop: a voice family, and references with the sung melody labelled as voice

This bundle tests whether AMTFlow's default should transcribe vocals. It compares the
current default (`configs/default.yaml`: the seven MIDI-REPA families plus **voice**,
arm R9) with the previous default without voice (arm R4, `results/multtipop-yt-v2-001`).
Both use MATPAC++ ∪ Music Flamingo, top-N routing and MuScriptor medium.

Audio, segments and the metric are as in `results/multtipop-yt-v2-001`. Scores are
note-onset F1 ×100 (50 ms), averaged over segments. CIs are from a paired bootstrap
over segments (10k resamples, seed 0). There is one run per arm.

## The vocal problem in the references

MulTTiPop's references are Lakh MIDI arrangements. Arrangers rarely put the sung melody
on a voice program. Of the reference tracks named as vocals or melody, only about 10%
use a voice program. The rest use flute (17%), alto/soprano sax (9%), synth lead (8%),
clarinet, piano, guitar and others. A transcriber that writes the singing as `voice` is
therefore scored as wrong in the instrument spaces (exact, group, family), even when the
notes are right.

To measure this, we also score against **vocal-corrected references**. These are the
official MIDI with the tracks that carry the sung melody moved to a voice program (52).
Notes and timing are unchanged.

A track is relabelled when two kinds of evidence agree:

- **A, the track name**: the track is named as vocals (`named_vocal_tracks.json`;
  GM patch names such as "Synth Voice" or "Pad" do not count).
- **B, the audio**: pYIN on the BS-RoFormer vocals stem gives the sung pitch every
  10 ms. A note frame matches when the stem is voiced and its pitch is within half a
  semitone of the note, modulo the octave (chance is about 1/12). `sung_precision` is
  the share of the track's sung frames that match. `coverage` is the share of the
  singing that the track explains.

The rule (`scripts/relabel.py`):

- a named track with sung_precision ≥ 0.3 is relabelled;
- an unnamed track is relabelled with sung_precision ≥ 0.6 and coverage ≥ 0.3, and
  only if its GM class is one that arrangers use for a sung line: pipe, reed, brass or
  synth lead.

Piano, guitar, bass, organ, strings and pads are never relabelled, because there a
match is more likely accompaniment that shares the melody's pitches.

Named vocal tracks match far above chance (median sung_precision 0.53, against 0.16
for other tracks). The rule relabels 210 tracks in 193 of 558 segments (8,541 notes):

- 87 tracks with both kinds of evidence;
- 123 unnamed melody tracks on flute, sax, brass or synth lead.

27 named tracks were left unchanged because the audio did not support them.
`vocal_relabel_changes.json` lists every change with its evidence, and
`vocal_evidence_tracks.csv` lists every track.

## Results (test, 392 segments)

| system | exact | family | agnostic | family, vocal-corrected refs |
|---|---:|---:|---:|---:|
| MT3 (published output) | 10.7 | 15.5 | 28.1 | 15.5 |
| YourMT3+ (published output) | 11.5 | 18.9 | 29.3 | 19.7 |
| MuScriptor medium, no tags (published output) | 17.2 | 29.4 | 43.0 | 29.7 |
| R4: previous default, 7 families | 18.9 | 33.7 | 43.1 | 33.6 |
| **R9: default, 7 families + voice** | 18.3 | 32.5 | **44.0** | 33.3 |

The vocal-corrected references raise family F1 for every system that writes voice
(R9 +0.8, YourMT3+ +0.8, MuScriptor +0.3). R4 cannot write voice and is unchanged
(−0.1); MT3 writes almost no voice (−0.0).

Paired differences on test:

| comparison | references | exact | family | agnostic |
|---|---|---:|---:|---:|
| R9 − MuScriptor | official | +1.11 [+0.42, +1.75] | +3.08 [+2.20, +3.99] | **+1.02 [+0.33, +1.75]** |
| R9 − MuScriptor | vocal-corrected | +1.55 [+0.85, +2.25] | +3.57 [+2.71, +4.49] | +1.02 [+0.33, +1.75] |
| R9 − R4 | official | −0.62 [−1.04, −0.26] | −1.18 [−1.61, −0.77] | +0.86 [+0.32, +1.41] |
| R9 − R4 | vocal-corrected | +0.25 [−0.21, +0.69] | −0.30 [−0.79, +0.19] | +0.86 [+0.32, +1.41] |

Dev (166 segments) agrees in every row (`dev_summary.csv`, `dev_bootstrap.csv`).

## Findings

- **With voice, AMTFlow finds more notes.** R9 beats unconditioned MuScriptor on
  instrument-agnostic F1 (+1.0). R4 without voice did not (+0.2 [−0.6, +0.9]):
  constraining a song with vocals to seven instrumental families made MuScriptor drop
  or misplace the singing.
- **The cost of voice on the official references is a labelling artefact.** R9 loses
  1.2 family F1 to R4 on the official references. Once the sung melody is labelled as
  voice, the difference is not significant (−0.3 [−0.8, +0.2]), and agnostic F1 still
  gains +0.9.
- R9 keeps AMTFlow's instrument advantage over MuScriptor: family +3.1 on the official
  references and +3.6 on the vocal-corrected ones.

The default therefore includes voice. `configs/paper.yaml`, the MIDI-REPA pipeline,
has no voice family, because MIDI-REPA transcribed vocal-removed audio.

## Other instruments

The references' instrument labels are unreliable beyond vocals too.
`scripts/stem_attribution.py` finds, for every reference track, the BS-RoFormer stem in
which its notes sound, relative to that stem's usual share at the same pitch
(`instrument_label_audit.csv`, share of notes):

- tracks labelled **bass** sound in the bass stem 77% of the time, and **voice** tracks
  in the vocals stem 58%;
- **winds** sound in the vocals stem 60% of the time, and synthesizer, organ and
  chromatic percussion about 30%: this is the vocal problem above;
- **piano** sounds in the piano stem 47% of the time, and **guitar** in the guitar stem
  only 31% (32% in the bass stem).

Separation errors (for example guitar low notes in the bass stem) cannot be told apart
from label errors here, so only the vocal labels are corrected. Exact- and group-level
scores should be read with this in mind. Family and agnostic F1 are the more reliable
metrics.

## Reproduce

```bash
# stems: amtflow separate --config stems over the MulTTiPop segments (STEMS)
python scripts/vocal_ref.py MULTTIPOP/raw STEMS named_vocal_tracks.json evidence/
python scripts/relabel.py MULTTIPOP/raw evidence/tracks.csv MULTTIPOP/derived/reference-vocal-ab-v1
amtflow run --config default --manifest runs/test.jsonl --out RUNS/multtipop-yt2-test-r9-mix-default-voice
MULTTIPOP_RAW=MULTTIPOP/raw MULTTIPOP_DERIVED=MULTTIPOP/derived AMTFLOW_RUNS=RUNS \
    python scripts/score_refs.py test .
```

`configs/r9-default-voice.yaml` is the exact config of R9. `<split>-r9-default-voice/`
has its per-segment scores against the official references.
