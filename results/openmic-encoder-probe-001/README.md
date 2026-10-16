# OpenMIC-2018 instrument tagging: frozen-encoder probes

What was measured: how well each frozen encoder available to AMTFlow supports
20-class instrument-presence tagging, to choose and ship instrument taggers for
AMTFlow's instrument-conditioned transcription.

## Protocol

- Data: OpenMIC-2018 (20,000 10 s clips). Official `split01` test (5,085 clips) is
  test; the official train partition is split 80/20 by track (seed 0) into train
  (11,932) / val (2,983). Labels are the official aggregated relevances
  (`openmic-2018.npz`): observed entries >= 0.5 are positive; unobserved (clip, class)
  entries are excluded from loss and metrics (about 10% of entries are observed).
- Features: every hidden layer of the frozen encoder, averaged over the clip
  (`src/amtflow/encoders/`; MATPAC++ via its HF port).
- Probes (encoder_mir protocol, adapted to partial labels): per layer and lr, a
  MARBLE MLP head (512, dropout 0.2) or a linear head on train-z-scored features,
  masked BCE, Adam at a constant lr for 50 epochs, best-val-mAP epoch kept. Per
  seed, the (layer, lr) with the best val mAP is scored on test; five seeds;
  ± is the 95% Student-t CI over seeds (probe randomness only, not data sampling).
- Metrics: masked mAP (mean of per-class AP over observed entries), macro/micro F1
  at probability 0.5.
- Reference: the MATPAC++ authors' released OpenMIC linear probe on the same test
  clips (sigmoid scores). It was trained on the full official train partition,
  which contains our val split, so its val score is not comparable (0.932).

## Results (MLP head, MARBLE lr grid 1e-4..5e-3)

| encoder | test mAP | macro-F1 | selected layer |
|---|---:|---:|---|
| AF3-Whisper (Audio Flamingo 3 audio tower) | **0.861** ± 0.001 | 0.793 | 31 |
| MATPAC++ music — authors' linear probe (reference) | 0.861 | 0.791 | 11 (last) |
| AF-Next-Whisper | 0.860 ± 0.003 | 0.788 | 28-32 |
| MATPAC++ music | 0.860 ± 0.002 | 0.792 | 10 |
| MATPAC++ general audio | 0.858 ± 0.002 | 0.787 | 9-10 |
| PupuM2D large | 0.858 ± 0.001 | 0.790 | 18-20 |
| MuQ large (msd-iter) | 0.832 ± 0.001 | 0.764 | 2 |
| Qwen3-Omni AuT | 0.821 ± 0.001 | 0.753 | 23-31 |
| MERT-v1-330M | 0.818 ± 0.002 | 0.740 | 9-10 |

All rows, the linear heads and the extended lr grid are in `metrics.csv`; per-class
test AP of the MLP heads in `per_class_ap_mlp.csv`.

## Findings

- Five encoders tie within 0.004 mAP (AF3-Whisper, AF-Next-Whisper, MATPAC++ music
  and general, PupuM2D); they match the authors' MATPAC++ probe. MuQ, Qwen3-Omni AuT
  and MERT trail by 0.03-0.04. MATPAC++ (about 90M parameters, 0.2 s per 30 s clip)
  is the cheapest of the top group, which supports the paper's choice.
- All encoders tag `synthesizer` well (AP 0.96-0.98), the family the paper's
  MATPAC++ routing could not produce (its thresholded families exclude it).
  Clarinet, organ and mandolin are the weakest classes for every encoder.
- The MARBLE grid selects its lowest lr (1e-4, best epoch 1-5) for the 3840-8192-d
  encoders. Extending the grid to 1e-5 (`lr_grid = wide` rows) changes no MLP score
  by more than 0.002 and no ranking, so the grid edge is not what limits them.
- Linear heads score 0.75-0.81, well below MLP heads and the authors' linear probe.
  This is under-regularization, not the features: a strongly L2-regularized logistic
  regression on MATPAC++ layer 11 reaches 0.858 test mAP (`linear_sanity.json`).
  Compare encoders by the MLP rows.

## Use in AMTFlow

`amtflow openmic export-head` turns each MLP probe (best-val seed) into a head file
for the `probe` tagger, with per-class thresholds calibrated on the val split (F2,
precision >= 0.25). Exported heads reproduce the probe's test mAP exactly.

## Caveats

OpenMIC is 10 s clips with sparse partial labels; this ranks clip-level tagging
features, not the recall of the combined tagger on full songs, which needs a
family-labelled multi-instrument set (see `results/multtipop-yt-v2-001`).

## Reproduce

```bash
# OpenMIC-2018 from Zenodo (record 1432913), extracted to OPENMIC
amtflow openmic manifest --root OPENMIC --manifest runs/openmic
amtflow openmic extract  --root OPENMIC --manifest runs/openmic --out runs/om --encoder matpac-music
amtflow openmic probe    --manifest runs/openmic --out runs/om --encoder matpac-music
amtflow openmic official --manifest runs/openmic --out runs/om   # the authors' MATPAC++ probe
amtflow openmic report   --out runs/om
```
