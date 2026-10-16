# Configuration reference

A config is one YAML file. `amtflow` resolves `--config NAME` to `configs/NAME.yaml`.
Any other value is treated as a path.

`extends: other.yaml` deep-merges the file onto its parent: mappings merge key by key,
and every other value (lists, scalars) replaces the parent's. The parent is looked up
relative to the file first, then among the built-in configs, so your own file can say
`extends: default.yaml` from anywhere. `${ENV}` in a config expands to that
environment variable. `amtflow run` writes the fully resolved config to
`<out>/resolved_config.yaml`.

## Families and routing

The pipeline works in two vocabularies:

- **families**: coarse instrument classes the tagger detects (`piano`, `guitar`,
  `bass`, `drums`, `strings`, `winds`, `synthesizer` in `paper.yaml`);
- **instruments**: MuScriptor's instrument groups (`acoustic_piano`,
  `clean_electric_guitar`, `violin`, ...), which are the actual decoding constraint.

| key | meaning |
|---|---|
| `families` | family → default MuScriptor instruments, used under `routing.mode: family` and as the fallback for a family without group evidence. The order is the canonical family order. |
| `fallback_families` | families used when no tagger detects anything (`[piano]`). |
| `label_families` | tagger label → family. Labels are normalized: lowercase, alphanumerics only, `instrument---` prefix removed. `null` means "recognized, routes nowhere" (vocals, organ, ...). Unknown labels route nowhere. |
| `family_groups` | family → every MuScriptor group fine routing may choose from. |
| `label_groups` | label → the specific groups it names (`electricguitar` → clean + distorted). Generic labels (`guitar`) name none and only vote for the family. |
| `routing.mode` | `family`: each detected family gets its `families` defaults (the paper). `top1`: the best-scoring group of each family. `topn`: groups within `ratio` of the family's best score and above `min_score`, at most `max_per_family` (`default.yaml`: ratio 0.5, max 3). |
| `routing.fallback` | for a detected family with no group evidence: `default` (its `families` entry) or `all` (all its `family_groups` when there are at most `fallback_all_max`). |
| `routing.source` | optional separate tagger config whose group scores drive routing, while `tagger` still decides the families. |
| `skip_untagged` | `true`: skip a stem whose constraint is empty rather than transcribe it unconstrained. |

Group evidence comes from the tagger: probe and MATPAC++ label scores, ALM answers
that name specific instruments, or MuScriptor's own unconstrained choice
(`muscriptor_self`).

## Separator and stems

```yaml
separator: {name: none}                       # transcribe the mixture (default)
separator: {name: bsroformer, checkpoint: auto, config: auto}   # SW-Fixed six-stem
separator: {name: precomputed, root: DIR}     # a pool written by `amtflow separate`
```

`bsroformer` accepts any BS-RoFormer `checkpoint` / `config` path. With `auto`, it
downloads the SW-Fixed six-stem model (bass, drums, guitar, piano, vocals, other).

| key | meaning |
|---|---|
| `stems.<stem>` | a rule for that stem. `{families, instruments}` fixes the constraint. `{families}` uses those families' defaults. `{families, route: true}` fixes the family but routes inside it. `{skip: true}` drops the stem. A stem without a rule goes to the tagger. |
| `min_stem_coverage` | skip stems with less than this percent of 100 ms blocks above −40 dBFS. |
| `stem_context` | `{source: drums \| click, gain_db, stems: [...]}`: mix the drum stem, or noise clicks on beats tracked on the mixture, into each listed stem before transcription. Tagging still sees the clean stem. |

## Tagger

Every tagger returns families, optional per-family scores and optional per-group
evidence. A member may set `device` to run on its own GPU.

| `name` | keys |
|---|---|
| `matpac` | `variant` (`music` \| `general_audio`), `heads` (the authors' probes: `openmic`, `mtg_instrument`), `probe_dir` (`auto` downloads them), `thresholds` (family → probability, or a JSON file), `window_seconds`, `hop_seconds`, `top_fraction` (clip score = mean of the strongest windows), `minimum_windows`. |
| `probe` | `head` (a path, `$AMTFLOW_HEADS/<name>.safetensors`, or a Hub name like `openmic/pupum2d-mlp`), optional `encoder`, `thresholds` (default: the head's calibrated per-class thresholds), window keys as above. |
| `alm` | `model` (`music-flamingo` \| `qwen3-omni`), `prompt` (`open` \| `closed` \| `both`), `backend` (`local` \| `http`), `base_url`, `path`, `api_key_env`, `label_overrides`, `max_new_tokens`, `max_audio_seconds`. An HTTP ALM that is unset or unreachable abstains. |
| `vote` / `union` | `members` (tagger configs), `policy` (`union` \| `majority` \| `at_least` with `k`), `fallback` (`plurality` \| `default` \| `none`). Abstaining members do not count. |
| `stems53` | instrument presence from a many-stem BS-RoFormer (e.g. MVSep's 53-stem model). Needs `checkpoint` and `config` paths; also `threshold_db`, `thresholds_db`, `min_coverage`, `scale_db`, `cache_dir`. |
| `muscriptor_self` | an unconstrained MuScriptor pass; each group's share of the notes is its evidence. `weights`, `min_share`, `vote_families`. |
| `fixed` | `families`: a constant answer (`[]` = unconstrained transcription). |
| `oracle` | families from the manifest row's `reference` MIDI (benchmarking only). `families`, `window_seconds`. |
| `precomputed` | `dir`: read the constraints written by `amtflow run --stage tag --out DIR`. |

## Transcriber

| key | meaning |
|---|---|
| `name` | `muscriptor` or `pyin` (monophonic baseline; `stems` lists the stems it transcribes). |
| `weights` | `small` \| `medium` \| `large`, or a path. |
| `batch_size` | decode this many 5 s chunks of a clip at once. Above 1, set `prelude_forcing: false`. |
| `prelude_forcing` | carry notes that are still sounding into the next chunk (better at chunk boundaries; forces batch 1). |
| `cfg_coef` | classifier-free guidance on the conditions (1.0 = off; doubles the batch). |
| `logit_bias` | `{pitch \| shift \| velocity \| program \| drum \| tie \| eos: bias}` added to the decoder's logits. |
| `boundaries` | `fixed` (every 5 s) or `quiet` (start each window at the quietest point). |
| `min_tail_seconds` | trim a final chunk shorter than this, which otherwise produces spurious notes. |
| `dtype` | MuScriptor compute dtype. |

## Output

| key | meaning |
|---|---|
| `output.cube_frames_per_second` | also write `<id>.cube.npz`: a family × frames × 88 uint8 piano-roll (same as `--cube-fps`). |
| `input_sample_rate` | rate audio is loaded at (44100). |
