# Using AMTFlow

## Model access

Weights download from the Hugging Face Hub on first use. Set `HF_HOME` to choose
where they go.

1. **MuScriptor** weights are gated: accept the terms on
   [MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium)
   (and `-large` for the `paper` config), then run `hf auth login` once, before the
   first download.
2. **Music Flamingo** (`nvidia/music-flamingo-hf`, about 16 GB) is part of the default
   tagger. To skip it, use a config without it (see [Configs](#configs)).
3. MATPAC++, its released probes and the BS-RoFormer checkpoint download
   automatically. Non-Hub assets are SHA-256-verified into `$AMTFLOW_CACHE`
   (default `~/.cache/amtflow`).


## Transcribing files

```bash
amtflow transcribe song.wav other.mp3 --out out/
# song.wav: 131 notes; mix: acoustic_piano, acoustic_guitar, electric_bass, drums -> out/clips/song.mid
```

`--out` also gets `manifest.jsonl` (the input list) and `resolved_config.yaml` (the
config actually used). Clips already in `--out` are skipped, so an interrupted run
resumes. Reusing an `--out` with a different config is refused unless you pass
`--overwrite`.

`--config paper` runs the MIDI-REPA pipeline instead of the default (see
[Configs](#configs)). `--device cuda:1` picks a GPU. `--cube-fps 50` also writes a
family × time × 88 piano-roll per clip.

From Python:

```python
from amtflow.audio import load_mono
from amtflow.families import load_config
from amtflow.pipeline import Pipeline

pipe = Pipeline(load_config("default"), device="cuda")
result = pipe(load_mono("song.wav", pipe.input_sr), pipe.input_sr)
for note in result["notes"][:5]:
    print(note)   # {'start': 0.52, 'end': 0.98, 'pitch': 64, 'instrument': 'acoustic_guitar', 'stem': 'mix', 'family': 'guitar'}
```

## Output

One JSON per clip:

| field | meaning |
|---|---|
| `notes` | `[{start, end, pitch, instrument, stem, family}]`: seconds from the clip start, MIDI pitch, MuScriptor instrument group, source stem (`mix` without separation), AMTFlow family |
| `stems` | per stem: detected `families`, the `instruments` MuScriptor was constrained to, `n_notes`, and the `tagger` record (scores, member votes, ALM answers) |
| `duration`, `timing` | clip length and seconds spent separating, tagging and transcribing |
| `pipeline` | `config_sha256` and the transcriber settings |

The MIDI file has one track per (stem, instrument) that produced notes. A track is
named after the instrument ("Acoustic Piano"), prefixed with the stem in stem mode, and uses the MuScriptor
group's representative General MIDI program. Drums go on the drum channel, and
velocity is constant. An instrument in the constraint that MuScriptor never used has
no track.


## Batch runs

For many files, write a JSONL manifest with one clip per line (`{"audio": path}`,
optional `id`, `offset`, `duration`) and use `amtflow run`:

```bash
amtflow run --config default --manifest clips.jsonl --out runs/x [--midi]
# shard across processes or GPUs (resumable; existing outputs are skipped):
amtflow run --manifest clips.jsonl --out runs/x --shard 0 --num-shards 4 --device cuda:0
```

Throughput comes from running several worker processes per GPU. For large corpora,
run the tagger once as its own stage, so the 8B ALM need not load in every worker:

```bash
amtflow run --config default --stage tag --manifest clips.jsonl --out runs/tags
# then transcribe with a config that sets  tagger: {name: precomputed, dir: runs/tags}
```


## Configs

Built-in configs live in [`configs/`](../configs/). `--config` takes a built-in name or a
path. Your own YAML can start with `extends: default.yaml` and override only what
differs (mappings merge, other values replace).

| config | what it is |
|---|---|
| `default` | **Recommended.** Mixture; MATPAC++ ∪ Music Flamingo tagger; 7 families plus voice; top-N routing inside each detected family; MuScriptor medium with prelude forcing. |
| `paper` | The MIDI-REPA paper's pipeline, made for vocal-removed music, so it has no voice family: one MuScriptor instrument per family, MuScriptor large. Add `--cube-fps 10.8` for its 7 × 108 × 88 cube per 10 s. Results in `results/` with "paper tagger" use this tagger and routing with MuScriptor medium (except GuitarSet `paper`). |
| `fast` | `paper` with MuScriptor medium and batched chunk decoding: about 6x faster, about 1 F1 point lower. |
| `stems`, `stems-click` | BS-RoFormer six-stem separation; fixed constraints on the bass/drums/guitar/piano/vocals stems, tagger on `other`. `stems-click` mixes a beat click into each stem. They trail the mixture on pop music. Needs `--extra separation`. |
| `vote` | Majority vote over encoder-probe taggers and two ALMs (Music Flamingo, Qwen3-Omni). |
| `fine`, `fine-voice` | Finer families: brass / reeds / flutes, organ, chromatic percussion, and optionally voice. |

To drop Music Flamingo, override the tagger, e.g. MATPAC++ only:

```yaml
extends: default.yaml
tagger:
  name: union
  members:
    - {name: matpac, variant: music, heads: [openmic, mtg_instrument], probe_dir: auto,
       thresholds: {piano: 0.5, guitar: 0.5, bass: 0.5, drums: 0.5, strings: 0.5, winds: 0.5}}
```

Lists replace rather than merge, so `members` is given in full. Omitted MATPAC++ keys
(`window_seconds`, `hop_seconds`, `top_fraction`, `minimum_windows`) take code defaults
equal to `paper.yaml`. Dropping Music Flamingo cuts tagging time about 3x, at
some recall: on our example it missed the bass that Music Flamingo found.

Every config key is documented in [configuration.md](configuration.md).


## All modules

| stage | modules |
|---|---|
| **Source separation** (optional) | `bsroformer`: any BS-RoFormer checkpoint, the SW-Fixed six-stem model by default · `precomputed`: reuse a stem pool written once by `amtflow separate` · `none`: transcribe the mixture |
| **Per-stem rules** | fixed instrument constraints for known stems (bass, drums, guitar, piano, vocals) · recognize instruments only on the residual `other` stem · fix a stem's family but route inside it · skip silent stems · mix a drum stem or a beat click into each stem for rhythmic context |
| **Instrument conditioning** (optional) | recognition and routing below, together; `tagger: {name: fixed, families: []}` skips both and runs MuScriptor unconstrained |
| **Instrument recognition** | `matpac`: MATPAC++ with its authors' OpenMIC and MTG-Jamendo probes · `probe`: any of 8 frozen encoders (MATPAC++, MERT, MuQ, PupuM2D, Audio Flamingo 3 / Next, Qwen3-Omni AuT) with our published OpenMIC heads · `alm`: Music Flamingo or Qwen3-Omni, in-process or over HTTP, with open or closed prompts · `stems53`: instrument presence from a 53-stem separator · `muscriptor_self`: MuScriptor's own unconditioned guess |
| **Ensembling** | `vote`: union, majority or at-least-*k* over any recognizers, aware of abstentions · `precomputed`: tag once, transcribe later on other machines |
| **Family routing** | 7 families (piano, guitar, bass, drums, strings, winds, synthesizer) plus voice in `default`, or finer ones (brass, reeds, flutes, organ, chromatic percussion, voice) · per family: one default instrument, the best-scoring one, or the top-*N* instruments by evidence |
| **Pretrained AMT model** | any model that accepts an instrument set; built in: MuScriptor small / medium / large · prelude forcing across 5 s chunks or batched decoding · classifier-free guidance · logit bias · quiet-point chunk boundaries · `pyin` monophonic baseline |
| **Output** | per-clip JSON with notes and the full recognition record · multi-track MIDI · family × time × 88 piano-roll cube |
| **Scale** | JSONL manifests · sharding over processes and GPUs · resumable runs that refuse to mix configs · two-stage tag → transcribe pipelines · one GPU per large recognizer |
| **Evaluation** | GuitarSet · MulTTiPop (exact / group / family / agnostic F1, published-output baselines) · OpenMIC-2018 encoder probing and head export |

Trained OpenMIC heads for every encoder are on the Hub at
[vaclis/amtflow](https://huggingface.co/vaclis/amtflow) (`head: openmic/<encoder>-mlp`).

## Registered names

The `name` to write in a config:

| stage | names |
|---|---|
| separator | `none` (mixture), `bsroformer` (SW-Fixed six-stem by default, or any BS-RoFormer checkpoint), `precomputed` (a stem pool written by `amtflow separate`) |
| tagger | `matpac`, `probe`, `alm`, `stems53`, `muscriptor_self`, `fixed`, `oracle` (reference MIDI), `precomputed`, `vote` / `union` |
| encoder (for `probe`) | `matpac-music`, `matpac-general`, `mert`, `muq`, `pupum2d`, `af3-whisper`, `afnext-whisper`, `qwen3-omni-aut` |
| transcriber | `muscriptor` (small / medium / large), `pyin` (monophonic baseline) |
