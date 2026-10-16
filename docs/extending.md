# Adding a component

Each stage is an interface plus a name registry. A new model is a class, one registry
entry and a config value. Do not write a new pipeline script for it.

| stage | registry | interface |
|---|---|---|
| separator | `separators.REGISTRY` | `__init__(cfg, device)`; `__call__(wav, sr, row) -> list[Stem]` |
| tagger | `taggers.REGISTRY` | `__init__(cfg, taxonomy, device)`; `__call__(wav, sr, **ctx) -> TagResult` |
| encoder | `encoders.REGISTRY` | `sample_rate`; `load(device) -> self`; `layer_features(wav) -> (layers, dim)`; `describe() -> dict` |
| transcriber | `transcribers.REGISTRY` | `__init__(cfg, device)`; `__call__(wav, sr, instruments, stem=None) -> list[Note]` |

`wav` is a mono float32 numpy array at sample rate `sr`. Resample inside the component
with `amtflow.audio.resample` when the model needs another rate. `ctx` carries `row`
(the manifest row) and `stem` (the stem name).

## Example: a tagger

```python
# src/amtflow/taggers.py
class MyTagger:
    name = "mine"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.taxonomy = taxonomy
        self.model = load_my_model(cfg["checkpoint"], device)   # your code
        self.threshold = float(cfg.get("threshold", 0.5))

    def __call__(self, wav, sr, **ctx) -> TagResult:
        label_scores = self.model(wav, sr)                       # {"electric guitar": 0.9, ...}
        fams = {self.taxonomy.label_families.get(normalize_label(l))
                for l, s in label_scores.items() if s >= self.threshold}
        return TagResult(self.name, self.taxonomy.order(f for f in fams if f),
                         groups=label_group_scores(label_scores, self.taxonomy))

REGISTRY["mine"] = MyTagger
```

```yaml
# my.yaml
extends: default.yaml
tagger:
  name: union
  members:
    - {name: mine, checkpoint: /path/to/ckpt}
    - {name: alm, model: music-flamingo, prompt: both, label_overrides: {percussion: drums}}
```

When a label is not yet mapped to a family, add it to `label_families` (and to
`label_groups` if it names a specific MuScriptor instrument) in your config.

Return `TagResult(..., abstained=True)` when the tagger cannot answer, e.g. a remote
service is down. `vote` then ignores it instead of counting it as "no instruments".

## Example: a transcriber

A transcriber receives the instrument constraint as a list of MuScriptor group names.
A model with another instrument vocabulary maps these names itself. A model without
instrument conditioning may ignore them. Return `Note(start, end, pitch, instrument)`
with times in seconds from the start of `wav`.

## Model weights

Load Hub weights with `from_pretrained` / `hf_hub_download` at a pinned revision. For
weights outside the Hub, add an entry (URL + SHA-256) to `assets.ASSETS` and call
`assets.resolve(cfg.get("checkpoint"), "<asset name>")`. Users can then write
`checkpoint: auto` to download and verify the file on first use.

## Checklist

Every new name appears in:
- the registry dict in its module, and the module docstring list where there is one;
- `docs/configuration.md` (its keys);
- the module tables in `docs/usage.md` (All modules, Registered names).

Also add:

- unit test under `tests/` that runs on CPU without downloading weights (fake the
  model as `tests/test_transcribers.py` does);
- a built-in config only when it is a recommended recipe, with a header comment
  saying what it is for and the evidence behind it.
