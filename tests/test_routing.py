from pathlib import Path

import numpy as np

from amtflow.families import Taxonomy, load_config
from amtflow.render import cube, merged_roll
from amtflow.taggers import ALMTagger, VoteTagger, TagResult, build_tagger, top_fraction_mean, window_starts

CFG = load_config(Path(__file__).parents[1] / "configs" / "paper.yaml")


def test_silent_stem_never_reaches_models_even_at_zero_threshold(monkeypatch):
    from amtflow import pipeline
    from amtflow.separators import Stem

    def unexpected(*args, **kwargs):
        raise AssertionError("silent audio must not reach a tagger or transcriber")

    monkeypatch.setattr(pipeline, "build_transcriber", lambda *args: unexpected)
    monkeypatch.setattr(pipeline, "build_tagger", unexpected)
    pipe = pipeline.Pipeline({**CFG, "min_stem_coverage": 0}, "cpu")
    pipe.separator = lambda *args: [Stem("other", np.zeros(16000, np.float32), 16000, 0)]
    tagged = pipe.tag_clip(np.zeros(16000, np.float32), 16000)
    result = pipe(np.zeros(16000, np.float32), 16000)
    assert tagged["stems"][0]["skipped"] == "silent"
    assert result["stems"][0]["skipped"] == "silent"
    assert result["complete"] and result["notes"] == []


def test_missing_other_is_not_silence(monkeypatch):
    from amtflow import pipeline
    from amtflow.separators import Stem

    monkeypatch.setattr(pipeline, "build_transcriber", lambda *args: None)
    pipe = pipeline.Pipeline({**CFG, "min_stem_coverage": 5}, "cpu")
    pipe.separator = lambda *args: [Stem("other", np.zeros(0, np.float32), 16000, 90, missing=True)]
    result = pipe(np.zeros(16000, np.float32), 16000)
    assert not result["complete"] and result["missing_stems"] == ["other"]
    assert pipe.tag_clip(np.zeros(16000, np.float32), 16000)["stems"][0]["skipped"] == "missing"


def test_paper_taxonomy_routes_one_instrument_per_family():
    tax = Taxonomy(CFG)
    assert list(tax.families) == ["piano", "guitar", "bass", "drums", "strings", "winds", "synthesizer"]
    assert tax.instruments(["winds", "piano"]) == ["acoustic_piano", "brass_section"]
    assert tax.label_families["instrument---electricguitar".removeprefix("instrument---")] == "guitar"
    assert tax.label_families["voice"] is None


def test_union_orders_families_and_falls_back():
    tax = Taxonomy(CFG)
    union = build_tagger({"name": "union", "members": [
        {"name": "fixed", "families": ["winds"]},
        {"name": "fixed", "families": ["bass", "winds"]},
        {"name": "alm", "backend": "http", "base_url": None}]}, tax, "cpu")
    r = union(np.zeros(16000, np.float32), 16000)
    assert r.families == ["bass", "winds"] and r.members[2]["abstained"]
    empty = build_tagger({"name": "union", "members": [{"name": "fixed", "families": []}]}, tax, "cpu")
    assert empty(np.zeros(10, np.float32), 16000).families == ["piano"]


def _votes(*fams, abstain=0):
    return [TagResult("m", list(f)) for f in fams] + [TagResult("a", [], abstained=True)] * abstain


def test_vote_policies():
    tax = Taxonomy(CFG)
    mk = lambda **kw: VoteTagger({"members": [], **kw}, tax, "cpu")
    votes = _votes(["guitar"], ["guitar", "bass", "drums"], ["guitar", "drums"], abstain=1)
    assert mk(policy="union").combine(votes).families == ["guitar", "bass", "drums"]
    maj = mk(policy="majority").combine(votes)                 # 3 voted -> need 2
    assert maj.families == ["guitar", "drums"] and maj.policy == "majority>=2"
    assert mk(policy="at_least", k=3).combine(votes).families == ["guitar"]
    split = _votes(["winds"], ["strings"], ["bass"])
    assert mk(policy="majority").combine(split).families == ["bass", "strings", "winds"]   # plurality
    assert mk(policy="majority", fallback="default").combine(split).families == ["piano"]
    assert mk(policy="majority", fallback="none").combine(split).families == []


def test_flamingo_percussion_override_routes_to_drums():
    tax = Taxonomy(CFG)
    mf = ALMTagger({"backend": "http", "base_url": "x", "label_overrides": {"percussion": "drums"}}, tax, "cpu")
    mf._ask = lambda wav, prompt: "Percussion, Electric Guitar, vocals" if "every" in prompt else "none"
    r = mf(np.zeros(10, np.float32), 16000)
    assert r.families == ["guitar", "drums"]


def test_windows_cover_the_tail_and_aggregate():
    assert window_starts(10, 6, 6) == [0, 4]
    assert window_starts(5, 6, 6) == [0]
    assert top_fraction_mean([0.1, 0.9, 0.5], 0.1, 2) == 0.7


def test_cube_marks_note_frames():
    notes = [{"start": 0.0, "end": 0.5, "pitch": 60, "family": "piano"},
             {"start": 0.25, "end": 0.3, "pitch": 21, "family": "bass"},
             {"start": 0.1, "end": 0.2, "pitch": 110, "family": "piano"}]
    c = cube(notes, ["piano", "bass"], 1.0, 10)
    assert c.shape == (2, 10, 88)
    assert c[0, :5, 39].all() and not c[0, 5:, 39].any()
    assert c[1, 2, 0] == 1 and c.sum() == 6
    assert merged_roll(c).shape == (10, 88)


def test_extends_merges_onto_parent():
    fast = load_config(Path(__file__).parents[1] / "configs" / "fast.yaml")
    assert fast["transcriber"] == {"name": "muscriptor", "weights": "medium", "batch_size": 6,
                                   "prelude_forcing": False}
    assert fast["families"] == CFG["families"]
    stems = load_config(Path(__file__).parents[1] / "configs" / "stems.yaml")
    assert stems["separator"]["name"] == "bsroformer" and "other" not in stems["stems"]


def test_alm_word_level_mapping():
    tax = Taxonomy(CFG)
    alm = ALMTagger({"backend": "http", "base_url": None}, tax, "cpu")
    assert alm.families_of("Synth Bass") == {"bass"}                 # exact alias wins
    assert alm.families_of("synth keys") == {"synthesizer"}
    assert alm.families_of("synth drums") == {"synthesizer", "drums"}
    assert alm.families_of("bass guitar") == {"bass"}
    assert alm.families_of("male vocals") == set()


def test_vote_config_resolves():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "vote.yaml")
    assert cfg["tagger"]["policy"] == "majority" and len(cfg["tagger"]["members"]) == 5
    assert cfg["families"] == CFG["families"]


def test_fine_routing_modes():
    tax = Taxonomy(CFG)
    groups = {"violin": 0.9, "contrabass": 0.6, "cello": 0.2, "synth_lead": 0.8, "trumpet": 0.1}
    assert tax.route(["strings", "synthesizer"], groups) == ["violin", "synth_pad"]      # paper
    assert tax.route(["strings", "synthesizer", "winds"], groups, {"mode": "top1"}) == \
        ["violin", "trumpet", "synth_lead"]
    assert tax.route(["strings"], groups, {"mode": "topn", "ratio": 0.5}) == ["violin", "contrabass"]
    assert tax.route(["piano"], groups, {"mode": "top1"}) == ["acoustic_piano"]       # no evidence
    assert tax.route(["winds"], {}, {"mode": "top1", "fallback": "all"}) == ["brass_section"]
    assert tax.route(["piano"], groups, {"mode": "top1", "fallback": "all"}) == \
        ["acoustic_piano", "electric_piano"]


def test_alm_fine_labels():
    tax = Taxonomy(CFG)
    alm = ALMTagger({"backend": "http", "base_url": None}, tax, "cpu")
    assert alm.groups_of("distorted electric guitar") == {"distorted_electric_guitar"}
    assert alm.groups_of("Electric Guitar") == {"clean_electric_guitar", "distorted_electric_guitar"}
    assert alm.groups_of("synth lead") == {"synth_lead"}
    assert alm.groups_of("tenor sax") == {"tenor_sax"}
    assert alm.groups_of("guitar") == set()                                    # generic


def test_lead_alone_is_not_a_synth():
    tax = Taxonomy(CFG)
    alm = ALMTagger({"backend": "http", "base_url": None}, tax, "cpu")
    assert alm.families_of("lead vocals") == set() and alm.families_of("lead guitar") == {"guitar"}


def test_builtin_configs_load_by_name(tmp_path):
    from amtflow.families import load_config

    default = load_config("default")
    assert default["routing"]["mode"] == "topn" and default["transcriber"]["weights"] == "medium"
    assert load_config("paper")["routing"]["mode"] == "family"
    own = tmp_path / "mine.yaml"
    own.write_text("extends: paper.yaml\ntranscriber: {weights: small}\n")
    cfg = load_config(own)
    assert cfg["transcriber"] == {**load_config("paper")["transcriber"], "weights": "small"}
    for name in ("fast", "stems", "stems-click", "vote", "fine", "fine-voice"):
        assert Taxonomy(load_config(name)).families


def test_resume_refuses_a_different_config(tmp_path):
    import pytest
    import yaml

    from amtflow.pipeline import run_manifest

    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump({"tagger": {"name": "fixed"}}))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("")
    with pytest.raises(SystemExit, match="different config"):
        run_manifest({"tagger": {"name": "vote"}}, manifest, tmp_path, device="cpu")
