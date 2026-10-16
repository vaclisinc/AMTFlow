"""`amtflow` command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _openmic(args: argparse.Namespace) -> None:
    from .tagging import benchmark, openmic

    if args.step == "manifest":
        print(json.dumps(openmic.write_manifest(args.root, args.manifest), indent=1))
    elif args.step == "extract":
        benchmark.run_extract(args.encoder, args.manifest, args.root, args.out,
                              smoke=args.smoke, workers=args.workers)
    elif args.step == "probe":
        seeds = [int(s) for s in args.seeds.split(",")]
        for head in args.heads.split(","):
            lrs = tuple(float(x) for x in args.lrs.split(",")) if args.lrs else None
            benchmark.run_probe(args.encoder, args.manifest, args.out, head, seeds, args.epochs,
                                lrs, args.probes_name)
    elif args.step == "official":
        benchmark.run_matpac_official(args.manifest, args.out, args.probe_weights)
    elif args.step == "export-head":
        print(json.dumps(benchmark.run_export_head(args.encoder, args.manifest, args.out,
                                                   args.heads.split(",")[0], args.head_out,
                                                   args.probes_name), indent=1))
    elif args.step == "report":
        rows = benchmark.report(args.out, args.probes_name)
        for r in rows:
            print(f"{r['encoder']:24s} {r['head']:24s} mAP {r['test_map']:.4f} "
                  f"macroF1 {r['test_macro_f1']:.4f} layers {r['layers']}")


def _run(args: argparse.Namespace) -> None:
    from .families import load_config
    from .pipeline import run_manifest

    cfg = load_config(args.config)
    fps = args.cube_fps or cfg.get("output", {}).get("cube_frames_per_second")
    run_manifest(cfg, args.manifest, args.out, args.audio_root, args.shard, args.num_shards,
                 args.device, fps, args.stage, args.follow_tags, midi=args.midi,
                 overwrite=args.overwrite)


def _transcribe(args: argparse.Namespace) -> None:
    from .families import load_config
    from .pipeline import run_manifest

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"audio": str(p.resolve())}) + "\n" for p in args.audio))
    run_manifest(cfg, manifest, args.out, device=args.device, cube_fps=args.cube_fps, midi=True,
                 overwrite=args.overwrite)
    for p in args.audio:
        rec = json.loads((args.out / "clips" / f"{p.stem}.json").read_text())
        stems = [s for s in rec["stems"] if "families" in s]
        print(f"{p.name}: {len(rec['notes'])} notes; "
              + "; ".join(f"{s['stem']}: {', '.join(s['instruments']) or 'unconstrained'}" for s in stems)
              + f" -> {args.out / 'clips' / (p.stem + '.mid')}")


def _separate(args: argparse.Namespace) -> None:
    from .families import load_config
    from .separators import save_pool

    cfg = load_config(args.config)["separator"]
    print(json.dumps(save_pool(cfg, args.manifest, args.out, args.device, args.shard, args.num_shards)))


def _eval(args: argparse.Namespace) -> None:
    from . import evaluate

    if args.step == "guitarset-manifest":
        rows = evaluate.guitarset_manifest(args.root, args.player, args.kind, args.seconds)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{len(rows)} clips -> {args.manifest}")
    else:
        print(json.dumps(evaluate.evaluate(args.manifest, args.out, args.reference), indent=1))


def _multtipop(args: argparse.Namespace) -> None:
    from . import multtipop

    if args.step == "manifest":
        print(multtipop.manifest(args.root, args.split, args.manifest, args.audio_dir), "segments ->", args.manifest)
    elif args.step == "cut":
        print(json.dumps(multtipop.cut_recordings(args.root, args.full_audio, args.audio_dir,
                                                           offsets=not args.no_offsets), indent=1))
    else:
        fams = set(args.families.split(",")) if args.families else None
        tag = args.split + (f"_{args.reference_tag}" if args.reference_tag else "") + \
            (f"_fam-{'-'.join(sorted(fams))}" if fams else "")
        s = multtipop.evaluate(args.root, args.split, str(args.estimates),
                               args.out_csv or (Path(args.estimates) / f"multtipop_{tag}.csv"
                                                if Path(args.estimates).exists() else None),
                               args.reference_dir, fams)
        if Path(args.estimates).exists():
            json.dump(s, open(Path(args.estimates) / f"multtipop_{tag}.json", "w"), indent=1)
        print(json.dumps(s, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="amtflow")
    sub = ap.add_subparsers(dest="command", required=True)

    config_help = ("a built-in config name (default, paper, fast, stems, stems-click, vote, fine, "
                   "fine-voice) or a path to a YAML file")

    tr = sub.add_parser("transcribe", help="transcribe audio files to <out>/clips/<name>.{json,mid}")
    tr.add_argument("audio", type=Path, nargs="+")
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--config", default="default", help=config_help)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--cube-fps", type=float, help="also write a family x time x 88 cube per clip")
    tr.add_argument("--overwrite", action="store_true",
                    help="redo clips that --out already holds from a different config")
    tr.set_defaults(func=_transcribe)

    run = sub.add_parser("run", help="transcribe a JSONL manifest {audio, [id, offset, duration]}")
    run.add_argument("--config", default="default", help=config_help)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--audio-root", type=Path, help="base directory for relative audio paths")
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--device", default="cuda")
    run.add_argument("--cube-fps", type=float, help="also write a family x time x 88 cube per clip")
    run.add_argument("--midi", action="store_true", help="also write a MIDI file per clip")
    run.add_argument("--overwrite", action="store_true",
                     help="redo clips that --out already holds from a different config")
    run.add_argument("--stage", default="all", choices=("all", "tag"),
                     help="tag: only separation + instrument constraints (read back later with "
                          "tagger {name: precomputed, dir: OUT})")
    run.add_argument("--follow-tags", action="store_true",
                     help="with a precomputed tagger: defer clips whose tags are not written yet "
                          "and revisit them until all are done")
    run.set_defaults(func=_run)

    ev = sub.add_parser("eval", help="note-level accuracy of an `amtflow run` output")
    ev.add_argument("step", choices=("guitarset-manifest", "score"))
    ev.add_argument("--root", type=Path, help="GuitarSet directory (audio_mono-mic/, annotation/)")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--out", type=Path, help="run output directory to score")
    ev.add_argument("--reference", default="guitarset")
    ev.add_argument("--player", default="05")
    ev.add_argument("--kind", default="comp")
    ev.add_argument("--seconds", type=float, default=30.0)
    ev.set_defaults(func=_eval)

    sp = sub.add_parser("separate", help="separate a manifest once into a precomputed stem pool")
    sp.add_argument("--config", default="stems", help="config whose `separator` is used: " + config_help)
    sp.add_argument("--manifest", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--shard", type=int, default=0)
    sp.add_argument("--num-shards", type=int, default=1)
    sp.set_defaults(func=_separate)

    mp = sub.add_parser("multtipop", help="MulTTiPop multitrack transcription benchmark")
    mp.add_argument("step", choices=("cut", "manifest", "score"))
    mp.add_argument("--full-audio", type=Path, help="cut: directory of full YouTube audio <ytid>.<ext>")
    mp.add_argument("--root", type=Path, required=True,
                    help="MulTTiPop data directory (see docs/benchmarks.md)")
    mp.add_argument("--audio-dir", type=Path, help="cut/manifest: segment audio directory "
                    "(default <root>/youtube)")
    mp.add_argument("--no-offsets", action="store_true", help="cut: ignore YOUTUBE_OFFSETS")
    mp.add_argument("--split", default="dev", choices=("dev", "test", "all", "dev_rich", "test_rich"))
    mp.add_argument("--manifest", type=Path)
    mp.add_argument("--estimates", help="`amtflow run` output dir, or mt3 | yourmt3 | muscriptor")
    mp.add_argument("--out-csv", type=Path)
    mp.add_argument("--reference-dir", type=Path, help="corrected reference MIDIs to score against")
    mp.add_argument("--reference-tag", help="suffix for the score files, e.g. clean1")
    mp.add_argument("--families", help="comma list: score only these families, e.g. piano,guitar,bass,drums")
    mp.set_defaults(func=_multtipop)

    om = sub.add_parser("openmic", help="OpenMIC-2018 instrument-tagging encoder benchmark")
    om.add_argument("step", choices=("manifest", "extract", "probe", "official", "export-head",
                                     "report"))
    om.add_argument("--root", type=Path, help="OpenMIC-2018 dataset directory")
    om.add_argument("--manifest", type=Path, help="directory holding manifest.jsonl/labels.npz")
    om.add_argument("--out", type=Path, help="benchmark run directory")
    om.add_argument("--encoder")
    om.add_argument("--heads", default="linear,mlp")
    om.add_argument("--seeds", default="0,1,2,3,4")
    om.add_argument("--epochs", type=int, default=50)
    om.add_argument("--smoke", type=int, default=0, help="first N clips per split")
    om.add_argument("--workers", type=int, default=8)
    om.add_argument("--probe-weights", type=Path)
    om.add_argument("--lrs", help="comma-separated lr grid (default MARBLE's 1e-4..5e-3)")
    om.add_argument("--probes-name", default="probes", help="subdirectory for probe results")
    om.add_argument("--head-out", type=Path, help="export-head: output head file")
    om.set_defaults(func=_openmic)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
