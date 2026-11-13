#!/usr/bin/env python3
"""Evaluate 5-direction BTC datasets (en-vi, ko-vi, vi-ko, vi-zh, zh-vi) under AIBOX-equivalent source_mode=0."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "s2s_edge"))
sys.path.insert(0, str(ROOT / "tools"))

from s2s_edge.audio import SR, concat_results, load_audio, save_wav
from s2s_edge.pipeline import Translator
from s2s_edge.registry import Registry, runtime_code_fingerprint
import entity_scorer


DIRECTIONS = [
    ("en-vi", "en", "vi"),
    ("ko-vi", "ko", "vi"),
    ("vi-ko", "vi", "ko"),
    ("vi-zh", "vi", "zh"),
    ("zh-vi", "zh", "vi"),
]


def _has_digits_or_entities(rec: dict, src: str, tgt: str) -> bool:
    text = rec.get("text") or ""
    ref = rec.get("ref") or ""
    old_asr = rec.get("asr") or ""
    if re.search(r"\d", text + " " + ref + " " + old_asr):
        return True
    if entity_scorer.extract_entities(text, lang=src):
        return True
    if entity_scorer.extract_entities(ref, lang=tgt):
        return True
    if entity_scorer.extract_entities(old_asr, lang=src):
        return True
    return False


def discover_btc_clips(btc_root: Path) -> dict[str, list[dict]]:
    by_dir: dict[str, list[dict]] = {}
    for dname, src, tgt in DIRECTIONS:
        ddir = btc_root / dname / "BTC"
        clips: list[dict] = []
        if ddir.is_dir():
            for jpath in sorted(ddir.glob("*.json")):
                cid = jpath.stem
                wav_path = ddir / f"{cid}_in.wav"
                if not wav_path.is_file():
                    continue
                meta = json.loads(jpath.read_text(encoding="utf-8"))
                clips.append({
                    "id": cid,
                    "dir": dname,
                    "src": src,
                    "tgt": tgt,
                    "json_path": str(jpath),
                    "in_wav": str(wav_path),
                    "gold_text": meta.get("text") or "",
                    "gold_ref": meta.get("ref") or "",
                    "baseline_asr": meta.get("asr") or "",
                    "baseline_mt": meta.get("mt") or "",
                    "has_entity": _has_digits_or_entities(meta, src, tgt),
                    "has_spoken_numeral": (
                        entity_scorer._has_spoken_numeral_words(meta.get("text") or "")
                        or entity_scorer._has_spoken_numeral_words(meta.get("ref") or "")
                    ),
                })
        by_dir[dname] = clips
    return by_dir


def select_clips(
    by_dir: dict[str, list[dict]],
    mode: str,
    per_dir: int,
    seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    rng = random.Random(seed)
    selected: dict[str, list[dict]] = {}
    coverage = {}
    for dname, _, _ in DIRECTIONS:
        pool = list(by_dir.get(dname, []))
        ent_clips = [c for c in pool if c["has_entity"]]
        spoken_only = [c for c in pool if not c["has_entity"] and c["has_spoken_numeral"]]
        plain = [c for c in pool if not c["has_entity"] and not c["has_spoken_numeral"]]

        if mode == "full":
            chosen = pool
        elif mode == "smoke":
            chosen = (ent_clips[:1] or spoken_only[:1] or pool[:1])
        else:
            # Always include ALL clips with digits/typed entities, then balanced fixed-seed sample
            chosen = list(ent_clips)
            chosen_ids = {c["id"] for c in chosen}
            remaining_target = max(0, per_dir - len(chosen))
            # Prefer balanced split between spoken-numeral clips and plain clips
            rng.shuffle(spoken_only)
            rng.shuffle(plain)
            half = remaining_target // 2
            extra = spoken_only[:half] + plain[: remaining_target - min(half, len(spoken_only))]
            if len(extra) < remaining_target:
                leftovers = [c for c in (spoken_only + plain) if c["id"] not in {x["id"] for x in extra}]
                extra.extend(leftovers[: remaining_target - len(extra)])
            for c in extra:
                if c["id"] not in chosen_ids:
                    chosen.append(c)
                    chosen_ids.add(c["id"])
            chosen.sort(key=lambda c: c["id"])

        selected[dname] = chosen
        coverage[dname] = {
            "total_available": len(pool),
            "selected": len(chosen),
            "entity_clips_available": len(ent_clips),
            "entity_clips_selected": sum(1 for c in chosen if c["has_entity"]),
            "spoken_numeral_clips_available": sum(1 for c in pool if c["has_spoken_numeral"]),
            "spoken_numeral_clips_selected": sum(1 for c in chosen if c["has_spoken_numeral"]),
        }
    return selected, coverage


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[idx], 4)


def _score_clip_entities(clip: dict, asr_text: str, mt_text: str) -> dict:
    src, tgt = clip["src"], clip["tgt"]
    gold_text = clip["gold_text"]
    gold_ref = clip["gold_ref"]
    src_expected = [t.raw for t in entity_scorer.extract_entities(gold_text, lang=src)]
    ref_expected = [t.raw for t in entity_scorer.extract_entities(gold_ref, lang=tgt)]
    asr_score = entity_scorer.score_text_entities(gold_text, asr_text, expected_entities=src_expected, lang=src)
    mt_vs_ref_score = entity_scorer.score_text_entities(gold_ref, mt_text, expected_entities=ref_expected, lang=tgt)
    asr_tok_expected = [t.raw for t in entity_scorer.extract_entities(asr_text, lang=src)]
    mt_vs_asr_score = entity_scorer.score_text_entities(asr_text, mt_text, expected_entities=asr_tok_expected, lang=tgt)
    return {
        "asr_vs_gold_text": {k: asr_score[k] for k in ("correct", "wrong", "unknown", "coverage", "labeled_expected", "all_labeled_correct")},
        "mt_vs_gold_ref": {k: mt_vs_ref_score[k] for k in ("correct", "wrong", "unknown", "coverage", "labeled_expected", "all_labeled_correct")},
        "mt_vs_asr_entities": {k: mt_vs_asr_score[k] for k in ("correct", "wrong", "unknown", "coverage", "labeled_expected", "all_labeled_correct")},
    }


def _aggregate_target_audit(segments: list[dict]) -> dict:
    counts = Counter({"total": 0, "preserved": 0, "mismatch": 0, "unverifiable": 0})
    mismatches = []
    unverifiable = []
    for seg in segments:
        ta = seg.get("target_audit") or {}
        c = ta.get("counts") or {}
        for k in ("total", "preserved", "mismatch", "unverifiable"):
            counts[k] += int(c.get(k, 0))
        mismatches.extend(ta.get("mismatch") or [])
        unverifiable.extend(ta.get("unverifiable") or [])
    return {
        "counts": dict(counts),
        "mismatch": mismatches,
        "unverifiable": unverifiable,
    }


def _aggregate_tts_format(segments: list[dict]) -> dict:
    planned = 0
    attempted = 0
    synthesized = 0
    oov_delta = 0
    warnings = []
    unsupported = []
    for seg in segments:
        tf = seg.get("tts_format") or {}
        if isinstance(tf, dict):
            chunks = tf.get("planned_chunks")
            if isinstance(chunks, list):
                planned += len(chunks)
            elif isinstance(chunks, int):
                planned += chunks
            att = tf.get("attempted_chunks")
            if isinstance(att, list):
                attempted += len(att)
            elif isinstance(att, int):
                attempted += att
            syn = tf.get("synthesized_chunks")
            if isinstance(syn, list):
                synthesized += len(syn)
            elif isinstance(syn, int):
                synthesized += syn
            oov_delta += int(tf.get("oov_delta") or 0)
            for w in tf.get("warnings") or []:
                warnings.append(w)
            for u in tf.get("unsupported") or []:
                unsupported.append(u)
    return {
        "planned_chunks": planned,
        "attempted_chunks": attempted,
        "synthesized_chunks": synthesized,
        "oov_delta": oov_delta,
        "warnings": warnings,
        "unsupported": unsupported,
    }


def run_eval(
    *,
    btc_root: Path,
    out_dir: Path,
    mode: str,
    per_dir: int,
    seed: int,
    backend: str,
    models_path: Path,
    vad_profile: str | None,
    ort_policy: str | None,
    tts_format: str,
    note: str | None = None,
) -> dict:
    os.environ["S2S_NEURAL_TN"] = "0"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_out_root = out_dir / "wavs"
    wav_out_root.mkdir(parents=True, exist_ok=True)

    by_dir = discover_btc_clips(btc_root)
    selected, coverage = select_clips(by_dir, mode=mode, per_dir=per_dir, seed=seed)

    reg = Registry(
        str(models_path),
        backend=backend,
        ort_policy=ort_policy,
        vad_profile=vad_profile,
        source_mode="0",
        tts_format=tts_format,
    )

    dir_summaries: dict[str, dict] = {}
    all_records: list[dict] = []
    eval_t0 = time.perf_counter()

    for dname, src, tgt in DIRECTIONS:
        clips = selected.get(dname, [])
        tr = Translator(reg, src, tgt, vad_profile=reg.vad_profile)
        dir_wav_dir = wav_out_root / dname
        dir_wav_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        lat_total: list[float] = []
        lat_asr: list[float] = []
        lat_mt: list[float] = []
        lat_tts: list[float] = []
        sg_audit = Counter({"total": 0, "preserved": 0, "mismatch": 0, "unverifiable": 0})
        ent_asr = Counter({"correct": 0, "wrong": 0, "unknown": 0})
        ent_mt = Counter({"correct": 0, "wrong": 0, "unknown": 0})
        sg_status_counts: Counter[str] = Counter()
        tts_chunk_counts = Counter({"planned": 0, "attempted": 0, "synthesized": 0})
        errors = 0

        for idx, clip in enumerate(clips, 1):
            t0 = time.perf_counter()
            out_wav_path = dir_wav_dir / f"{clip['id']}_out.wav"
            try:
                wav = load_audio(clip["in_wav"])
                audio_in_s = len(wav) / SR
                segs = tr.translate(wav)
                out_audio, sr = concat_results(segs)
                save_wav(str(out_wav_path), out_audio, sr)
                wall_s = time.perf_counter() - t0
                audio_out_s = (len(out_audio) / sr) if sr else 0.0

                clean_segs = [{k: v for k, v in s.items() if k != "wav"} for s in segs]
                asr_before = " ".join((s.get("asr_before_normalization") or "").strip() for s in clean_segs).strip()
                asr_text = " ".join((s.get("asr") or "").strip() for s in clean_segs).strip()
                mt_text = ("" if tgt == "zh" else " ").join((s.get("mt") or "").strip() for s in clean_segs).strip()
                tts_text = ("" if tgt == "zh" else " ").join((s.get("tts_text") or "").strip() for s in clean_segs).strip()

                asr_s = sum(float((s.get("timing") or {}).get("asr_s", 0.0)) for s in clean_segs)
                norm_s = sum(float((s.get("timing") or {}).get("source_normalize_s", 0.0)) for s in clean_segs)
                mt_s = sum(float((s.get("timing") or {}).get("mt_s", 0.0)) for s in clean_segs)
                tts_s = sum(float((s.get("timing") or {}).get("tts_s", 0.0)) for s in clean_segs)
                pipe_total_s = sum(float((s.get("timing") or {}).get("total_s", 0.0)) for s in clean_segs)

                ta = _aggregate_target_audit(clean_segs)
                tf_agg = _aggregate_tts_format(clean_segs)
                ent_scores = _score_clip_entities(clip, asr_text, mt_text)

                sg_statuses = [(s.get("source_guard") or {}).get("status", "unknown") for s in clean_segs]
                for st in sg_statuses:
                    sg_status_counts[st] += 1
                for k in ("total", "preserved", "mismatch", "unverifiable"):
                    sg_audit[k] += ta["counts"][k]
                for k in ("correct", "wrong", "unknown"):
                    ent_asr[k] += ent_scores["asr_vs_gold_text"][k]
                    ent_mt[k] += ent_scores["mt_vs_gold_ref"][k]
                tts_chunk_counts["planned"] += tf_agg["planned_chunks"]
                tts_chunk_counts["attempted"] += tf_agg["attempted_chunks"]
                tts_chunk_counts["synthesized"] += tf_agg["synthesized_chunks"]

                lat_total.append(pipe_total_s)
                lat_asr.append(asr_s)
                lat_mt.append(mt_s)
                lat_tts.append(tts_s)

                rec = {
                    "id": clip["id"],
                    "dir": dname,
                    "src": src,
                    "tgt": tgt,
                    "status": "ok",
                    "has_entity": clip["has_entity"],
                    "has_spoken_numeral": clip["has_spoken_numeral"],
                    "in_wav": clip["in_wav"],
                    "out_wav": str(out_wav_path),
                    "gold_text": clip["gold_text"],
                    "gold_ref": clip["gold_ref"],
                    "asr_before_normalization": asr_before,
                    "asr": asr_text,
                    "mt": mt_text,
                    "tts_text": tts_text,
                    "n_segments": len(clean_segs),
                    "audio_in_s": round(audio_in_s, 4),
                    "audio_out_s": round(audio_out_s, 4),
                    "timing": {
                        "asr_s": round(asr_s, 4),
                        "source_normalize_s": round(norm_s, 4),
                        "mt_s": round(mt_s, 4),
                        "tts_s": round(tts_s, 4),
                        "total_s": round(pipe_total_s, 4),
                        "wall_s": round(wall_s, 4),
                    },
                    "source_guard_statuses": sg_statuses,
                    "target_audit": ta,
                    "tts_format_summary": tf_agg,
                    "entity_scores": ent_scores,
                    "segments": clean_segs,
                }
            except Exception as exc:
                errors += 1
                rec = {
                    "id": clip["id"],
                    "dir": dname,
                    "src": src,
                    "tgt": tgt,
                    "status": "error",
                    "error": repr(exc),
                    "has_entity": clip["has_entity"],
                    "has_spoken_numeral": clip["has_spoken_numeral"],
                    "in_wav": clip["in_wav"],
                    "out_wav": None,
                    "gold_text": clip["gold_text"],
                    "gold_ref": clip["gold_ref"],
                }
            records.append(rec)
            all_records.append(rec)
            if idx % 10 == 0 or idx == len(clips):
                print(f"[{dname}] {idx}/{len(clips)} done (last={clip['id']} status={rec['status']})", flush=True)

        ok_recs = [r for r in records if r["status"] == "ok"]
        dir_summaries[dname] = {
            "direction": dname,
            "src": src,
            "tgt": tgt,
            "coverage": coverage[dname],
            "clips_ok": len(ok_recs),
            "clips_error": errors,
            "latency_s": {
                "mean_total_s": round(statistics.mean(lat_total), 4) if lat_total else None,
                "p50_total_s": _percentile(lat_total, 50),
                "p95_total_s": _percentile(lat_total, 95),
                "mean_asr_s": round(statistics.mean(lat_asr), 4) if lat_asr else None,
                "mean_mt_s": round(statistics.mean(lat_mt), 4) if lat_mt else None,
                "mean_tts_s": round(statistics.mean(lat_tts), 4) if lat_tts else None,
                "sum_total_s": round(sum(lat_total), 4),
            },
            "source_to_mt_target_audit": dict(sg_audit),
            "source_guard_statuses": dict(sg_status_counts),
            "entity_scorer_asr_vs_gold_text": dict(ent_asr),
            "entity_scorer_mt_vs_gold_ref": dict(ent_mt),
            "tts_chunks": dict(tts_chunk_counts),
        }

    wall_total_s = time.perf_counter() - eval_t0
    summary = {
        "eval_id": "btc_5dirs_eval",
        "mode": mode,
        "seed": seed,
        "per_dir_target": per_dir,
        "note": note,
        "wall_total_s": round(wall_total_s, 3),
        "backend": backend,
        "models_path": str(models_path),
        "effective_profile": reg.profile,
        "env_S2S_NEURAL_TN": os.environ.get("S2S_NEURAL_TN"),
        "code_fingerprint": runtime_code_fingerprint(),
        "ort_sessions": reg.graph_telemetry(),
        "coverage_by_direction": coverage,
        "directions": dir_summaries,
        "records": all_records,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = out_dir / "REPORT.md"
    report_path.write_text(render_markdown_report(summary, out_dir), encoding="utf-8")
    print(f"WROTE {summary_path} and {report_path} in {wall_total_s:.2f}s", flush=True)
    return summary


def _pick_examples(records: list[dict], limit: int = 3) -> list[dict]:
    ok = [r for r in records if r["status"] == "ok"]
    with_ent = [r for r in ok if r["has_entity"] or (r.get("target_audit", {}).get("counts", {}).get("total", 0) > 0)]
    spoken = [r for r in ok if r not in with_ent and r["has_spoken_numeral"]]
    rest = [r for r in ok if r not in with_ent and r not in spoken]
    ordered = with_ent + spoken + rest
    return ordered[:limit]


def render_markdown_report(summary: dict, out_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# BTC 5-Direction End-to-End Evaluation Report (`source_mode=0`)")
    lines.append("")
    lines.append(f"- **Mode:** `{summary['mode']}` (seed=`{summary['seed']}`, wall_total=`{summary['wall_total_s']}s`)")
    if summary.get("note"):
        lines.append(f"- **Coverage / Timing Note:** {summary['note']}")
    prof = summary["effective_profile"]
    lines.append(
        f"- **Runtime profile:** `source_mode={prof['source_mode']}` (`S2S_NEURAL_TN={summary['env_S2S_NEURAL_TN']}`), "
        f"`tts_format={prof['tts_format']}`, `vad_profile={prof['vad_profile']}`, `ort_policy={prof['ort_policy']}`, `backend={summary['backend']}`"
    )
    fp = summary["code_fingerprint"]
    lines.append(f"- **Code fingerprint:** `git_head={fp.get('git_head')}`, `s2s_edge_sha256={fp.get('s2s_edge_sha256')}`")
    lines.append("")
    lines.append("## 1. Per-Direction Coverage, Latency, and Entity Preservation")
    lines.append("")
    lines.append(
        "| Dir | Selected / Total | Entity Clips | OK / Err | Mean Total (s) | P50 / P95 (s) | Mean ASR / MT / TTS (s) | "
        "TargetAudit (pres/mis/unv/tot) | ASR vs Gold (c/w/u) | MT vs Ref (c/w/u) | TTS Chunks (plan/att/syn) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")

    for dname, _, _ in DIRECTIONS:
        ds = summary["directions"][dname]
        cov = ds["coverage"]
        lat = ds["latency_s"]
        ta = ds["source_to_mt_target_audit"]
        ea = ds["entity_scorer_asr_vs_gold_text"]
        em = ds["entity_scorer_mt_vs_gold_ref"]
        tc = ds["tts_chunks"]
        lines.append(
            f"| `{dname}` | {cov['selected']} / {cov['total_available']} | {cov['entity_clips_selected']} / {cov['entity_clips_available']} | "
            f"{ds['clips_ok']} / {ds['clips_error']} | {lat['mean_total_s']} | {lat['p50_total_s']} / {lat['p95_total_s']} | "
            f"{lat['mean_asr_s']} / {lat['mean_mt_s']} / {lat['mean_tts_s']} | "
            f"{ta['preserved']}/{ta['mismatch']}/{ta['unverifiable']}/{ta['total']} | "
            f"{ea['correct']}/{ea['wrong']}/{ea['unknown']} | "
            f"{em['correct']}/{em['wrong']}/{em['unknown']} | "
            f"{tc['planned']}/{tc['attempted']}/{tc['synthesized']} |"
        )

    lines.append("")
    lines.append("## 2. Sample Examples per Direction (ASR / MT / tts_text / WAV)")
    lines.append("")
    for dname, _, _ in DIRECTIONS:
        lines.append(f"### `{dname}`")
        recs = [r for r in summary["records"] if r["dir"] == dname]
        for r in _pick_examples(recs, limit=4):
            rel_wav = r["out_wav"]
            if rel_wav:
                try:
                    rel_wav = str(Path(rel_wav).relative_to(ROOT))
                except ValueError:
                    pass
            ta_c = r.get("target_audit", {}).get("counts", {})
            es_m = r.get("entity_scores", {}).get("mt_vs_gold_ref", {})
            es_a = r.get("entity_scores", {}).get("asr_vs_gold_text", {})
            lines.append(f"- **`{r['id']}`** (`total={r['timing']['total_s']}s`, `in={r['audio_in_s']}s`, `out={r['audio_out_s']}s`, `target_audit={ta_c}`, `asr_ent={es_a}`, `mt_ent={es_m}`)")
            lines.append(f"  - `gold_text`: {r['gold_text']}")
            lines.append(f"  - `asr_raw`: {r['asr_before_normalization']}")
            lines.append(f"  - `asr`: {r['asr']}")
            lines.append(f"  - `mt`: {r['mt']}")
            lines.append(f"  - `tts_text`: {r['tts_text']}")
            lines.append(f"  - `wav`: `{rel_wav}`")
        lines.append("")

    mismatches = [
        r for r in summary["records"]
        if r["status"] == "ok" and (
            r["target_audit"]["counts"]["mismatch"] > 0
            or r["entity_scores"]["mt_vs_gold_ref"]["wrong"] > 0
        )
    ]
    lines.append("## 3. Entity Mismatches / Warnings Observed")
    lines.append("")
    if not mismatches:
        lines.append("No target entity mismatches or wrong entity scores observed in selected clips.")
    else:
        for r in mismatches[:20]:
            lines.append(
                f"- `{r['dir']}` / `{r['id']}`: `target_audit.mismatch={r['target_audit']['mismatch']}`, "
                f"`mt_vs_gold_ref={r['entity_scores']['mt_vs_gold_ref']}` | `asr={r['asr']}` -> `mt={r['mt']}`"
            )
    lines.append("")
    lines.append(f"Generated WAV directory: `{out_dir / 'wavs'}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate 5-direction BTC datasets under S2S_NEURAL_TN=0.")
    ap.add_argument("--btc-root", type=Path, default=ROOT.parent)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/btc_5dirs_eval")
    ap.add_argument("--mode", choices=("smoke", "sampled", "full"), default="sampled")
    ap.add_argument("--per-dir", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=("cpu", "npu"), default="cpu")
    ap.add_argument("--models", type=Path, default=ROOT / "models/models.cpu.json")
    ap.add_argument("--vad-profile", default=None)
    ap.add_argument("--ort-policy", default=None)
    ap.add_argument("--tts-format", default="hybrid")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    run_eval(
        btc_root=args.btc_root,
        out_dir=args.out_dir,
        mode=args.mode,
        per_dir=args.per_dir,
        seed=args.seed,
        backend=args.backend,
        models_path=args.models,
        vad_profile=args.vad_profile,
        ort_policy=args.ort_policy,
        tts_format=args.tts_format,
        note=args.note,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
