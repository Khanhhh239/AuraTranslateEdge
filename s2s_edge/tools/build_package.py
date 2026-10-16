# -*- coding: utf-8 -*-
"""Assemble the offline installation package (a folder, zip it afterwards).
  python tools/build_package.py --models-dir <dir with <name>/<name>.bin + manifest.json> --wheelhouse <dir> --out dist/s2s_edge_pkg
Result: code + assets + tools + golden + samples + models/ (QNN context binaries + models.json) + wheelhouse/ + install.sh + run_demo.sh."""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# component -> manifest entry name
MAP = {"asr_vi": "asr_vi_zip150_ctc_fp16", "asr_xx_frontend": "asr_xx_sensevoice_frontend_v3", "asr_xx_encoder": "asr_xx_sensevoice_encoder_q",
       "mt_enc": "mt_enc_nllb", "mt_dec": "mt_dec_nllb", "tts_vi": "tts_vi_piper_fullg8", "tts_zh": "tts_zh_melo_one", "tts_enko": "tts_enko_supertonic_full_v3"}

INSTALL = """#!/usr/bin/env bash
# Offline install (no network needed): Python 3.12 on Ubuntu 24.04 aarch64 (Dragonwing IQ-9075).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links wheelhouse -r requirements.txt
pip install --no-index --find-links wheelhouse qai_appbuilder
python3 -c "import qai_appbuilder, os; print('qai_appbuilder', os.path.dirname(qai_appbuilder.__file__))"
echo "ADSP_LIBRARY_PATH is set automatically by qai_appbuilder (bundled Hexagon skel libs)."
echo "Next: bash tools/recon.sh ; python tools/golden.py check --backend npu ; bash run_demo.sh vi en"
"""

RUN = """#!/usr/bin/env bash
# usage: bash run_demo.sh <src> <tgt>      e.g. bash run_demo.sh vi en
cd "$(dirname "$0")"; . .venv/bin/activate
python -m s2s_edge live --src "${1:-vi}" --tgt "${2:-en}" --models models/models.json --backend npu
"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models-dir", required=True); ap.add_argument("--wheelhouse", default=""); ap.add_argument("--out", default="dist/s2s_edge_pkg"); ap.add_argument("--no-models", action="store_true"); ap.add_argument("--eval-dir", default=""); ap.add_argument("--eval-n", type=int, default=10)
    a = ap.parse_args(); out = a.out
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    for d in ("s2s_edge", "assets", "tools", "golden", "samples"):
        if os.path.isdir(os.path.join(HERE, d)): shutil.copytree(os.path.join(HERE, d), os.path.join(out, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("README.md", "requirements.txt"): shutil.copy(os.path.join(HERE, f), out)
    open(os.path.join(out, "install.sh"), "w", newline="\n").write(INSTALL); open(os.path.join(out, "run_demo.sh"), "w", newline="\n").write(RUN)
    if a.wheelhouse and os.path.isdir(a.wheelhouse): shutil.copytree(a.wheelhouse, os.path.join(out, "wheelhouse"))
    if a.eval_dir:                                              # small evaluation subset (FLEURS test audio + FLORES references) for judges
        os.makedirs(os.path.join(out, "eval"))
        for l in ("vi", "en", "zh", "ko"):
            man = json.load(open(os.path.join(a.eval_dir, f"manifest_{l}.json"), encoding="utf-8"))[: a.eval_n]; os.makedirs(os.path.join(out, "eval", l), exist_ok=True)
            for it in man: shutil.copy(os.path.join(a.eval_dir, it["path"]), os.path.join(out, "eval", l, os.path.basename(it["path"]))); it["path"] = f"{l}/" + os.path.basename(it["path"])
            json.dump(man, open(os.path.join(out, "eval", f"manifest_{l}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not a.no_models:
        man = json.load(open(os.path.join(a.models_dir, "manifest.json"))); cfg = {}; os.makedirs(os.path.join(out, "models"))
        for comp, name in MAP.items():
            if name not in man: print("MISSING", comp, name); continue
            src = os.path.join(a.models_dir, man[name]["file"]); dst = os.path.join("models", os.path.basename(src)); shutil.copy(src, os.path.join(out, dst))
            cfg[comp] = {"ctx": os.path.basename(src), "name": comp, "job": man[name]["job"]}
        json.dump(cfg, open(os.path.join(out, "models", "models.json"), "w"), indent=1)
        print("models:", {k: v["ctx"] for k, v in cfg.items()})
    tot = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs); print(f"package {out}: {tot / 1e9:.2f} GB")


if __name__ == "__main__": main()
