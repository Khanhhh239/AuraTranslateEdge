# -*- coding: utf-8 -*-
"""Audio I/O and utterance segmentation (pure numpy; VAD is energy based so it runs anywhere, no extra model)."""
import os, subprocess, shutil, wave, numpy as np

SR = 16000
# ALSA devices for the live demo. The KIMQ8550 box has no built-in mic/speaker, so the USB device is often NOT the ALSA default:
# find it with `arecord -l` / `aplay -l` (tools/audio_check.sh) and set e.g. S2S_MIC=plughw:1,0 S2S_SPK=plughw:1,0.
_MIC = ["-D", os.environ["S2S_MIC"]] if os.environ.get("S2S_MIC") else []
_SPK = ["-D", os.environ["S2S_SPK"]] if os.environ.get("S2S_SPK") else []


def load_audio(path, sr=SR):
    import soundfile as sf
    w, r = sf.read(path, dtype="float32", always_2d=False)
    if w.ndim > 1: w = w.mean(axis=1)
    if r != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(r), sr); w = resample_poly(w, sr // g, int(r) // g).astype(np.float32)
    return w


def save_wav(path, x, sr):
    x = np.clip(np.asarray(x, np.float32), -1, 1)
    with wave.open(path, "wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(int(sr)); f.writeframes((x * 32767).astype(np.int16).tobytes())


def record(seconds, sr=SR):
    """Capture from the default microphone (ALSA arecord, or sounddevice if available)."""
    if shutil.which("arecord"):
        p = subprocess.run(["arecord", "-q", *_MIC, "-f", "S16_LE", "-r", str(sr), "-c", "1", "-d", str(int(seconds)), "-t", "raw"], capture_output=True, check=True)
        return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0
    import sounddevice as sd
    return sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", blocking=True)[:, 0]


def play(x, sr):
    import tempfile, os
    p = tempfile.mktemp(suffix=".wav"); save_wav(p, x, sr)
    for c in (["aplay", "-q", *_SPK, p], ["paplay", p], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", p]):
        if shutil.which(c[0]): subprocess.run(c); break
    os.remove(p)


SEG_MAX_S = float(os.environ.get("S2S_SEG_MAX_S", "14.0"))
LISTEN_MIN_RMS = float(os.environ.get("S2S_LISTEN_MIN_RMS", "2e-4"))


def active_rms(x, sr=SR, frame_ms=20):
    """RMS over non-silent 20 ms frames so pauses do not dilute active speech loudness."""
    if not len(x):
        return 0.0
    n = max(1, int(sr * frame_ms / 1000))
    nf = len(x) // n
    if nf == 0:
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    e = np.sqrt((x[: nf * n].reshape(nf, n).astype(np.float64) ** 2).mean(1))
    pk = float(e.max())
    if pk < 1e-5:
        return 0.0
    mask = e >= max(pk * 0.05, 1e-4)
    vals = e[mask] if mask.any() else e
    return float(np.sqrt(np.mean(vals ** 2)))


def normalize_level(x, target_db=-24.0, max_gain_db=35.0, loud_db=-14.0, peak_ceiling=0.9):
    """Continuous active-RMS loudness normalization bounded by 99.9th-percentile transient peak ceiling."""
    if not len(x):
        return x
    rms = active_rms(x)
    if rms <= 1e-6:
        return x
    target_rms = 10.0 ** (target_db / 20.0)
    loud_rms = 10.0 ** (loud_db / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    if rms < target_rms:
        gain = min(target_rms / rms, max_gain)
    elif rms > loud_rms:
        gain = loud_rms / rms
    else:
        return x
    p999 = float(np.percentile(np.abs(x), 99.9))
    if p999 > 1e-6:
        gain = min(gain, peak_ceiling / p999)
    return np.clip(x.astype(np.float32, copy=False) * np.float32(gain), -1.0, 1.0).astype(np.float32)


VAD_PROFILES = {
    "legacy": {"pad_ms": 150, "min_sil_ms": 500, "merge_gap_s": 1.5, "short_s": 3.0},
    "upstream": {"pad_ms": 120, "min_sil_ms": 320, "merge_gap_s": 0.32, "short_s": 1.1},
    "hybrid": {"head_pad_ms": 250, "tail_pad_ms": 350, "ext_ms": 400, "join_gap_s": 0.8, "break_ms": 320, "min_sil_ms": 200},
}
DEFAULT_VAD_PROFILE = "legacy"
_VAD_MODES = {"hybrid": (0.8, 200), "split": (0.0, 200), "join": (1.5, 400)}


def vad_mode():
    """Return active S2S_VAD_MODE ('hybrid' default, 'split', 'join')."""
    mode = (os.environ.get("S2S_VAD_MODE") or "hybrid").strip().lower()
    return mode if mode in _VAD_MODES else "hybrid"


def get_vad_options(profile: str) -> dict:
    """Return VAD segmentation parameters for the named profile ('legacy', 'upstream', or 'hybrid')."""
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(f"Unsupported vad_profile {profile!r}; expected one of {tuple(VAD_PROFILES)}")
    key = profile.strip().lower()
    if key not in VAD_PROFILES:
        raise ValueError(f"Unsupported vad_profile {profile!r}; expected one of {tuple(VAD_PROFILES)}")
    return dict(VAD_PROFILES[key])


def effective_max_s(asr_max_s, vad_profile=None, max_seg_s=None) -> float:
    """Resolve segment duration cap honoring explicit overrides, S2S_SEG_MAX_S, and hybrid profile default."""
    if max_seg_s is not None:
        return min(float(asr_max_s), float(max_seg_s))
    cap = SEG_MAX_S if ("S2S_SEG_MAX_S" in os.environ or vad_profile == "hybrid") else 12.0
    return min(float(asr_max_s), cap)


def resolve_vad_options(
    vad_profile=None,
    pad_ms=None,
    min_sil_ms=None,
    merge_gap_s=None,
    short_s=None,
    head_pad_ms=None,
    tail_pad_ms=None,
    ext_ms=None,
    join_gap_s=None,
    break_ms=None,
):
    """Resolve effective VAD profile and parameter dict without altering explicit caller values."""
    if vad_profile is None:
        env_prof = os.environ.get("S2S_VAD_PROFILE")
        prof = env_prof if (env_prof is not None and env_prof != "") else DEFAULT_VAD_PROFILE
    else:
        prof = vad_profile
    opts = get_vad_options(prof)
    norm_prof = prof.strip().lower()
    if norm_prof == "hybrid":
        mode_join, mode_sil = _VAD_MODES[vad_mode()]
        opts["join_gap_s"] = mode_join
        opts["min_sil_ms"] = mode_sil
    if pad_ms is not None:
        opts["pad_ms"] = pad_ms
    if min_sil_ms is not None:
        opts["min_sil_ms"] = min_sil_ms
    if merge_gap_s is not None:
        opts["merge_gap_s"] = merge_gap_s
    if short_s is not None:
        opts["short_s"] = short_s
    if head_pad_ms is not None:
        opts["head_pad_ms"] = head_pad_ms
    if tail_pad_ms is not None:
        opts["tail_pad_ms"] = tail_pad_ms
    if ext_ms is not None:
        opts["ext_ms"] = ext_ms
    if join_gap_s is not None:
        if norm_prof in ("legacy", "upstream"):
            if merge_gap_s is None:
                opts["merge_gap_s"] = join_gap_s
            if join_gap_s == 0.0 and short_s is None:
                opts["short_s"] = 0.0
        else:
            opts["join_gap_s"] = join_gap_s
    if break_ms is not None:
        opts["break_ms"] = break_ms
    return norm_prof, opts


def _voiced_spans(e, ceil, floor, thr, frame_ms=20, min_sil_ms=200, ext_ms=400):
    """Find voiced frame spans [i, last+1) with hysteresis release threshold ~30 dB below peak up to `ext_ms`."""
    nf = len(e)
    voiced = e > thr
    rel = np.minimum(thr, np.maximum.reduce([floor * 1.35, ceil * 0.0316, np.full_like(e, 1e-5)]))
    k = max(1, int(min_sil_ms / frame_ms))
    ext = max(0, int(ext_ms / frame_ms))
    out = []
    i = 0
    while i < nf:
        if not voiced[i]:
            i += 1
            continue
        j = i
        last_strong = i
        last_rel = i
        while j < nf:
            if voiced[j]:
                last_strong = j
                last_rel = j
                j += 1
            elif j - last_strong <= k:
                if e[j] > rel[j]:
                    last_rel = j
                j += 1
            elif e[j] > rel[j] and j - last_strong <= ext:
                last_rel = j
                j += 1
            else:
                break
        out.append((i, max(last_strong, last_rel) + 1))
        i = j
    return out


def _split_long_with_breaks(a, b, brs, e, n, sr, max_s, min_s):
    """Split a long span at quietest frames while keeping internal break offsets aligned per chunk."""
    out_spans, out_breaks = [], []
    while (b - a) / sr > max_s:
        lo = a + int(0.6 * max_s * sr)
        hi = a + int(max_s * sr)
        w = e[lo // n: max(lo // n + 1, hi // n)]
        cut = (lo // n + int(np.argmin(w))) * n
        if cut <= a:
            cut = min(b, a + int(max_s * sr))
        out_spans.append((a, cut))
        out_breaks.append([p for p in brs if 0 < p < (cut - a)])
        brs = [p - (cut - a) for p in brs if p > (cut - a)]
        a = cut
    if (b - a) / sr >= min_s:
        out_spans.append((a, b))
        out_breaks.append([p for p in brs if 0 < p < (b - a)])
    return out_spans, out_breaks


def segment(
    x,
    sr=SR,
    max_s=None,
    min_s=0.3,
    frame_ms=20,
    pad_ms=None,
    min_sil_ms=None,
    merge_gap_s=None,
    short_s=None,
    vad_profile=None,
    head_pad_ms=None,
    tail_pad_ms=None,
    ext_ms=None,
    join_gap_s=None,
    break_ms=320,
    return_breaks=False,
):
    """Split a recording into utterances at pauses.
    Resolves vad_profile ('legacy' default, 'upstream', or 'hybrid' with hysteresis + join_gap_s) without altering explicit caller args.
    When `return_breaks=True`, returns `(spans, breaks)` where each break entry lists pause midpoints relative to span start."""
    prof, _vad = resolve_vad_options(
        vad_profile=vad_profile,
        pad_ms=pad_ms,
        min_sil_ms=min_sil_ms,
        merge_gap_s=merge_gap_s,
        short_s=short_s,
        head_pad_ms=head_pad_ms,
        tail_pad_ms=tail_pad_ms,
        ext_ms=ext_ms,
        join_gap_s=join_gap_s,
        break_ms=break_ms,
    )
    if max_s is None:
        max_s = SEG_MAX_S if ("S2S_SEG_MAX_S" in os.environ or prof == "hybrid") else 12.0
    n = int(sr * frame_ms / 1000)
    nf = len(x) // n
    if nf == 0:
        return ([], []) if return_breaks else []
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    e = np.sqrt((x[:nf * n].reshape(nf, n) ** 2).mean(1) + 1e-12)
    if np.percentile(e, 95) < 3e-5:
        return ([], []) if return_breaks else []
    es = uniform_filter1d(e, 3)
    ceil = maximum_filter1d(es, int(3.0 * 1000 / frame_ms))
    floor = np.maximum(minimum_filter1d(es, int(6.0 * 1000 / frame_ms)), 1e-6)
    thr = np.maximum.reduce([floor * 2.5, floor + 0.12 * (ceil - floor), np.full_like(e, 2e-5)])

    use_hysteresis = prof == "hybrid" or any(v is not None for v in (head_pad_ms, tail_pad_ms, ext_ms))
    if use_hysteresis:
        hpad = int(sr * _vad.get("head_pad_ms", 250) / 1000)
        tpad = int(sr * _vad.get("tail_pad_ms", 350) / 1000)
        raw = _voiced_spans(e, ceil, floor, thr, frame_ms=frame_ms, min_sil_ms=_vad["min_sil_ms"], ext_ms=_vad.get("ext_ms", 400))
        if not raw:
            return ([], []) if return_breaks else []
        j_gap = _vad.get("join_gap_s", 0.8)
        b_gap = _vad.get("break_ms", 320) / 1000.0
        merged_raw = []
        for fa, fb in raw:
            sa_raw, sb_raw = fa * n, fb * n
            if merged_raw:
                pa_raw, pb_raw, brs = merged_raw[-1]
                raw_gap_s = (sa_raw - pb_raw) / sr
                if j_gap > 0 and raw_gap_s <= j_gap and (min(len(x), sb_raw + tpad) - max(0, pa_raw - hpad)) / sr <= max_s:
                    if raw_gap_s >= b_gap:
                        brs.append(int((pb_raw + sa_raw) * 0.5))
                    merged_raw[-1] = (pa_raw, sb_raw, brs)
                    continue
            merged_raw.append((sa_raw, sb_raw, []))
        padded_groups = []
        for ra, rb, brs in merged_raw:
            sa = max(0, ra - hpad)
            sb = min(len(x), rb + tpad)
            if padded_groups and sa < padded_groups[-1][1]:
                prev_sa, _prev_sb, prev_ra, prev_rb, prev_brs = padded_groups[-1]
                mid = max(prev_sa, min(sb, (prev_rb + ra) // 2))
                padded_groups[-1] = (prev_sa, mid, prev_ra, prev_rb, prev_brs)
                sa = mid
            padded_groups.append((sa, sb, ra, rb, brs))
        spans, breaks = [], []
        for a, b, _ra, _rb, br_abs in padded_groups:
            br_rel = [p - a for p in br_abs if a < p < b]
            sp_list, br_list = _split_long_with_breaks(a, b, br_rel, e, n, sr, max_s, min_s)
            spans.extend(sp_list)
            breaks.extend(br_list)
        return (spans, breaks) if return_breaks else spans

    # Plan 01 energy segmentation path ('legacy' / 'upstream')
    pad_ms_v = _vad["pad_ms"]
    min_sil_v = _vad["min_sil_ms"]
    merge_gap_v = _vad["merge_gap_s"]
    short_v = _vad["short_s"]
    voiced = e > thr; k = max(1, int(min_sil_v / frame_ms)); out = []; i = 0
    while i < nf:
        if not voiced[i]: i += 1; continue
        j = i; last = i
        while j < nf and j - last <= k:
            if voiced[j]: last = j
            j += 1
        out.append((i, last + 1)); i = j
    pad = int(sr * pad_ms_v / 1000); segs = []
    for a, b in out:
        a, b = max(0, a * n - pad), min(len(x), b * n + pad)
        sp_list, _ = _split_long_with_breaks(a, b, [], e, n, sr, max_s, min_s)
        segs.extend(sp_list)
    b_gap = (_vad.get("break_ms", break_ms) if break_ms is not None else 320) / 1000.0
    merged = []; breaks = []
    for a, b in segs:
        if merged:
            pa, pb = merged[-1]
            if (a - pb) / sr < merge_gap_v and (b - pa) / sr <= max_s and ((pb - pa) / sr < short_v or (b - a) / sr < short_v):
                if (a - pb) / sr >= b_gap:
                    breaks[-1].append(int((pb + a) * 0.5) - pa)
                merged[-1] = (pa, b); continue
        merged.append((a, b)); breaks.append([])
    return (merged, breaks) if return_breaks else merged


def listen(max_s=12.0, silence_s=0.9, start_timeout_s=8.0, sr=SR):
    """Push-to-talk capture without extra dependencies: reads raw 16 kHz mono from `arecord` (or sounddevice) and stops after
    `silence_s` of silence following speech, or at max_s. Returns float32 audio (may be empty if nothing was said)."""
    n = int(sr * 0.02); chunks = []; spoken = False; quiet = 0; floor = None; t = 0.0
    if shutil.which("arecord"):
        p = subprocess.Popen(["arecord", "-q", *_MIC, "-f", "S16_LE", "-r", str(sr), "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        read = lambda: p.stdout.read(n * 2)
        stop = p.terminate
        def frame():
            b = read(); return np.frombuffer(b, np.int16).astype(np.float32) / 32768.0 if b and len(b) == n * 2 else None
    else:
        import sounddevice as sd
        st = sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=n); st.start(); stop = st.stop
        def frame():
            d, _ = st.read(n); return d[:, 0]
    try:
        while t < max_s:
            f = frame()
            if f is None: break
            t += 0.02; chunks.append(f); e = float(np.sqrt((f ** 2).mean() + 1e-10))
            if floor is None or t < 0.3: floor = e if floor is None else min(floor, e) * 0.5 + floor * 0.5
            loud = e > max(floor * 4.0, LISTEN_MIN_RMS)
            if loud: spoken = True; quiet = 0
            elif spoken: quiet += 1
            if spoken and quiet * 0.02 >= silence_s: break
            if not spoken and t >= start_timeout_s: return np.zeros(0, np.float32)
    finally:
        stop()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def concat_results(results, gap_s=0.25):
    """Join per-utterance outputs that may have different sample rates (Piper 22.05 kHz, Melo / Supertonic 44.1 kHz) into one wav. Returns (wave, sr)."""
    from math import gcd
    from scipy.signal import resample_poly
    ws = [(r["wav"], r["sr"]) for r in results if len(r["wav"])]
    if not ws: return np.zeros(0, np.float32), 16000
    sr = max(s for _, s in ws); out = []
    for w, s in ws:
        if s != sr: g = gcd(int(sr), int(s)); w = resample_poly(w, int(sr) // g, int(s) // g).astype(np.float32)
        out += [w, np.zeros(int(sr * gap_s), np.float32)]
    return np.concatenate(out[:-1]), sr
