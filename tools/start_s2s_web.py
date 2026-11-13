"""Start or safely reuse the user-facing local OneVoice web server."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, tempfile, time, urllib.request

root = Path(__file__).resolve().parents[1]
if str(root / "s2s_edge") not in sys.path:
    sys.path.insert(0, str(root / "s2s_edge"))

from s2s_edge.neural_textnorm import default_hf_home as _default_hf_home, default_source_mode
from s2s_edge.registry import resolve_runtime_profile, runtime_code_fingerprint

PROFILE_KEYS = ("source_mode", "vad_profile", "ort_policy", "tts_format")


class LauncherError(RuntimeError):
    """Explicit launcher failure carrying a deterministic non-zero exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)


def _norm_env_arg(val):
    if val in (None, ""):
        return None
    return str(val)


def build_launch_env(
    base_env=None,
    vad_profile=None,
    ort_policy=None,
    source_mode=None,
    tts_format=None,
    backend=None,
    models=None,
):
    """Build subprocess env honoring caller S2S_NEURAL_TN/VAD/ORT/TTS settings without forcing 1 over 0."""
    env = dict(os.environ if base_env is None else base_env)
    eff_backend = _norm_env_arg(backend) or _norm_env_arg(env.get("S2S_BACKEND")) or "cpu"
    if eff_backend not in ("cpu", "npu"):
        raise ValueError(f"Unsupported backend={eff_backend!r}; expected 'cpu' or 'npu'.")
    if backend is not None:
        env["S2S_BACKEND"] = eff_backend
    if models is not None:
        env["S2S_MODELS"] = str(models)
    if vad_profile is not None:
        env["S2S_VAD_PROFILE"] = str(vad_profile)
    if ort_policy is not None:
        env["S2S_ORT_POLICY"] = str(ort_policy)
    if source_mode is not None:
        env["S2S_NEURAL_TN"] = str(source_mode)
    if tts_format is not None:
        env["S2S_TTS_FORMAT"] = str(tts_format)
    env.setdefault("S2S_NEURAL_TN", default_source_mode(eff_backend))
    env.setdefault("S2S_TTS_FORMAT", "hybrid")
    env.setdefault("S2S_VAD_PROFILE", "legacy")
    env.setdefault("S2S_ORT_POLICY", "auto")
    env.setdefault("HF_HOME", _default_hf_home())
    requested_profile_from_env(env)
    return env


def requested_profile_from_env(env):
    """Resolve and validate the effective runtime profile purely from the supplied mapping."""
    return resolve_runtime_profile(
        vad_profile=_norm_env_arg(env.get("S2S_VAD_PROFILE")),
        ort_policy=_norm_env_arg(env.get("S2S_ORT_POLICY")),
        source_mode=_norm_env_arg(env.get("S2S_NEURAL_TN")),
        tts_format=_norm_env_arg(env.get("S2S_TTS_FORMAT")),
    )


def disk_code_fingerprint():
    """Return current on-disk fingerprint covering s2s_edge modules, tools/s2s_web.py, and git HEAD."""
    return runtime_code_fingerprint(extra_files=[str(root / "tools/s2s_web.py")])


def fetch_health(url, timeout=2):
    """Fetch /api/health JSON payload when server responds healthy, else return None."""
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=timeout) as r:
            data = json.load(r)
        if isinstance(data, dict) and data.get("ok") is True:
            return data
        return None
    except Exception:
        return None


def ready(url):
    return fetch_health(url) is not None


def _norm_models_path(val):
    if not val:
        return None
    p = Path(str(val))
    if not p.is_absolute():
        p = root / p
    return str(p.resolve())


def _terminate_child(proc):
    """Best-effort cleanup of a spawned child process on startup verification failure."""
    try:
        if proc is not None and proc.poll() is None and hasattr(proc, "terminate"):
            proc.terminate()
    except Exception:
        pass


def compare_server_state(
    health,
    requested_profile,
    disk_fp,
    requested_backend=None,
    requested_models=None,
):
    """Verify server /api/health matches requested profile, backend/models, and on-disk code fingerprint."""
    reasons = []
    if not isinstance(health, dict) or health.get("ok") is not True:
        return False, ["health_not_ok"]

    eff = health.get("effective_profile")
    if not isinstance(eff, dict):
        reasons.append("missing_effective_profile")

    for key in PROFILE_KEYS:
        want = requested_profile.get(key)
        got_top = health.get(key)
        if got_top != want:
            reasons.append(f"profile_mismatch:{key}:want={want!r},got={got_top!r}")
        if isinstance(eff, dict):
            got_eff = eff.get(key)
            if got_eff != want:
                reasons.append(f"effective_profile_mismatch:{key}:want={want!r},got={got_eff!r}")

    loaded_fp = health.get("loaded_code_fingerprint")
    if not isinstance(loaded_fp, dict):
        reasons.append("missing_loaded_code_fingerprint")
    else:
        for fp_key in ("s2s_edge_sha256", "server_files_sha256", "git_head"):
            if fp_key not in loaded_fp:
                reasons.append(f"missing_fingerprint_field:{fp_key}")
            elif loaded_fp.get(fp_key) != disk_fp.get(fp_key):
                reasons.append(
                    f"stale_code:{fp_key}:loaded={loaded_fp.get(fp_key)!r},disk={disk_fp.get(fp_key)!r}"
                )

    if requested_backend is not None and (requested_backend != "cpu" or "backend" in health):
        got_backend = health.get("backend")
        if got_backend != requested_backend:
            reasons.append(f"backend_mismatch:want={requested_backend!r},got={got_backend!r}")

    if requested_models is not None:
        want_models = _norm_models_path(requested_models)
        got_models = _norm_models_path(health.get("models"))
        if got_models != want_models:
            reasons.append(f"models_mismatch:want={want_models!r},got={got_models!r}")
    elif health.get("models") is not None:
        want_models = _norm_models_path(root / "models/models.cpu.json")
        got_models = _norm_models_path(health.get("models"))
        if got_models != want_models:
            reasons.append(f"models_mismatch:want={want_models!r},got={got_models!r}")

    return len(reasons) == 0, reasons


def launch_or_reuse(
    args,
    *,
    base_env=None,
    health_fn=None,
    fingerprint_fn=None,
    popen_fn=None,
    kill_fn=None,
    sleep_fn=None,
):
    """Validate requested config, safely reuse matching server or reload idle server cleanly."""
    health_fn = fetch_health if health_fn is None else health_fn
    fingerprint_fn = disk_code_fingerprint if fingerprint_fn is None else fingerprint_fn
    popen_fn = subprocess.Popen if popen_fn is None else popen_fn
    kill_fn = os.kill if kill_fn is None else kill_fn
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn

    arg_backend = getattr(args, "backend", None)
    arg_models = getattr(args, "models", None)
    env = build_launch_env(
        base_env=base_env,
        vad_profile=getattr(args, "vad_profile", None),
        ort_policy=getattr(args, "ort_policy", None),
        source_mode=getattr(args, "source_mode", None),
        tts_format=getattr(args, "tts_format", None),
        backend=arg_backend,
        models=arg_models,
    )
    req_backend = _norm_env_arg(env.get("S2S_BACKEND")) or "cpu"
    req_models = _norm_env_arg(env.get("S2S_MODELS"))
    req_profile = requested_profile_from_env(env)
    url = f"http://localhost:{args.port}"
    disk_fp = None

    health = health_fn(url)
    was_healthy = health is not None
    if was_healthy:
        disk_fp = fingerprint_fn()
        matches, reasons = compare_server_state(
            health,
            req_profile,
            disk_fp,
            requested_backend=req_backend,
            requested_models=req_models,
        )
        if matches and not getattr(args, "reload", False):
            return {"action": "reused", "url": url, "profile": req_profile, "reasons": []}
        if not getattr(args, "reload", False):
            raise LauncherError(
                2,
                f"Existing server at {url} does not match requested profile or disk code "
                f"(refusing unsafe reuse; pass --reload to replace when idle): {'; '.join(reasons)}",
            )

        pending = health.get("pending")
        if not isinstance(pending, int) or isinstance(pending, bool) or pending != 0:
            raise LauncherError(
                3,
                f"Refusing --reload for {url}: server reports pending={pending!r} (must be 0).",
            )
        pid = health.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise LauncherError(
                3,
                f"Refusing --reload for {url}: missing or invalid server pid={pid!r}.",
            )

        kill_fn(pid, signal.SIGTERM)
        stopped = False
        for _ in range(40):
            if health_fn(url) is None:
                stopped = True
                break
            sleep_fn(0.25)
        if not stopped:
            raise LauncherError(4, f"Server pid={pid} at {url} did not stop after SIGTERM.")

    if disk_fp is None:
        disk_fp = fingerprint_fn()

    cmd = [
        sys.executable,
        "-u",
        str(root / "tools/s2s_web.py"),
        "--port",
        str(args.port),
        "--vad-profile",
        req_profile["vad_profile"],
        "--ort-policy",
        req_profile["ort_policy"],
        "--source-mode",
        req_profile["source_mode"],
        "--tts-format",
        req_profile["tts_format"],
    ]
    if req_backend != "cpu" or arg_backend is not None or "S2S_BACKEND" in env:
        cmd.extend(["--backend", req_backend])
    if req_models is not None:
        cmd.extend(["--models", req_models])
    if getattr(args, "keep_loaded", False):
        cmd.append("--keep-loaded")
    if args.out_dir:
        cmd.extend(["--out-dir", str(args.out_dir)])
    log_path = Path(args.log_file) if args.log_file else (root / "logs/s2s_web.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        proc = popen_fn(
            cmd,
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    started = False
    for _ in range(40):
        rc = proc.poll()
        if rc is not None:
            foreign = health_fn(url)
            detail = (
                f" (another process pid={foreign.get('pid')!r} responded at {url})"
                if isinstance(foreign, dict)
                else ""
            )
            raise LauncherError(5, f"Web server exited (rc={rc}){detail}; inspect {log_path}")

        spawned_health = health_fn(url)
        if spawned_health is not None:
            rc = proc.poll()
            if rc is not None:
                raise LauncherError(
                    5,
                    f"Web server exited (rc={rc}) while another process responded at {url}; inspect {log_path}",
                )
            seen_pid = spawned_health.get("pid")
            if seen_pid != proc.pid:
                _terminate_child(proc)
                raise LauncherError(
                    2,
                    f"Foreign server pid={seen_pid!r} responded at {url} while waiting for spawned pid={proc.pid}; "
                    "refusing unsafe reuse or server.pid overwrite.",
                )
            matches, reasons = compare_server_state(
                spawned_health,
                req_profile,
                disk_fp,
                requested_backend=req_backend,
                requested_models=req_models,
            )
            if not matches:
                _terminate_child(proc)
                raise LauncherError(
                    2,
                    f"Spawned server pid={proc.pid} at {url} failed readiness verification: {'; '.join(reasons)}",
                )
            started = True
            break
        sleep_fn(0.25)

    if not started:
        _terminate_child(proc)
        raise LauncherError(5, f"Web server pid={proc.pid} at {url} did not become ready; inspect {log_path}")

    pid_path = None
    if args.out_dir:
        pid_path = Path(args.out_dir) / "server.pid"
    elif args.port == 8766:
        pid_path = root / "results/mic_web/server.pid"
    if pid_path is not None:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid))

    return {
        "action": "reloaded" if was_healthy else "started",
        "url": url,
        "profile": req_profile,
        "pid": proc.pid,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Start the OneVoice web server.")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--models", default=None, help="Path to models JSON (e.g. models/models.npu.json for AIBOX 8550).")
    ap.add_argument("--backend", default=None, choices=["cpu", "npu"], help="Execution backend (cpu or npu).")
    ap.add_argument("--keep-loaded", action="store_true", help="Keep Registry/Translator loaded across requests.")
    ap.add_argument("--vad-profile", default=None, choices=["legacy", "upstream", "hybrid"])
    ap.add_argument("--ort-policy", default=None, choices=["auto", "basic", "disable"])
    ap.add_argument("--source-mode", default=None, choices=["0", "1"])
    ap.add_argument("--tts-format", default=None, choices=["hybrid", "legacy"])
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Safely SIGTERM a matching-PID idle (pending==0) server on mismatch and restart.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        res = launch_or_reuse(args)
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(exc.code)
    except ValueError as exc:
        print(f"ERROR: invalid runtime configuration: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    print(res["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
