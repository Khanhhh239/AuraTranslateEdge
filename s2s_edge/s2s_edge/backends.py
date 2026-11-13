# -*- coding: utf-8 -*-
"""Graph backends. A Graph wraps ONE compiled model and is called with named numpy inputs, returning outputs in ONNX output order.
OrtGraph : onnxruntime CPU (development / reference, runs everywhere).
QnnGraph : Qualcomm NPU on the device (context binary produced by AI Hub). Loader chosen at run time, see `QnnGraph.load` (decided after tools/recon.sh)."""
import os, sys, time, json
import numpy as np


class Graph:
    name = "graph"

    def __init__(self):
        self.calls = 0; self.total_s = 0.0

    def run(self, inputs):
        raise NotImplementedError

    def __call__(self, **inputs):
        t = time.perf_counter(); out = self.run({k: np.ascontiguousarray(v) for k, v in inputs.items()})
        self.calls += 1; self.total_s += time.perf_counter() - t; return out

    def release(self):
        """Free the runtime resources (used by the MT fallback after an out-of-memory / runtime error)."""
        pass


ORT_POLICIES = ("auto", "basic", "disable")
DEFAULT_ORT_POLICY = "auto"


def resolve_ort_policy(ort_policy=None) -> str:
    """Resolve explicit ORT session optimization policy ('auto', 'basic', 'disable')."""
    if ort_policy is None:
        env_pol = os.environ.get("S2S_ORT_POLICY")
        pol = env_pol if (env_pol is not None and env_pol != "") else DEFAULT_ORT_POLICY
    else:
        pol = ort_policy
    if not isinstance(pol, str) or not pol.strip():
        raise ValueError(f"Unsupported ort_policy {pol!r}; expected one of {ORT_POLICIES}")
    key = pol.strip().lower()
    if key not in ORT_POLICIES:
        raise ValueError(f"Unsupported ort_policy {pol!r}; expected one of {ORT_POLICIES}")
    return key


def ort_optimization_plan(ort_policy=None, large_or_ext=False):
    """Return (resolved_policy, ordered [(level_name, level_enum), ...]) for ONNX Runtime CPU."""
    import onnxruntime as ort
    resolved = resolve_ort_policy(ort_policy)
    if resolved == "disable" or (resolved == "auto" and large_or_ext):
        plan = [("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL)]
    else:
        plan = [
            ("ORT_ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
            ("ORT_DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ]
    return resolved, plan


def create_cpu_ort_session(path, ort_policy=None, large_or_ext=None, intra_op_num_threads=None):
    """Build a bounded-memory CPU InferenceSession using resolved ORT policy and fallback retries."""
    import onnxruntime as ort
    ext_data = os.path.exists(path + ".data")
    size_bytes = os.path.getsize(path) if os.path.exists(path) else None
    is_large = bool(ext_data or (size_bytes is not None and size_bytes > 200_000_000)) if large_or_ext is None else bool(large_or_ext)
    resolved, plan = ort_optimization_plan(ort_policy=ort_policy, large_or_ext=is_large)

    attempts = []
    effective_level = None
    fallback = False
    session = None
    last_exc = None
    for idx, (level_name, level_enum) in enumerate(plan, start=1):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        if intra_op_num_threads is not None:
            so.intra_op_num_threads = int(intra_op_num_threads)
        so.graph_optimization_level = level_enum
        try:
            session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            attempts.append({"attempt": idx, "level": level_name, "ok": True})
            effective_level = level_name
            fallback = idx > 1
            last_exc = None
            break
        except Exception as exc:
            attempts.append({"attempt": idx, "level": level_name, "ok": False, "error": repr(exc)[:300]})
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return session, attempts, effective_level, fallback, resolved, ext_data, size_bytes, is_large


class OrtGraph(Graph):
    def __init__(self, path, name=None, ort_policy=None):
        super().__init__()
        self.path = path
        (
            self.s,
            self.ort_attempts,
            self.effective_ort_level,
            self.ort_fallback,
            self.ort_policy,
            self.external_data,
            self.size_bytes,
            self.large_or_ext,
        ) = create_cpu_ort_session(path, ort_policy=ort_policy)
        self.name = name or os.path.basename(path)
        self.in_names = [i.name for i in self.s.get_inputs()]
        self.in_types = {i.name: {"tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(float)": np.float32, "tensor(float16)": np.float16}.get(i.type, np.float32) for i in self.s.get_inputs()}

    def run(self, inputs):
        return self.s.run(None, {k: inputs[k].astype(self.in_types[k], copy=False) for k in self.in_names})

    def release(self):
        self.s = None

    def ort_info(self):
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "external_data": self.external_data,
            "large_or_ext": self.large_or_ext,
            "ort_policy": self.ort_policy,
            "effective_ort_level": self.effective_ort_level,
            "ort_fallback": self.ort_fallback,
            "attempts": list(self.ort_attempts),
        }


_NP = {"float": np.float32, "float32": np.float32, "float16": np.float16, "int32": np.int32, "int64": np.int64, "uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16": np.int16, "uint32": np.uint32}
_qnn_ready = False
_perf = None
_perf_done = False
_contexts = {}
_context_refs = {}


def _qnn_init():
    """QAI AppBuilder (python wheel qai_appbuilder-2.51.40 for linux_aarch64 bundles the QAIRT 2.51 runtime incl. the Hexagon v73 skel; AI Hub jobs are pinned with `--qairt_version 2.51`)."""
    global _qnn_ready, _perf
    if _qnn_ready: return
    import qai_appbuilder
    from qai_appbuilder import QNNConfig, Runtime, LogLevel, ProfilingLevel, PerfProfile
    # The wheel sets ADSP_LIBRARY_PATH only when it is empty; if the box already exports one (HSPTek installs QAIRT 2.41 skels in
    # /usr/lib/rfsa/adsp), the 2.51 stub would meet a 2.41 skel. Put the bundled libs first (same rule as tools/env.sh; S2S_SYSTEM_QNN=1 opts out).
    libs = os.path.join(os.path.dirname(qai_appbuilder.__file__), "libs")
    if os.environ.get("S2S_SYSTEM_QNN", "0") != "1" and not sys.platform.startswith("win") and not os.environ.get("ADSP_LIBRARY_PATH", "").startswith(libs):
        os.environ["ADSP_LIBRARY_PATH"] = f"{libs};/usr/lib/rfsa/adsp;/dsp"
    print(f"[s2s] ADSP_LIBRARY_PATH={os.environ.get('ADSP_LIBRARY_PATH', '')}", file=sys.stderr)
    QNNConfig.Config(Runtime.HTP, LogLevel.WARN, ProfilingLevel.OFF)
    _perf = PerfProfile
    _qnn_ready = True


def _apply_perf():
    """Apply HTP perf profile (default BURST) AFTER loading the first QNNContext (QAI AppBuilder 2.51.40 §3.9)."""
    global _perf_done
    if _perf_done or _perf is None:
        return
    want = os.environ.get("S2S_PERF", "BURST").upper()
    _perf_done = True
    if want == "NONE":
        return
    try:
        _perf.SetPerfProfileGlobal(getattr(_perf, want))
        print(f"[s2s] HTP perf profile: {want}", file=sys.stderr)
    except Exception as e:
        print(f"[s2s] could not set perf profile {want}: {e!r}", file=sys.stderr)


class QnnGraph(Graph):
    """One compiled graph on the Hexagon NPU through QAI AppBuilder. `path` is a context binary (.bin from an AI Hub link job, must match the wheel's
    QAIRT version) or a portable .dlc (needs QAIRT >= 2.41; the first load writes a cached <name>.dlc.bin next to it, later loads are fast).
    Multiple graphs packed in one multi-graph context binary share the same QNNContext and select their graph by `graph_index`."""

    def __init__(self, path, name=None, input_order=None, output_order=None, graph_index=None):
        super().__init__(); _qnn_init()
        from qai_appbuilder import QNNContext
        self.path = path; self.name = name or os.path.basename(path); self.out_order = output_order
        self.graph_index = int(graph_index) if graph_index is not None else None
        self._ctx_key = os.path.abspath(path)
        ctx = _contexts.get(self._ctx_key)
        if ctx is None:
            # keyword arguments: the positional order of QNNContext is (name, path, backend_lib, system_lib, is_async, input_dtype, output_dtype, ...)
            ctx = QNNContext(model_name=self.name, model_path=path, input_data_type="native", output_data_type="native")
            _contexts[self._ctx_key] = ctx
            _context_refs[self._ctx_key] = 0
        _context_refs[self._ctx_key] = _context_refs.get(self._ctx_key, 0) + 1
        self.ctx = ctx
        _apply_perf()
        def _try(f, default=None):
            try:
                if self.graph_index is not None:
                    try: return f(self.graph_index)
                    except TypeError: pass
                return f()
            except Exception:
                return default
        self.in_names = _try(self.ctx.getInputName, None); self.in_shapes = _try(self.ctx.getInputShapes, None); self.in_types = _try(self.ctx.getInputDataType, None)
        self.out_shapes = _try(self.ctx.getOutputShapes, None); self.out_types = _try(self.ctx.getOutputDataType, None)
        self.in_order = input_order or self.in_names

    def run(self, inputs):
        order = self.in_order or list(inputs); arrs = []
        for i, k in enumerate(order):
            a = inputs[k]
            if self.in_types and i < len(self.in_types):
                dt = _NP.get(str(self.in_types[i]).lower().replace("np.", "").replace("numpy.", ""), None)
                if dt is not None: a = a.astype(dt, copy=False)
            arrs.append(np.ascontiguousarray(a).reshape(-1))
        outs = self.ctx.Inference(arrs, graphIndex=self.graph_index) if self.graph_index is not None else self.ctx.Inference(arrs)
        res = []
        for i, o in enumerate(outs):
            o = np.asarray(o)
            if self.out_shapes and i < len(self.out_shapes):
                try: o = o.reshape(tuple(self.out_shapes[i]))
                except Exception: pass
            res.append(o)
        return [res[j] for j in self.out_order] if self.out_order else res

    def release(self):
        """Release this graph reference; calls QNNContext.release() once no live QnnGraph shares the context."""
        if self.ctx is None:
            return
        key = getattr(self, "_ctx_key", None)
        if key in _context_refs:
            _context_refs[key] = max(0, _context_refs[key] - 1)
            if _context_refs[key] > 0:
                self.ctx = None
                return
            _context_refs.pop(key, None)
            _contexts.pop(key, None)
        try: self.ctx.release()
        except Exception: pass
        self.ctx = None


def make_graph(model_dir, spec, backend, ort_policy=None):
    """spec: {'onnx': relative path (cpu ref), 'ctx': relative path (npu), 'in': [...], 'out': [...]}"""
    if backend == "cpu": return OrtGraph(os.path.join(model_dir, spec["onnx"]), spec.get("name"), ort_policy=ort_policy)
    if backend == "npu": return QnnGraph(os.path.join(model_dir, spec["ctx"]), spec.get("name"), spec.get("in"), spec.get("out"), graph_index=spec.get("graph_index"))
    raise ValueError(backend)
