# -*- coding: utf-8 -*-
"""Graph backends. A Graph wraps ONE compiled model and is called with named numpy inputs, returning outputs in ONNX output order.
OrtGraph : onnxruntime CPU (development / reference, runs everywhere).
QnnGraph : Qualcomm NPU on the device (context binary produced by AI Hub). Loader chosen at run time, see `QnnGraph.load` (decided after tools/recon.sh)."""
import os, time, json
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


class OrtGraph(Graph):
    def __init__(self, path, name=None):
        super().__init__(); import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3; so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); self.name = name or os.path.basename(path)
        self.in_names = [i.name for i in self.s.get_inputs()]
        self.in_types = {i.name: {"tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(float)": np.float32, "tensor(float16)": np.float16}.get(i.type, np.float32) for i in self.s.get_inputs()}

    def run(self, inputs):
        return self.s.run(None, {k: inputs[k].astype(self.in_types[k], copy=False) for k in self.in_names})


_NP = {"float": np.float32, "float32": np.float32, "float16": np.float16, "int32": np.int32, "int64": np.int64, "uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16": np.int16, "uint32": np.uint32}
_qnn_ready = False


def _qnn_init():
    """QAI AppBuilder (python wheel qai_appbuilder-2.51.40 for linux_aarch64 bundles the QAIRT 2.51 runtime incl. the Hexagon v73 skel; AI Hub jobs are pinned with `--qairt_version 2.51`)."""
    global _qnn_ready
    if _qnn_ready: return
    from qai_appbuilder import QNNConfig, Runtime, LogLevel, ProfilingLevel, PerfProfile
    QNNConfig.Config(Runtime.HTP, LogLevel.WARN, ProfilingLevel.OFF)
    try: PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)          # keep the HTP at burst clocks while the app runs
    except Exception: pass
    _qnn_ready = True


class QnnGraph(Graph):
    """One compiled graph on the Hexagon NPU through QAI AppBuilder. `path` is a context binary (.bin from an AI Hub link job, must match the wheel's
    QAIRT version) or a portable .dlc (needs QAIRT >= 2.41; the first load writes a cached <name>.dlc.bin next to it, later loads are fast).
    Inputs / outputs use the graph's native dtypes (int32 ids, float32 tensors). Output order is the graph's output order; `out_order` can
    remap it if golden check shows a permutation (models.json: "out": [index, ...])."""

    def __init__(self, path, name=None, input_order=None, output_order=None):
        super().__init__(); _qnn_init()
        from qai_appbuilder import QNNContext
        self.path = path; self.name = name or os.path.basename(path); self.out_order = output_order
        # keyword arguments: the positional order of QNNContext is (name, path, backend_lib, system_lib, is_async, input_dtype, output_dtype, ...)
        self.ctx = QNNContext(model_name=self.name, model_path=path, input_data_type="native", output_data_type="native")
        def _try(f, default=None):
            try: return f()
            except Exception: return default
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
        outs = self.ctx.Inference(arrs); res = []
        for i, o in enumerate(outs):
            o = np.asarray(o)
            if self.out_shapes and i < len(self.out_shapes):
                try: o = o.reshape(tuple(self.out_shapes[i]))
                except Exception: pass
            res.append(o)
        return [res[j] for j in self.out_order] if self.out_order else res


def make_graph(model_dir, spec, backend):
    """spec: {'onnx': relative path (cpu ref), 'ctx': relative path (npu), 'in': [...], 'out': [...]}"""
    if backend == "cpu": return OrtGraph(os.path.join(model_dir, spec["onnx"]), spec.get("name"))
    if backend == "npu": return QnnGraph(os.path.join(model_dir, spec["ctx"]), spec.get("name"), spec.get("in"), spec.get("out"))
    raise ValueError(backend)
