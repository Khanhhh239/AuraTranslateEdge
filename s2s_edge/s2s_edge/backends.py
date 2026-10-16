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


class QnnGraph(Graph):
    """NPU backend. Implementations are tried in order; the first that imports wins.
    1) qai_appbuilder.QNNContext (Qualcomm python wrapper over libQnnHtp; loads an AI Hub context binary)
    2) subprocess qnn-net-run (slow: process start per call; used only for correctness checks)
    The correct choice is recorded in tools/recon.sh output; this class is completed on the device (session 0)."""

    def __init__(self, path, name=None, input_order=None, output_order=None):
        super().__init__(); self.path = path; self.name = name or os.path.basename(path); self.in_order = input_order; self.out_order = output_order
        self.impl = None
        try:
            from qai_appbuilder import QNNContext, QNNConfig, Runtime, LogLevel, ProfilingLevel, PerfProfile
            QNNConfig.Config(os.environ.get("QNN_LIB_DIR", ""), Runtime.HTP, LogLevel.WARN, ProfilingLevel.OFF)
            self.impl = ("appbuilder", QNNContext(self.name, path))
        except Exception as e:
            raise RuntimeError(f"No NPU runtime available for {path}: {e}")

    def run(self, inputs):
        kind, ctx = self.impl
        order = self.in_order or list(inputs)
        out = ctx.Inference([inputs[k].reshape(-1) for k in order])
        return [np.asarray(o) for o in out]


def make_graph(model_dir, spec, backend):
    """spec: {'onnx': relative path (cpu ref), 'ctx': relative path (npu), 'in': [...], 'out': [...]}"""
    if backend == "cpu": return OrtGraph(os.path.join(model_dir, spec["onnx"]), spec.get("name"))
    if backend == "npu": return QnnGraph(os.path.join(model_dir, spec["ctx"]), spec.get("name"), spec.get("in"), spec.get("out"))
    raise ValueError(backend)
