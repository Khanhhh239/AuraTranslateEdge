# -*- coding: utf-8 -*-
"""ASR front ends. AsrVi: Zipformer-150M CTC (fbank on host, CTC collapse on host). AsrXx: SenseVoice-Small for en/zh/ko (byte output decoded on host)."""
import os, numpy as np
from .audio import SR

T_FRAMES = 1500                                                  # Zipformer static window: 1500 fbank frames = 15 s
LID = {"zh": 3, "en": 4, "ko": 12}


def _fbank(w):
    import kaldi_native_fbank as knf
    o = knf.FbankOptions(); o.mel_opts.num_bins = 80; o.frame_opts.samp_freq = SR; o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = -400.0
    f = knf.OnlineFbank(o); f.accept_waveform(SR, w.tolist()); f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


class AsrVi:
    max_s = 14.5

    def __init__(self, graph, bpe_path):
        import sentencepiece as spm
        self.g = graph; self.sp = spm.SentencePieceProcessor(model_file=bpe_path)

    def __call__(self, wav):
        f = _fbank(wav[: int(self.max_s * SR)]); n = min(len(f), T_FRAMES); x = np.zeros((1, T_FRAMES, 80), np.float32); x[0, :n] = f[:n]
        out = self.g(x=x, x_lens=np.array([n], np.int64))
        lg, el = (out[0], out[1]) if np.asarray(out[0]).ndim == 3 else (out[1], out[0])
        lg = np.asarray(lg).reshape(1, -1, 2000); real = int(np.asarray(el).reshape(-1)[0]); ids = lg[0, :real].argmax(-1)
        pieces, prev = [], None
        for t in ids:
            if t != 0 and t != prev: pieces.append(int(t))
            prev = t
        return self.sp.decode(pieces).lower().strip()


class AsrXx:
    """SenseVoice-Small (en / zh / ko). Two graphs as deployed on the NPU: frontend (wav -> fbank) and encoder+CTC (fbank -> UTF-8 bytes);
    or one combined graph (cpu reference). language ids: zh 3, en 4, ko 12; textnorm 14 = with inverse text normalisation/punctuation."""
    max_s = 28.5

    def __init__(self, frontend, encoder, lang):
        self.fe = frontend; self.enc = encoder; self.lang = lang

    def __call__(self, wav):
        n = min(len(wav), 464000); b = np.zeros((1, 464000), np.float32); b[0, :n] = wav[:n]
        lid = np.array([LID[self.lang]], np.int32); tn = np.array([14], np.int32)
        if self.enc is None:                                              # combined graph
            o = self.fe(wav=b, wav_len=np.array([n], np.int32), language=lid, textnorm=tn); a = np.asarray(o[0])
        else:
            f = self.fe(wav=b, wav_len=np.array([n], np.int32)); fb = [np.asarray(x) for x in f]
            big = [x for x in fb if x.size > 1000][0].reshape(1, 500, 560).astype(np.float32); sl = [x for x in fb if x.size == 1][0].reshape(1).astype(np.int32)
            o = self.enc(fbank=big, speech_lengths=sl, language=lid, textnorm=tn); a = np.asarray(o[0])
        return bytes(int(v) & 0xFF for v in a.astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
