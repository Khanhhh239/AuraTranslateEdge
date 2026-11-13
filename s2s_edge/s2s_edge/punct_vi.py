# -*- coding: utf-8 -*-
"""Vietnamese punctuation + capitalisation restoration for the Zipformer ASR output (which is lower case, no punctuation), on the CPU with
onnxruntime + numpy only (no torch / transformers on the box).

Model: dragonSwing/vibert-capu (viBERT-base token tagger, GECToR style; CC-BY-SA-4.0), exported to ONNX and int8-quantised
(assets/vi_capu/vibert_capu_int8.onnx, 111 MiB). Labels per word: keep / capitalise / upper / append "," "." ":" "?" ...
This file re-implements the reference wrapper (gec_model.py, utils.py of that repository) without torch:
  * tokenisation = what transformers' BertTokenizerFast actually does there (the wrapper asks for do_basic_tokenize=False, which the fast
    tokenizer ignores): clean text, pad CJK chars, split on whitespace + punctuation, no lower-casing / accent stripping, greedy WordPiece;
  * up to 3 tagging iterations, sentences of < 3 words left to the rule-based vi_punct (the reference returns them unchanged);
  * long inputs split into 48-word chunks with 12 words overlap and merged with difflib, as in the reference.
Measured (text level, BTC sentences with punctuation removed, NLLB-600M): chrF vi->en 52.0 vs 50.4 (vi_punct), vi->ko 28.9 vs 25.9; int8 = fp32 quality.
Parity of this port with the reference implementation: tools/check_punct_vi.py."""
import os, re, unicodedata
from difflib import SequenceMatcher
import numpy as np
from .numeric_guard import GuardReport, extract_known_numeric_spans, guard_source_numeric_changes

START = "$START"
LABELS = ["$KEEP", "$TRANSFORM_CASE_CAPITAL", "$APPEND_,", "$APPEND_.", "$TRANSFORM_VERB_VB_VBN", "$TRANSFORM_CASE_UPPER", "$APPEND_:", "$APPEND_?",
          "$TRANSFORM_VERB_VB_VBC", "$TRANSFORM_CASE_LOWER", "$TRANSFORM_CASE_CAPITAL_1", "$TRANSFORM_CASE_UPPER_-1", "$MERGE_SPACE", "@@UNKNOWN@@", "@@PADDING@@"]
INCORRECT = 1                                                         # d_tags: CORRECT, INCORRECT, @@UNKNOWN@@, @@PADDING@@
PUNC = {":", ".", ",", "?"}


def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126: return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F
            or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    """BertNormalizer(clean_text, handle_chinese_chars, no lowercase, no accent strip) + BertPreTokenizer + WordPiece('##', [UNK], 100 chars)."""

    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as fh:
            self.vocab = {w: i for i, w in enumerate(fh.read().split("\n")) if w}
        self.unk = self.vocab["[UNK]"]; self.start_id = len(self.vocab)          # "$START" was added after the vocabulary (id 38168)

    def _pieces(self, word):
        out = []
        for ch in word:                                                         # normaliser
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"): continue
            out.append(f" {ch} " if _is_cjk(cp) else (" " if ch.isspace() else ch))
        text = "".join(out); toks = []
        for w in text.split():                                                  # pre-tokeniser: whitespace, then every punctuation char alone
            cur = ""
            for ch in w:
                if _is_punct(ch):
                    if cur: toks.append(cur); cur = ""
                    toks.append(ch)
                else: cur += ch
            if cur: toks.append(cur)
        ids = []
        for t in toks:                                                          # WordPiece, greedy longest match
            if len(t) > 100: ids.append(self.unk); continue
            start, sub = 0, []
            while start < len(t):
                end = len(t); cur = None
                while start < end:
                    s = t[start:end] if start == 0 else "##" + t[start:end]
                    if s in self.vocab: cur = self.vocab[s]; break
                    end -= 1
                if cur is None: sub = [self.unk]; break
                sub.append(cur); start = end
            ids.extend(sub)
        return ids

    def encode_words(self, words):
        """-> input_ids, offsets (index of the first sub-token of $START and of every word)."""
        ids, offsets = [self.start_id], [0]
        for w in words:
            p = self._pieces(w)
            if not p: continue                                                  # a word that normalises to nothing gets no sub-token (no word id)
            offsets.append(len(ids)); ids.extend(p)
        return ids, offsets


def _case(token, action):
    if action.endswith("LOWER"): return token.lower()
    if action.endswith("UPPER"): return token.upper()
    if action.endswith("CAPITAL"): return token.capitalize()
    if action.endswith("CAPITAL_1"): return token[0] + token[1:].capitalize()
    if action.endswith("UPPER_-1"): return token[:-1].upper() + token[-1]
    return token


def load_verb_forms(path):
    """verb-form-vocab.txt ('ba_3:VB_VBN' ...): the reference repurposes GECToR's English verb-form labels for Vietnamese number words -> digits."""
    dec = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if ":" not in line: continue
        words, tags = line.split(":"); w1, w2 = words.split("_"); t1, t2 = tags.split("_"); k = f"{w1}_{t1}_{t2.strip()}"
        if k not in dec: dec[k] = w2
    return dec


def _protected_edit_constraints(tokens):
    """Identify token-boundary indices inside protected spans where punct/case/verb edits must be blocked."""
    text = " ".join(tokens)
    spans = extract_known_numeric_spans(text, "vi")
    # Also protect contiguous Vietnamese spoken numeral sequences and "số không" against internal punctuation.
    word_matches = list(re.finditer(r"[^\W\d_]+", text, re.UNICODE))
    words_low = [m.group(0).casefold() for m in word_matches]
    spoken_words = {
        "không", "một", "mốt", "hai", "ba", "tư", "bốn", "năm", "lăm", "nhăm",
        "sáu", "bảy", "bẩy", "tám", "chín", "mười", "mươi", "linh", "lẻ",
        "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ", "phẩy", "âm", "rưỡi",
    }
    protected_ranges = [(sp.start, sp.end, sp.signature is None) for sp in spans]
    wi = 0
    while wi < len(word_matches):
        if words_low[wi] in spoken_words:
            wj = wi + 1
            while wj < len(word_matches) and words_low[wj] in spoken_words:
                wj += 1
            prev_w = words_low[wi - 1] if wi > 0 else ""
            if (wj - wi) >= 2 or (wj - wi == 1 and words_low[wi] == "không" and prev_w == "số"):
                protected_ranges.append((word_matches[wi].start(), word_matches[wj - 1].end(), False))
            wi = wj
        else:
            wi += 1

    tok_offsets = []
    pos = 0
    for tok in tokens:
        tok_offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1

    blocked_append_after = set()
    opaque_token_idx = set()
    for idx, (ta, tb) in enumerate(tok_offsets):
        for ra, rb, is_opaque in protected_ranges:
            if is_opaque and not (tb <= ra or ta >= rb):
                opaque_token_idx.add(idx)
            # If the gap immediately after token `idx` is strictly inside a protected range, block $APPEND_ after it.
            if ra <= ta and tb < rb:
                blocked_append_after.add(idx)
    return blocked_append_after, opaque_token_idx


def _apply(tokens, edits, verbs=None):
    """utils.get_target_sent_by_edits restricted to safe case/append labels (never altering token identity or numeric spans)."""
    blocked_append_after, opaque_token_idx = _protected_edit_constraints(tokens)
    t = tokens[:]; shift = 0
    for start, end, label in edits:
        if start < 0: continue
        if label.startswith("$TRANSFORM_VERB_") or label.startswith("$MERGE_"):
            # Preserve exact source token order and never rewrite spoken words into digits (e.g. "không" -> "0").
            continue
        pos = start + shift
        src = t[pos] if len(t) > pos else ""
        if start == end:                                                       # append after original token (start - 1)
            if (start - 1) in blocked_append_after:
                continue
            word = label.replace("$APPEND_", "")
            if (pos < len(t) and t[pos] == word) or (pos > 0 and t[pos - 1] == word): continue
            t[pos:pos] = [word]; shift += 1
        elif label.startswith("$TRANSFORM_CASE"):
            if start in opaque_token_idx:
                continue
            t[pos] = _case(src, label) if src else src
    return t


def _softmax(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class ViCapu:
    def __init__(self, onnx_path, vocab_path, verb_path=None, threads=4, iterations=3, min_len=3, chunk=48, overlap=12, min_words_cut=6, max_len=64, ort_policy=None):
        from .backends import create_cpu_ort_session
        verb_path = verb_path or os.path.join(os.path.dirname(vocab_path), "verb-form-vocab.txt"); self.verbs = load_verb_forms(verb_path) if os.path.exists(verb_path) else {}
        self.s, self.ort_attempts, self.effective_ort_level, self.ort_fallback, self.ort_policy, _, _, _ = create_cpu_ort_session(
            onnx_path, ort_policy=ort_policy, large_or_ext=False, intra_op_num_threads=threads
        )
        self.names = [i.name for i in self.s.get_inputs()]
        self.wp = WordPiece(vocab_path); self.iterations, self.min_len, self.chunk, self.overlap, self.min_words_cut, self.max_len = iterations, min_len, chunk, overlap, min_words_cut, max_len
        self.last_validation: GuardReport | None = None

    def _tag(self, words):
        ids, offs = self.wp.encode_words(words[: self.max_len])
        feed = {"input_ids": np.array([ids], np.int64), "token_type_ids": np.zeros((1, len(ids)), np.int64), "attention_mask": np.ones((1, len(ids)), np.int64),
                "input_offsets": np.array([offs], np.int64)}
        lg, dl = self.s.run(None, {n: feed[n] for n in self.names})
        p = _softmax(lg[0]); return p.max(-1), p.argmax(-1)

    def _one(self, words):
        """Up to `iterations` tagging passes on one chunk (handle_batch for a batch of one)."""
        final, prev = words[:], [words[:]]
        for _ in range(self.iterations):
            probs, idx = self._tag(final)
            if idx.max() == 0: break
            edits = []
            for i in range(min(len(final), self.max_len) + 1):
                if i >= len(idx) or idx[i] == 0: continue
                lab = LABELS[idx[i]]
                if lab in ("@@UNKNOWN@@", "@@PADDING@@", "$KEEP"): continue
                s, e = (i, i + 1) if lab.startswith("$TRANSFORM_") else (i + 1, i + 1)
                edits.append((s - 1, e - 1, lab if (lab.startswith("$TRANSFORM_") or lab.startswith("$MERGE_")) else lab[lab.index("_") + 1:]))
            pred = _apply(final, edits, self.verbs)
            if pred == final: break
            stop = pred in prev; final = pred; prev.append(pred)
            if stop: break
        return final

    def _chunks_with_ranges(self, w):
        n, c, o = len(w), self.chunk, self.overlap
        if n <= c:
            return [((0, n), w)]
        if n < c * 2 - o:
            k = (n + o + 1) // 2
            return [((0, k), w[:k]), ((k - o, n), w[k - o:])]
        return [((i, min(n, i + c)), w[i: i + c]) for i in range(0, n - o, c - o)]

    def _chunks(self, w):
        return [chunk for _, chunk in self._chunks_with_ranges(w)]

    @staticmethod
    def _non_punct(tokens):
        return [t for t in tokens if t not in PUNC]

    @staticmethod
    def _prefix_through_nth_word(tokens, n_words):
        """Return slice of `tokens` containing the first `n_words` non-punctuation words (plus immediate trailing punct)."""
        if n_words <= 0:
            return []
        count = 0
        end_idx = 0
        for idx, tok in enumerate(tokens):
            if tok not in PUNC:
                if count == n_words:
                    break
                count += 1
            end_idx = idx + 1
        return tokens[:end_idx]

    @staticmethod
    def _suffix_from_nth_word(tokens, skip_words):
        """Return slice of `tokens` starting at the `skip_words`-th non-punctuation word."""
        if skip_words <= 0:
            return list(tokens)
        count = 0
        for idx, tok in enumerate(tokens):
            if tok not in PUNC:
                if count == skip_words:
                    return tokens[idx:]
                count += 1
        return []

    def _merge(self, tokens, nxt):
        if not tokens: return nxt
        si, ti, st, tt = [], [], [], []; keep = self.overlap - self.min_words_cut; i = 0
        while len(si) < self.overlap and -i < len(tokens):
            i -= 1
            if tokens[i] not in PUNC: si.insert(0, i); st.insert(0, tokens[i].lower())
        i = 0
        while len(ti) < self.overlap and i < len(nxt):
            if nxt[i] not in PUNC: ti.append(i); tt.append(nxt[i].lower())
            i += 1
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, st, tt).get_opcodes():
            if tag == "equal":
                if i1 >= keep: return tokens[: si[i1]] + nxt[ti[j1]:]
                if i2 > keep: return tokens[: si[keep]] + nxt[ti[j2 - i2 + keep]:]
            elif tag == "delete" and i1 == 0: keep += i2 // 2
        raise ValueError("no overlap")                                          # the reference prints and keeps the previous result

    def __call__(self, text):
        self.last_validation = None
        words = text.split()
        if not words:
            self.last_validation = guard_source_numeric_changes(text, text, "vi")
            return text
        if len(words) < self.min_len:
            from .textnorm import vi_punct
            out_text = vi_punct(text)
            report = guard_source_numeric_changes(text, out_text, "vi")
            if report.status == "invalid":
                out_text = text
                report.fallback_policy = "fallback_to_raw_source"
            self.last_validation = report
            return out_text

        chunk_items = self._chunks_with_ranges(words)
        parts = [self._one(c) if len(c) >= self.min_len else c for _, c in chunk_items]
        merge_warnings = []

        def _tokens_match(toks, expected_words):
            return [t.lower() for t in self._non_punct(toks)] == [w.lower() for w in expected_words]

        if len(parts) == 1:
            out = parts[0]
            if not _tokens_match(out, words):
                merge_warnings.append({"code": "capu_token_alignment_fallback", "reason": "single_chunk_token_mismatch"})
                out = list(words)
        else:
            ((first_a, first_b), _), first_part = chunk_items[0], parts[0]
            if not _tokens_match(first_part, words[first_a:first_b]):
                merge_warnings.append({"code": "capu_chunk_alignment_fallback", "chunk_start": first_a, "chunk_end": first_b})
                out = list(words[first_a:first_b])
            else:
                out = list(first_part)
            covered_end = first_b

            for ((c_start, c_end), _), p in zip(chunk_items[1:], parts[1:]):
                p_clean = list(p)
                if not _tokens_match(p_clean, words[c_start:c_end]):
                    merge_warnings.append({"code": "capu_chunk_alignment_fallback", "chunk_start": c_start, "chunk_end": c_end})
                    p_clean = list(words[c_start:c_end])
                merged = None
                merge_err = None
                try:
                    cand = self._merge(out, p_clean)
                    if _tokens_match(cand, words[:c_end]):
                        merged = cand
                    else:
                        merge_err = "overlap_alignment_mismatch"
                except Exception as exc:
                    merge_err = repr(exc)

                if merged is None:
                    merge_warnings.append({
                        "code": "overlap_merge_fallback",
                        "chunk_start": c_start,
                        "chunk_end": c_end,
                        "reason": merge_err or "overlap_merge_failed",
                    })
                    cut_src = min(covered_end, c_start + max(1, (covered_end - c_start) // 2))
                    left = self._prefix_through_nth_word(out, cut_src)
                    right = self._suffix_from_nth_word(p_clean, cut_src - c_start)
                    spliced = left + right
                    if _tokens_match(spliced, words[:c_end]):
                        out = spliced
                    else:
                        out = self._prefix_through_nth_word(out, c_start) + p_clean
                        if not _tokens_match(out, words[:c_end]):
                            out = list(words[:c_end])
                else:
                    out = merged
                covered_end = c_end

        out_text = re.sub(r"\s+([:.,?])", r"\1", " ".join(out))
        report = guard_source_numeric_changes(text, out_text, "vi")
        if merge_warnings:
            report.warnings.extend(merge_warnings)
            if report.fallback_policy is None:
                report.fallback_policy = "source_indexed_overlap_splice"
        if report.status == "invalid":
            from .textnorm import vi_punct
            fallback_text = vi_punct(text)
            fb_report = guard_source_numeric_changes(text, fallback_text, "vi")
            if fb_report.status == "invalid":
                fallback_text = text
                fb_report = guard_source_numeric_changes(text, fallback_text, "vi")
                fb_report.fallback_policy = "fallback_to_raw_source"
            else:
                fb_report.fallback_policy = "fallback_to_vi_punct"
            fb_report.warnings.extend(merge_warnings)
            fb_report.warnings.append({"code": "capu_output_rejected", "issues": list(report.issues)})
            self.last_validation = fb_report
            return fallback_text

        self.last_validation = report
        return out_text


_INSTANCE = None


def get(assets_dir, ort_policy=None):
    """Lazy singleton; None when the model files are not in the package (the caller then falls back to vi_punct)."""
    global _INSTANCE
    if _INSTANCE is None:
        d = os.path.join(assets_dir, "vi_capu"); onnx, vocab = os.path.join(d, "vibert_capu_int8.onnx"), os.path.join(d, "vocab.txt")
        if not (os.path.exists(onnx) and os.path.exists(vocab)): return None
        _INSTANCE = ViCapu(onnx, vocab, ort_policy=ort_policy)
    return _INSTANCE
