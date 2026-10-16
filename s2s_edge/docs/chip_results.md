# Measured on the Dragonwing IQ-9075 EVK through Qualcomm AI Hub (development measurements)

These numbers come from AI Hub `profile` / `inference` jobs on a hosted IQ-9075 EVK: chip execution time per call, **excluding** queueing and model loading.
They are *not* on-device end-to-end measurements; those are produced by `python -m s2s_edge.bench` on the device (see README).

| Graph | Role | Job (compile / link QAIRT 2.51) | Layers on NPU | Chip time per call | Quality check |
|---|---|---|---|---|---|
| `asr_vi` Zipformer-150M CTC, fp16 | vi ASR, 15 s window | `jpv226qjg` / `jg9ood6wg`; profile `j57ooy99g`, infer `jp4eel31g` | 3092 / 3092 | **174.3 ms** | WER on 15 clips: clean 7.5 %, 5 dB 9.5 %, 0 dB 6.2 % |
| `asr_xx_*` SenseVoice-Small (frontend + w8a16 encoder) | en/zh/ko ASR, 29 s window | frontend `j57ez99vp`, encoder (inference model `jglynxxj5`) / `jp1oo6r75`, `jgd662jzp` | all NPU | not re-profiled in this session | earlier 30-clip eval: en CER 4.8 %, zh CER 8.3 %, ko CER 7.3 % (CPU fp32 reference: 4.0 / 8.7 / 7.6 %) |
| `mt_enc` NLLB encoder, fp16 | MT | `jp4ee3yvg` / `jp1oo6r85`; profile `jp4ee3evg`, infer `j5m9989wg` | 463 / 463 | **7.1 ms** | cosine 1.0 vs CPU on 8 sentences (all 3 outputs) |
| `mt_dec` NLLB decoder step, fp16 | MT, one greedy step with static KV cache | `jpe66y71g` / `jgd662jrp`; profile `jgd6624rp`, infer `j5q44v1ng` | 667 / 667 | **14.8 ms per step** | 42 / 42 teacher-forced next tokens equal to CPU; cache outputs cosine 1.0 |
| `tts_vi` Piper vais1000 (byte text → audio) | vi TTS | `j5q46w3ng` / `jgodd87q5` | 2077 / 2077 | 480 ms (fixed, 40 windows) | not ASR-verified |
| `tts_zh` MeloTTS-ZH + static front + BERT | zh TTS | `jpy8mw17g` / `j5wyy4jmg`; profile `j5m9m9e9g`, infer `jgn1n10qp` | 3451 / 3451 | **720.2 ms** (fixed, 384 frames) | ASR CER 6.4 % on 40 texts (CPU: 6.0 %) |
| `tts_enko` Supertonic-3 fused | en/ko TTS | `jprxn9nep` / `jg9ood68g`; profile `jpe6l2l0g`, infer `jgzz4w46g` | 4476 / 4476 | 748.3 ms (fixed, 150 latent frames) | round trip WER 0.0 % (en), CER 1.5 % (ko), 6 sentences each |

## Estimated chip time per utterance (sum of the profiles above; an *estimate*, not an end-to-end measurement)

| Direction | ASR | MT (enc + ~25 decoder steps) | TTS | Sum |
|---|---|---|---|---|
| vi → en / ko | 174 ms | ~380 ms | ~750 ms | ~1.3 s |
| vi → zh | 174 ms | ~380 ms | ~720 ms | ~1.3 s |
| en / zh / ko → vi | SenseVoice (to be measured) | ~380 ms | ~480 ms | ≥ 0.9 s + ASR |

The TTS graphs always compute their full static window, so their time is constant per chunk. Host work (fbank, SentencePiece, number
normalisation, I/O) comes on top and is measured by the benchmark harness.

## Translation quality (CPU reference, FLORES-200 devtest, gold source text; see `logs/eval_mt_split.json`)

Filled in by `tools/eval_mt.py`; first result: vi→en chrF 51.0 / BLEU 20.8 on 25 sentences (no truncation with the split decoder).

## Findings worth keeping

* The 16-step unrolled NLLB graph compiles to an **8.8 GB DLC** (the link job dedupes it to a 1.3 GB context binary) and can only produce 15 target tokens.
  The split encoder + decoder-step version is 0.64 + 0.93 GB as context binaries, has no 15-token limit and runs the greedy loop on the host.
* Context binaries are bound to the QAIRT version: AI Hub link jobs are pinned with `--qairt_version 2.51`, matching the `qai_appbuilder 2.51.40` wheel.
* Token ids above 2048 are not exact in fp16 on the HTP when compared as a single number; compare `(hi, lo)` halves instead (this is how the Melo front works).
  `int32` Cast / Mul / Add / Gather on ids up to 256 k are exact on the chip (384 / 384 in a probe).
