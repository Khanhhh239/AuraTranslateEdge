# Zipformer ASR tren NPU Dragonwing IQ-9075 (Qualcomm AI Hub)

Code copy nguyen trang tu OneVoice/src/step4_hardware va work_local/vendored_zipformer (chua tung duoc commit truoc do).

- step4_hardware/build_zip150_ctc_static.py - dung graph Zipformer CTC shape tinh
- step4_hardware/run_zip150_ctc_w8a16_iq9075.py - compile + chay W8A16 tren IQ-9075
- step4_hardware/build_full_pipeline_onnx.py, un_full_pipeline_w16a16_iq9075.py - ca pipeline W16A16
- step4_hardware/zipformer_onecall_service.py - dich vu goi mot lan
- step4_hardware/eval_zip150_ctc_vs_rnnt.py - so CTC voi RNNT
- step4_hardware/diag_*.py - chan doan (mat tu dau cau, do chinh xac fbank)
- endored_zipformer/ - zipformer.py, k2.py

Chua kiem lai tren may nay luc push: file chua duoc chay lai, ket qua/so lieu khong kem theo. Can cau hinh qai-hub (API token) de chay.
