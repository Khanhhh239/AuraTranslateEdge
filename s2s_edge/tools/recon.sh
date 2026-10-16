#!/usr/bin/env bash
# One-shot environment survey for a new device / QDC session (about 1-2 minutes). Output: recon_<host>_<date>.txt next to this script's cwd.
# Usage: bash tools/recon.sh
OUT="recon_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
sec() { echo; echo "===== $* ====="; }
{
sec "date / host";            date -Is; hostname; uname -a
sec "os-release";             cat /etc/os-release 2>/dev/null | head -8
sec "soc";                    cat /sys/devices/soc0/machine /sys/devices/soc0/soc_id /sys/devices/soc0/family 2>/dev/null; grep -m1 -i "hardware\|model name" /proc/cpuinfo
sec "cpu";                    lscpu 2>/dev/null | grep -E "Model name|CPU\(s\)|Architecture|Max MHz" | head -6
sec "memory";                 free -h
sec "disk";                   df -h / /home /tmp 2>/dev/null | head -6
sec "network (expect: none when testing offline)"; ip -br addr 2>/dev/null | head -10; (ping -c1 -W2 8.8.8.8 >/dev/null 2>&1 && echo "INTERNET: reachable") || echo "INTERNET: unreachable"
sec "python";                 python3 --version 2>&1; pip3 --version 2>&1 | head -1
sec "python packages (relevant)"; pip3 list 2>/dev/null | grep -iE "numpy|scipy|onnx|onnxruntime|qai|qnn|sentencepiece|kaldi|soundfile|sounddevice|torch|psutil|jiwer|sacrebleu"
sec "onnxruntime providers"; python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())" 2>&1 | tail -1
sec "qai_appbuilder";         python3 -c "import qai_appbuilder, sys; print('qai_appbuilder OK', getattr(qai_appbuilder,'__version__',''))" 2>&1 | tail -1
sec "QNN tools / libs";       which qnn-net-run qnn-context-binary-generator qnn-platform-validator 2>&1 | head; ls -d /opt/qcom* /opt/qairt* /usr/lib/qnn* 2>/dev/null
                              ls /usr/lib/aarch64-linux-gnu/libQnn* /usr/lib/libQnn* /opt/*/lib/*/libQnnHtp*.so 2>/dev/null | head -20
                              (find /usr /opt /lib /home -maxdepth 6 \( -name "libQnnHtp*.so" -o -name "libQnnSystem.so" -o -name "qnn-net-run" -o -name "libQnnDsp*.so" \) 2>/dev/null | head -20)
sec "HTP / DSP devices";      ls -l /dev/fastrpc* /dev/adsprpc* /dev/dri 2>/dev/null; ls /usr/lib/rfsa/adsp /usr/lib/dsp 2>/dev/null | head -20
sec "audio devices";          arecord -l 2>&1 | head -12; aplay -l 2>&1 | head -12; (pactl info 2>/dev/null | head -4)
sec "sound python";           python3 -c "import sounddevice as sd; print(sd.query_devices())" 2>&1 | tail -6
sec "limits";                 ulimit -a | grep -E "open files|max memory|virtual" ; nproc
} 2>&1 | tee "$OUT"
echo; echo "saved $OUT  (copy it back for analysis)"
