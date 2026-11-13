# -*- coding: utf-8 -*-
"""LAPTOP side of the web UI: forwards port 8770 over the USB cable (adb) and opens the page served by the box.
  python tools/laptop_web.py [--port 8770]
The page, the models and the data are on the box; this laptop only displays it (and records the microphone for the mic test)."""
import argparse, shutil, subprocess, sys, time, urllib.request, webbrowser
ADB = shutil.which("adb") or r"C:\platform-toolsdb.exe"

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8770); a = ap.parse_args()
try: print(subprocess.run([ADB, "forward", f"tcp:{a.port}", f"tcp:{a.port}"], capture_output=True, text=True, check=True).stdout or "adb forward ok")
except Exception as e: sys.exit(f"adb forward failed ({e!r}): is the box plugged in (adb devices)?")
for i in range(30):
    try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3).read(); break
    except Exception: print("waiting for the box server (bash tools/box_serve.sh on the box)..."); time.sleep(2)
else: sys.exit("box server not reachable")
webbrowser.open(f"http://127.0.0.1:{a.port}/"); print(f"opened http://127.0.0.1:{a.port}/")
