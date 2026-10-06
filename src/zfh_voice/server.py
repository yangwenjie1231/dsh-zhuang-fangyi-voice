# -*- coding: utf-8 -*-
"""HTTP 服务：给桌宠等外部程序调用

启动：
    python -m zfh_voice serve --port 8765

接口：
    GET  /health          → {"ok":true,"backend":"onnx","ready":true}
    GET  /voices          → 可用音色/配置信息
    POST /tts             → body {"text":"...","seed":null,"format":"wav"}
                            返回 audio/wav 二进制
    POST /tts.json        → 返回 {"ok":true,"duration":1.23,"wav_base64":"..."}
    POST /cache/clear     → 清空合成缓存

只用标准库实现，避免额外依赖（不需要 fastapi/flask）。
"""
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__


def make_handler(tts):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = f"zfh-voice/{__version__}"

        def log_message(self, fmt, *args):
            if os.environ.get("ZFH_VERBOSE"):
                super().log_message(fmt, *args)

        # ---- 工具 ----
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except Exception:
                return {}

        # ---- 路由 ----
        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path in ("/health", "/"):
                return self._send(200, {
                    "ok": True, "service": "zfh-voice",
                    "version": __version__, "backend": tts._backend_kind,
                    "ready": True})
            if path == "/voices":
                return self._send(200, {
                    "ok": True,
                    "voices": [{
                        "id": "zhuang-fangyi",
                        "name": "庄方宜",
                        "language": "zh",
                        "backend": tts._backend_kind,
                        "ref": os.path.basename(tts.ref_wav or ""),
                    }]})
            return self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/tts":
                return self._tts(binary=True)
            if path == "/tts.json":
                return self._tts(binary=False)
            if path == "/cache/clear":
                n = tts.clear_cache()
                return self._send(200, {"ok": True, "cleared": n})
            return self._send(404, {"ok": False, "error": "not found"})

        def _tts(self, binary):
            req = self._read_json()
            text = (req.get("text") or "").strip()
            if not text:
                return self._send(400, {"ok": False, "error": "text 不能为空"})
            seed = req.get("seed")
            t0 = time.time()
            try:
                with lock:                     # 后端非线程安全，串行化
                    r = tts.say(text, seed=seed, use_cache=True)
            except Exception as e:
                return self._send(500, {"ok": False,
                                        "error": f"{type(e).__name__}: {e}"})
            if binary:
                return self._send(200, r.to_wav_bytes(),
                                  ctype="audio/wav")
            return self._send(200, {
                "ok": True, "duration": round(r.duration, 3),
                "sr": r.sr, "cached": r.cached,
                "elapsed": round(time.time() - t0, 3),
                "wav_base64": base64.b64encode(r.to_wav_bytes()).decode()})

    return Handler


def serve(tts, host="127.0.0.1", port=8765):
    # 预热：提前加载后端，避免第一个请求等待
    print("预热推理后端 ...")
    t0 = time.time()
    _ = tts.backend
    print(f"就绪（{time.time()-t0:.1f}s）")
    httpd = ThreadingHTTPServer((host, port), make_handler(tts))
    print(f"服务已启动: http://{host}:{port}")
    print(f"  健康检查  GET  /health")
    print(f"  合成      POST /tts        body: {{\"text\":\"...\"}} → audio/wav")
    print(f"  合成(JSON) POST /tts.json   → {{ok,duration,wav_base64}}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
