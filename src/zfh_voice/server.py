# -*- coding: utf-8 -*-
"""HTTP 服务：给桌宠等外部程序调用

启动：
    python -m zfh_voice serve --port 8765

接口：
    GET  /health          → {"ok":true,"backend":"onnx","ready":true}
    GET  /voices          → 可用音色/配置信息
    POST /tts             → body {"text":"...","seed":null,"format":"wav"}
                            可选 brightness_db（亮度补偿 dB，缺省 0 = 关闭）
                            返回 audio/wav 二进制
    POST /tts.json        → 返回 {"ok":true,"duration":1.23,"wav_base64":"..."}
    POST /cache/clear     → 清空合成缓存

seed 语义：缺省/null 表示「未指定」，会用 TTS 实例的 default_seed（默认 42）
保证短句可复现；显式传数字则用该值。

只用标准库实现，避免额外依赖（不需要 fastapi/flask）。
"""
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__


def make_handler(tts, state=None):
    lock = threading.Lock()
    state = state if state is not None else {}
    state.setdefault("last_activity", time.time())
    state.setdefault("loads", 0)
    state.setdefault("unloads", 0)
    # 在途请求计数：看门狗据此避开"正在合成"的窗口。
    # 否则长合成期间 last_activity 不更新，看门狗会误判为空闲而把模型卸掉。
    state.setdefault("inflight", 0)

    class Handler(BaseHTTPRequestHandler):
        server_version = f"zfh-voice/{__version__}"

        def log_message(self, fmt, *args):
            if os.environ.get("ZFH_VERBOSE"):
                super().log_message(fmt, *args)

        def _touch(self):
            state["last_activity"] = time.time()

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
                self._touch()
                idle = time.time() - state["last_activity"]
                return self._send(200, {
                    "ok": True, "service": "zfh-voice",
                    "version": __version__, "backend": tts._backend_kind,
                    "ready": True,
                    # 常驻状态：模型是否在内存/显存里，以及距上次活动多久
                    "resident": tts.is_loaded,
                    "idle_seconds": round(idle, 1),
                    "idle_timeout": state.get("idle_timeout", 0),
                    "loads": state["loads"], "unloads": state["unloads"],
                })
            if path == "/voices":
                self._touch()
                return self._send(200, {
                    "ok": True,
                    "voices": [{
                        "id": "zhuang-fangyi",
                        "name": "庄方宜",
                        "language": "zh",
                        "backend": tts._backend_kind,
                        "ref": os.path.basename(tts.ref_wav or ""),
                    }]})
            if path == "/unload":
                self._touch()
                done = tts.unload()
                if done:
                    state["unloads"] += 1
                return self._send(200, {"ok": True, "unloaded": done})
            return self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/tts":
                return self._tts(binary=True)
            if path == "/tts.json":
                return self._tts(binary=False)
            if path == "/cache/clear":
                self._touch()
                n = tts.clear_cache()
                return self._send(200, {"ok": True, "cleared": n})
            if path == "/unload":
                self._touch()
                done = tts.unload()
                if done:
                    state["unloads"] += 1
                return self._send(200, {"ok": True, "unloaded": done})
            return self._send(404, {"ok": False, "error": "not found"})

        def _tts(self, binary):
            self._touch()
            state["inflight"] += 1
            try:
                return self._do_tts(binary)
            finally:
                state["inflight"] -= 1
                self._touch()

        def _do_tts(self, binary):
            req = self._read_json()
            text = (req.get("text") or "").strip()
            if not text:
                return self._send(400, {"ok": False, "error": "text 不能为空"})
            seed = req.get("seed")
            # 亮度补偿：不传则用 TTS 实例的默认值（0 = 关闭）
            br = req.get("brightness_db")
            t0 = time.time()
            was_loaded = tts.is_loaded
            try:
                with lock:                     # 后端非线程安全，串行化
                    r = tts.say(text, seed=seed, use_cache=True,
                                brightness_db=br)
            except Exception as e:
                return self._send(500, {"ok": False,
                                        "error": f"{type(e).__name__}: {e}"})
            if not was_loaded and tts.is_loaded:
                state["loads"] += 1
            if binary:
                return self._send(200, r.to_wav_bytes(),
                                  ctype="audio/wav")
            return self._send(200, {
                "ok": True, "duration": round(r.duration, 3),
                "sr": r.sr, "cached": r.cached,
                "elapsed": round(time.time() - t0, 3),
                "cold_start": not was_loaded,
                # 实际送去合成的文本（英文已转成中文读法）。
                # 与请求里的 text 不同时，调用方可据此告诉用户"实际念的是什么"。
                "spoken": getattr(r, "spoken", text),
                "wav_base64": base64.b64encode(r.to_wav_bytes()).decode()})

    return Handler


def _idle_watchdog(tts, state, idle_timeout, verbose=True):
    """空闲超时后释放模型，腾出内存/显存；下次请求按需重新加载

    注意：有在途请求时绝不释放——长合成可能远超 idle_timeout，
    若此时卸载，模型会在推理过程中被回收。
    """
    while not state.get("stop"):
        time.sleep(min(5, max(1, idle_timeout / 4)))
        if state.get("stop"):
            break
        if state.get("inflight", 0) > 0:
            continue                     # 正在合成，跳过本轮
        if not tts.is_loaded:
            continue
        idle = time.time() - state["last_activity"]
        if idle >= idle_timeout:
            if tts.unload():
                state["unloads"] += 1
                if verbose:
                    print(f"[idle] 空闲 {idle:.0f}s ≥ {idle_timeout}s，"
                          f"已释放模型（下次请求将重新加载）", flush=True)


def serve(tts, host="127.0.0.1", port=8765, idle_timeout=0, preload=True):
    """
    idle_timeout: 空闲多少秒后自动释放模型；0 = 一直常驻（默认）
    preload:      启动时是否预先加载模型（预热）
    """
    state = {"last_activity": time.time(), "loads": 0, "unloads": 0,
             "idle_timeout": idle_timeout}

    if preload:
        print("预热推理后端 ...")
        t0 = time.time()
        _ = tts.backend
        state["loads"] = 1
        print(f"就绪（{time.time()-t0:.1f}s）")
    else:
        print("已跳过预热（模型将在首次请求时加载）")

    if idle_timeout and idle_timeout > 0:
        th = threading.Thread(target=_idle_watchdog,
                              args=(tts, state, idle_timeout), daemon=True)
        th.start()
        print(f"常驻策略: 空闲 {idle_timeout}s 后自动释放模型")
    else:
        print("常驻策略: 一直常驻（模型不释放）")

    httpd = ThreadingHTTPServer((host, port), make_handler(tts, state))
    print(f"服务已启动: http://{host}:{port}")
    print(f"  健康检查  GET  /health     （含 resident / idle_seconds）")
    print(f"  合成      POST /tts        body: {{\"text\":\"...\"}} → audio/wav")
    print(f"  合成(JSON) POST /tts.json   → {{ok,duration,wav_base64}}")
    print(f"  手动释放  POST /unload")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        state["stop"] = True
        httpd.server_close()
