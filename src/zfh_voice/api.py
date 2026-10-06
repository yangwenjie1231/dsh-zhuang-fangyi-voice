# -*- coding: utf-8 -*-
"""高层 API：统一入口 + 合成缓存

    from zfh_voice import TTS
    tts = TTS()                       # 默认 onnx 后端
    r = tts.say("今天天气不错")        # 返回 SynthResult
    r.save("out.wav")

缓存：以 (文本, 后端, 参考音频, seed) 为键，命中直接返回，避免重复合成。
"""
import hashlib
import json
import os
import time

from . import audio, paths
from .backends import make_backend

DEFAULT_REF_NAME = "ref/default.wav"
DEFAULT_REF_TEXT_NAME = "ref/default.txt"

# 随模型包一起分发的默认参考音频（若无则回落到内置示例）
FALLBACK_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"


class SynthResult:
    def __init__(self, wav, sr, text, cached=False, seconds=0.0, backend=""):
        self.wav = wav
        self.sr = sr
        self.text = text
        self.cached = cached
        self.seconds = seconds
        self.backend = backend

    @property
    def duration(self):
        return len(self.wav) / self.sr

    def save(self, path):
        audio.save_wav(path, self.wav, self.sr)
        return path

    def to_wav_bytes(self):
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, self.wav, self.sr, format="WAV")
        return buf.getvalue()

    def __repr__(self):
        tag = "缓存" if self.cached else f"{self.seconds:.1f}s"
        return f"<SynthResult {self.duration:.2f}s @{self.sr} {tag}>"


class TTS:
    def __init__(self, backend="onnx", model_dir=None, cache_dir=None,
                 ref_wav=None, ref_text=None, use_cache=True, **backend_kw):
        self._backend_kind = backend
        # 按后端分别校验：onnx 后端必须有 7 个 ONNX；
        # torch 后端只认 .ckpt/.pth，**不要求 ONNX 存在**（两者互不通用）
        require = "onnx" if backend == "onnx" else None
        self.model_dir = paths.resolve_model_dir(model_dir, require=require)
        self.use_cache = use_cache
        self._backend_kw = backend_kw
        self._backend = None

        # 参考音频：显式指定 > 模型目录里的默认 > 报错
        self.ref_wav = ref_wav or self._default_ref()
        self.ref_text = ref_text or self._default_ref_text()
        if not self.ref_wav or not os.path.exists(self.ref_wav):
            extra = ""
            if backend != "onnx":
                extra = ("\n（torch 后端同样需要参考音频；"
                         "它来自 zfh-voice-aux.zip，不随 torch 权重一起下载）")
            raise FileNotFoundError(
                f"找不到默认参考音频（期望路径："
                f"{os.path.join(self.model_dir, DEFAULT_REF_NAME)}）。\n"
                f"两种解决方式：\n"
                f"  1) 用 ref_wav= 与 ref_text= 指定自己的参考音频\n"
                f"     （3~10 秒目标音色干声，文本需逐字对应）\n"
                f"  2) 下载辅助包：python download_models.py --with-torch\n"
                f"     （辅助包含 G2PW 数据、tokenizer 与默认参考音频）" + extra)

        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(self.model_dir), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ---------- 参考音频 ----------
    def _default_ref(self):
        p = os.path.join(self.model_dir, DEFAULT_REF_NAME)
        return p if os.path.exists(p) else None

    def _default_ref_text(self):
        p = os.path.join(self.model_dir, DEFAULT_REF_TEXT_NAME)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
        return FALLBACK_REF_TEXT

    # ---------- 后端 ----------
    @property
    def backend(self):
        if self._backend is None:
            t0 = time.time()
            self._backend = make_backend(
                self._backend_kind, model_dir=self.model_dir, **self._backend_kw)
            self._load_seconds = time.time() - t0
        return self._backend

    # ---------- 缓存 ----------
    def _cache_key(self, text, seed):
        h = hashlib.sha1()
        h.update(text.encode("utf-8"))
        h.update(str(seed).encode())
        h.update(self._backend_kind.encode())
        try:
            st = os.stat(self.ref_wav)
            h.update(f"{self.ref_wav}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(self.ref_wav.encode())
        return h.hexdigest()[:16]

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.wav")

    def clear_cache(self):
        n = 0
        for f in os.listdir(self.cache_dir):
            if f.endswith(".wav"):
                os.remove(os.path.join(self.cache_dir, f))
                n += 1
        return n

    # ---------- 合成 ----------
    def say(self, text, seed=None, out=None, use_cache=None, verbose=False):
        """合成一句，返回 SynthResult"""
        text = (text or "").strip()
        if not text:
            raise ValueError("文本为空")
        use_cache = self.use_cache if use_cache is None else use_cache
        key = self._cache_key(text, seed)
        cp = self._cache_path(key)

        if use_cache and os.path.exists(cp):
            wav, sr = audio.load_wav(cp)
            r = SynthResult(wav, sr, text, cached=True, backend=self._backend_kind)
            if out:
                r.save(out)
            return r

        t0 = time.time()
        wav, sr = self.backend.synth(
            text, self.ref_wav, self.ref_text, seed=seed, verbose=verbose)
        dt = time.time() - t0
        wav = audio.peak_normalize(wav, 0.95)
        if use_cache:
            audio.save_wav(cp, wav, sr)
        r = SynthResult(wav, sr, text, cached=False, seconds=dt,
                        backend=self._backend_kind)
        if out:
            r.save(out)
        return r

    def say_many(self, texts, out_dir, seed=None, verbose=True):
        """批量合成，返回结果列表"""
        os.makedirs(out_dir, exist_ok=True)
        out = []
        for i, t in enumerate(texts, 1):
            t = (t or "").strip()
            if not t:
                continue
            p = os.path.join(out_dir, f"{i:03d}.wav")
            r = self.say(t, seed=seed, out=p, verbose=False)
            out.append(r)
            if verbose:
                mark = "缓存" if r.cached else f"{r.seconds:.1f}s"
                print(f"  [{i}/{len(texts)}] {r.duration:5.2f}s {mark:>8}  {t[:30]}")
        return out

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
