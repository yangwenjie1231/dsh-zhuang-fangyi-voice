# -*- coding: utf-8 -*-
"""高层 API：统一入口 + 合成缓存

    from zfh_voice import TTS
    tts = TTS()                       # 默认 onnx 后端
    r = tts.say("今天天气不错")        # 返回 SynthResult
    r.save("out.wav")

缓存：以 (文本, 后端, 参考音频, seed) 为键，命中直接返回，避免重复合成。

英文处理：中文文本前端会**静默删掉英文**（见 textprep 模块头），
所以合成前统一走一遍 textprep.localize()，把英文转成中文读法。
"""
import hashlib
import json
import os
import re
import time

from . import audio, paths, textprep
from .backends import make_backend

DEFAULT_REF_NAME = "ref/default.wav"
DEFAULT_REF_TEXT_NAME = "ref/default.txt"

# 随模型包一起分发的默认参考音频（若无则回落到内置示例）
FALLBACK_REF_TEXT = "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"

# 前端能发音的内容：汉字或数字。
# 数字/符号前端会自己转写（3→三、￥100→幺零零、50%→百分之五十），
# 所以算"可发音"。英文字母是否算，取决于走哪条路（见 _has_speakable）。
_SPEAKABLE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf0-9]")
_SPEAKABLE_LATIN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf0-9A-Za-z]")


def _has_speakable(text, allow_latin=False):
    """文本里有没有能发音的内容

    allow_latin=True 表示英文段能产出音素（中英混排可用时），
    此时纯英文/含英文的句子是可发音的。
    """
    rx = _SPEAKABLE_LATIN if allow_latin else _SPEAKABLE
    return bool(rx.search(text or ""))


class SynthResult:
    def __init__(self, wav, sr, text, cached=False, seconds=0.0, backend=""):
        self.wav = wav
        self.sr = sr
        self.text = text              # 用户原始输入
        self.spoken = text            # 实际送去合成的文本（英文已转中文读法）
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
                 ref_wav=None, ref_text=None, use_cache=True, localize="auto",
                 **backend_kw):
        self._backend_kind = backend
        # 按后端分别校验：onnx 后端必须有 7 个 ONNX；
        # torch 后端只认 .ckpt/.pth，**不要求 ONNX 存在**（两者互不通用）
        require = "onnx" if backend == "onnx" else None
        self.model_dir = paths.resolve_model_dir(model_dir, require=require)
        self.use_cache = use_cache
        # 英文怎么念：auto（默认）/ True（转中文读法）/ False（原样送前端）
        #   auto → 中英混排可用时**原样送**（真英文发音，实测更好）；
        #          不可用时转中文读法兜底（否则英文会被静默删掉）
        self.localize = localize
        self._localize_auto = None      # 惰性求值，避免构造时就加载前端
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

    # ---------- 英文处理策略 ----------
    def _resolve_localize(self, override=None):
        """这一句要不要把英文转成中文读法

        `localize` 语义：
          · "auto"（默认）→ 中英混排可用时**原样送**（真英文发音，实测更准）；
                            不可用时转中文读法兜底（否则英文被静默删掉）
          · True  → 总是转（纯离线、零 nltk 依赖）
          · False → 总是不转（英文交给前端分段处理）

        ⚠️ 注意 `"auto"` 是**字符串**，不能直接 `bool()` ——
        `bool("auto")` 是 True，会把"自动"误判成"总是转换"。
        """
        if override is not None:
            return self._decide_localize(override)
        if isinstance(self.localize, str):
            return self._decide_localize(self.localize)
        return bool(self.localize)

    def _decide_localize(self, value):
        """把 'auto' / True / False 解析成"这一句是否转换" """
        if isinstance(value, str):
            if value.strip().lower() != "auto":
                # 未知字符串按 auto 处理，避免静默改变行为
                pass
            else:
                return self._auto_localize()
        return bool(value)

    def _auto_localize(self):
        """中英混排可用 → 不转换（用真英文念）；否则转换（兜底）"""
        if self._localize_auto is None:
            try:
                from . import frontend
                self._localize_auto = not frontend.mixed_available()
            except Exception:
                self._localize_auto = True
        return self._localize_auto

    # ---------- 后端 ----------
    @property
    def backend(self):
        if self._backend is None:
            t0 = time.time()
            self._backend = make_backend(
                self._backend_kind, model_dir=self.model_dir, **self._backend_kw)
            self._load_seconds = time.time() - t0
        return self._backend

    @property
    def is_loaded(self):
        """模型当前是否常驻在内存/显存里"""
        return self._backend is not None

    def preload(self):
        """预加载（把加载耗时前移，避免第一次合成等待）"""
        _ = self.backend
        return self._load_seconds

    def unload(self):
        """释放模型，腾出内存与显存；下次合成时按需重新加载"""
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None
            import gc
            gc.collect()
            return True
        return False

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
    def say(self, text, seed=None, out=None, use_cache=None, verbose=False,
            localize=None):
        """合成一句，返回 SynthResult

        localize: 是否把文本里的英文转成中文读法（默认跟随 self.localize）。
                  **必须做** —— 中文文本前端会**静默删掉**英文
                  （chinese2.py: `re.sub("[a-zA-Z]+", "", seg)`），
                  不转换的话 `GPU`、`Hello` 会被整句吃掉。
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("文本为空")
        do_localize = self._resolve_localize(localize)
        spoken = textprep.localize(text) if do_localize else text
        # 提前拦住"念不出声"的输入 —— 否则后端会产出 0 音素并抛出
        # 难以理解的底层错误（如 "need at least one array to concatenate"）。
        #
        # 判定要区分两条路：
        #   · 中英混排可用 → 英文段能走 english.py 产出音素，所以**英文也算可发音**
        #   · 不可用（英文会被前端删掉）→ 只有汉字/数字算可发音
        allow_latin = not do_localize
        if not _has_speakable(spoken, allow_latin=allow_latin):
            raise ValueError(
                f"文本里没有可发音的内容：{text!r}。"
                + ("纯符号请换一句。" if allow_latin else
                   "当前环境不支持英文发音（缺 nltk/g2p_en 等），"
                   "请设 TTS(localize=True) 用中文读法念英文。"))
        use_cache = self.use_cache if use_cache is None else use_cache
        # 缓存键用**转换后**的文本：同一句的不同写法若读法相同，可复用
        key = self._cache_key(spoken, seed)
        cp = self._cache_path(key)

        if use_cache and os.path.exists(cp):
            wav, sr = audio.load_wav(cp)
            r = SynthResult(wav, sr, text, cached=True, backend=self._backend_kind)
            r.spoken = spoken
            if out:
                r.save(out)
            return r

        t0 = time.time()
        wav, sr = self.backend.synth(
            spoken, self.ref_wav, self.ref_text, seed=seed, verbose=verbose)
        dt = time.time() - t0
        wav = audio.peak_normalize(wav, 0.95)
        if use_cache:
            audio.save_wav(cp, wav, sr)
        r = SynthResult(wav, sr, text, cached=False, seconds=dt,
                        backend=self._backend_kind)
        r.spoken = spoken
        if out:
            r.save(out)
        return r

    def say_many(self, texts, out_dir, seed=None, verbose=True, localize=None):
        """批量合成，返回结果列表"""
        os.makedirs(out_dir, exist_ok=True)
        out = []
        for i, t in enumerate(texts, 1):
            t = (t or "").strip()
            if not t:
                continue
            p = os.path.join(out_dir, f"{i:03d}.wav")
            r = self.say(t, seed=seed, out=p, verbose=False, localize=localize)
            out.append(r)
            if verbose:
                mark = "缓存" if r.cached else f"{r.seconds:.1f}s"
                print(f"  [{i}/{len(texts)}] {r.duration:5.2f}s {mark:>8}  {t[:30]}")
        return out

    def close(self):
        """同 unload()，语义上表示"用完释放" """
        self.unload()
