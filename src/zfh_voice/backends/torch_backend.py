# -*- coding: utf-8 -*-
"""torch 后端：复用 GPT-SoVITS 原版实现

优点：GPU 下最快（实测 RTF 0.45，ONNX 约 6.1），支持 fp16 等全部特性
缺点：需要 torch 环境 + 一份 GPT-SoVITS 源码检出，体积大

用法：
    OnnxBackend(...)                                    # 便携
    TorchBackend(gsv_root=r"D:\\GPT-SoVITS", ...)        # 快

gsv_root 需包含 GPT_SoVITS/ 目录（可从官方仓库 clone）。
权重文件放在 <gsv_root>/GPT_weights_v2 与 SoVITS_weights_v2，
或通过 gpt_path / sovits_path 显式指定。
"""
import os
import sys
import threading

# os.chdir 是进程级状态：推理期间要切到 gsv_root（上游用相对路径），
# 所以合成必须串行 —— 否则并发调用会互相把 cwd 搅乱。
_SYNTH_LOCK = threading.Lock()

import numpy as np

from .base import BackendError, SynthBackend


class TorchBackend(SynthBackend):
    name = "torch"

    def __init__(self, gsv_root, model_dir=None, gpt_path=None,
                 sovits_path=None, ref_wav=None, ref_text=None,
                 device="cuda", is_half=True, exp_name="zfh",
                 bert_dir=None, hubert_dir=None):
        super().__init__(model_dir)
        # 解析模型目录（用于兜底查找 torch_weights/；找不到也不影响，
        # 因为 torch 权重通常直接从 gsv_root 里取）
        if self.model_dir:
            from .. import paths
            try:
                self.model_dir = paths.resolve_model_dir(self.model_dir)
            except FileNotFoundError:
                self.model_dir = os.path.abspath(self.model_dir)
        self.gsv_root = os.path.abspath(gsv_root)
        if not os.path.isdir(os.path.join(self.gsv_root, "GPT_SoVITS")):
            raise BackendError(
                f"gsv_root 下找不到 GPT_SoVITS/ 目录: {self.gsv_root}\n"
                "请 clone https://github.com/RVC-Boss/GPT-SoVITS 并指定其根目录。")

        sys.path[:0] = [self.gsv_root, os.path.join(self.gsv_root, "GPT_SoVITS")]
        prev_cwd = os.getcwd()
        os.chdir(self.gsv_root)          # 上游大量使用相对路径
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS as _TTS
        finally:
            os.chdir(prev_cwd)

        gpt = gpt_path or os.path.join(
            self.gsv_root, "GPT_weights_v2", f"{exp_name}-e4.ckpt")
        sov = sovits_path or os.path.join(
            self.gsv_root, "SoVITS_weights_v2", f"{exp_name}_e6_s186.pth")
        # 回退：`download_models.py --with-torch` 会把权重放在
        # <模型目录>/torch_weights/，这里自动兜住，省得手动搬文件
        if self.model_dir:
            if not os.path.exists(gpt):
                cand = os.path.join(self.model_dir, "torch_weights",
                                    f"{exp_name}-e4.ckpt")
                if os.path.exists(cand):
                    gpt = cand
            if not os.path.exists(sov):
                cand = os.path.join(self.model_dir, "torch_weights",
                                    f"{exp_name}_e6_s186.pth")
                if os.path.exists(cand):
                    sov = cand
        for p in (gpt, sov):
            if not os.path.exists(p):
                raise BackendError(
                    f"找不到 torch 权重: {p}\n"
                    "torch 后端需要 .ckpt/.pth —— **ONNX 文件无法被 torch 加载**。\n"
                    "下载方式：python download_models.py --with-torch\n"
                    "或用 gpt_path / sovits_path 显式指定路径。")

        bert = bert_dir or os.path.join(
            self.gsv_root, "GPT_SoVITS", "pretrained_models",
            "chinese-roberta-wwm-ext-large")
        hubert = hubert_dir or os.path.join(
            self.gsv_root, "GPT_SoVITS", "pretrained_models",
            "chinese-hubert-base")

        # 上游用相对路径，这里换成绝对路径副本
        cfg = {"custom": {
            "device": device, "is_half": is_half, "version": "v2",
            "t2s_weights_path": gpt, "vits_weights_path": sov,
            "bert_base_path": bert, "cnhuhbert_base_path": hubert,
        }}
        prev_cwd = os.getcwd()
        os.chdir(self.gsv_root)
        try:
            self.tts = _TTS(cfg)
        finally:
            os.chdir(prev_cwd)

        self.ref_wav = ref_wav
        self.ref_text = ref_text
        self.device = device

    def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
        import numpy as np
        ref_wav = ref_wav or self.ref_wav
        ref_text = ref_text or self.ref_text
        if not (ref_wav and ref_text):
            raise BackendError("torch 后端需要指定参考音频与参考文本")

        inputs = {
            "text": text, "text_lang": "zh",
            "ref_audio_path": ref_wav, "prompt_text": ref_text,
            "prompt_lang": "zh",
            "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
        }
        if seed is not None:
            inputs["seed"] = int(seed)
        sr, wav = None, None
        # ⚠️ **推理期间也必须 chdir 到 gsv_root** —— 这不是洁癖，是上游的硬依赖：
        # GPT-SoVITS 内部大量使用**相对路径**找文本前端资源
        #（`GPT_SoVITS/text/G2PWModel/`、pretrained_models 等）。
        # 只在上面的 `__init__` 里 chdir 而这里不切，症状非常隐蔽：
        # 它会去**联网下载** g2pw 模型（打印 "Downloading g2pw model..."），
        # 下载失败就静默卡住 —— 看起来像"模型没装"，其实文件就在 gsv_root 里。
        # 实测：cwd 在 GSV 根目录 → 3.5 秒出音频；cwd 在别处 → 卡在下载。
        #
        # `os.chdir` 是**进程级**状态，所以必须串行（HTTP 服务那边也有 lock，
        # 这里再加一道，防止别的调用方并发进来把 cwd 搅乱）。
        with _SYNTH_LOCK:
            prev_cwd = os.getcwd()
            try:
                os.chdir(self.gsv_root)
                for _sr, _chunk in self.tts.run(inputs):   # run() 是生成器
                    sr, wav = _sr, _chunk
            finally:
                os.chdir(prev_cwd)
        if wav is None:
            raise BackendError("合成失败：未返回音频")
        wav = np.asarray(wav)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
            if np.abs(wav).max() > 2:              # int16
                wav = wav / 32768.0
        return wav.reshape(-1), sr

    def close(self):
        try:
            del self.tts
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
