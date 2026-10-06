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

import numpy as np

from .base import BackendError, SynthBackend


class TorchBackend(SynthBackend):
    name = "torch"

    def __init__(self, gsv_root, model_dir=None, gpt_path=None,
                 sovits_path=None, ref_wav=None, ref_text=None,
                 device="cuda", is_half=True, exp_name="zfh",
                 bert_dir=None, hubert_dir=None):
        super().__init__(model_dir)
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
        for p in (gpt, sov):
            if not os.path.exists(p):
                raise BackendError(
                    f"找不到权重: {p}\n"
                    "请把 Release 中的 .ckpt / .pth 放到对应目录，"
                    "或用 gpt_path / sovits_path 指定。")

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
        for _sr, _chunk in self.tts.run(inputs):   # run() 是生成器
            sr, wav = _sr, _chunk
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
