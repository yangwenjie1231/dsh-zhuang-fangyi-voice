# -*- coding: utf-8 -*-
"""dsh-zhuang-fangyi-voice —— 庄方宜专属中文音色推理库

支持多种推理后端与调用方式：

后端（backend）
  onnx   仅依赖 onnxruntime，无 torch / 无 CUDA，跨平台，包体小
  torch  复用 GPT-SoVITS 原版实现，GPU 下最快

入口
  Python API  from zfh_voice import TTS; TTS().say("台词")
  CLI         python -m zfh_voice say "台词" -o out.wav
  HTTP        python -m zfh_voice serve   （POST /tts）
"""
from .api import TTS, SynthResult  # noqa: F401
from .paths import resolve_model_dir, model_status  # noqa: F401

__version__ = "0.1.0"
__all__ = ["TTS", "SynthResult", "resolve_model_dir", "model_status"]
