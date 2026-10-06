# -*- coding: utf-8 -*-
"""后端抽象与共享常量（GPT-SoVITS v2）"""
import os

import numpy as np

# ---- v2 模型常量 ----
VERSION = "v2"
EOS = 1024
HZ = 50                    # 语义 token 帧率
MAX_SEC = 54               # 单句最长秒数
EARLY_STOP = HZ * MAX_SEC  # 提前终止阈值
MAX_LOOP = 1500            # 自回归最大步数
SAMPLE_RATE = 32000        # 输出采样率

# 参考音频限制（与上游一致）
REF_MIN_SEC = 3.0
REF_MAX_SEC = 10.0

# 上游怪癖：HuBERT 输入（16k）后面追加「按 32k 计算」的 0.3 秒静音
# 即 9600 个样本。必须精确复刻，否则音色会偏移。
ZERO_PAD_SAMPLES = int(SAMPLE_RATE * 0.3)


class BackendError(RuntimeError):
    pass


class SynthBackend:
    """推理后端接口"""

    name = "base"

    def __init__(self, model_dir=None, **kw):
        self.model_dir = model_dir

    def synth(self, text, ref_wav, ref_text, **params):
        """返回 (wav: np.float32 mono, sr: int)"""
        raise NotImplementedError

    def close(self):
        pass

    # ---- 共用工具 ----
    @staticmethod
    def check_ref_duration(seconds):
        if not (REF_MIN_SEC <= seconds <= REF_MAX_SEC):
            raise ValueError(
                f"参考音频时长 {seconds:.2f}s 超出 {REF_MIN_SEC}~{REF_MAX_SEC}s 范围。"
                "请换一段干净的单句语音。")

    @staticmethod
    def zero_pad_for_hubert(wav16k):
        """复刻上游的静音追加行为"""
        return np.concatenate(
            [wav16k.astype(np.float32),
             np.zeros(ZERO_PAD_SAMPLES, dtype=np.float32)])
