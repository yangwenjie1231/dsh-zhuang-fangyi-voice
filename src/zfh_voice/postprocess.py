# -*- coding: utf-8 -*-
"""校准式亮度补偿（high-shelf）

## 为什么需要

模型输出 32 kHz（v2/v2Pro），而素材是 48 kHz。48k→32k 的下采样在
Nyquist(16 kHz) 附近有抗混叠滤波滚降，实测造成 12–16 kHz 频段比原声低
约 2.8 dB。听感上表现为「糊」。

实测数据（`D:\\A\\voiceclone\\calib_brightness.py`）：
    频段        48k 原声   合成(未处理)   补偿后
    8–12 kHz     -34.7      -33.9        -33.8     （+0.09，几乎不动）
    12–16 kHz    -46.8      -49.6        -46.7     （+2.9，对齐原声）

## 诚实说明

这是**听感补偿**，恢复的是重采样滤波造成的频谱倾斜，
**不是信息恢复** —— 12–16 kHz 的真实内容在 32 kHz 里仍然受限。
参数经网格搜索校准（`fc=12k, gain=3dB, Q=1.0`），12–16 kHz 误差 0.09 dB。

## 默认关闭

`gain_db=0.0` 时**直接返回入参，逐字节不变**（短路，不做任何滤波）。
"""
from __future__ import annotations

import math

import numpy as np

# 校准所得（见 calib_brightness.json）
DEFAULT_FC = 12000.0
DEFAULT_Q = 1.0
DEFAULT_GAIN_DB = 0.0


def shelf_coeffs(sr: float, fc: float, gain_db: float, q: float):
    """RBJ audio-EQ-cookbook 高通搁架双二阶系数 → (b, a)。

    b/a 均为长度 3 的 float64 数组，a[0] 已归一为 1。
    """
    if sr <= 0:
        raise ValueError(f"采样率必须为正: {sr}")
    if fc <= 0 or fc >= sr / 2:
        raise ValueError(f"截止频率须在 (0, sr/2) 内: fc={fc}, sr={sr}")
    if q <= 0:
        raise ValueError(f"Q 必须为正: {q}")

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    sqA = math.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cw + 2 * sqA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * sqA * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * sqA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * sqA * alpha

    return (np.array([b0, b1, b2], dtype=np.float64) / a0,
            np.array([a0, a1, a2], dtype=np.float64) / a0)


def _lfilter_numpy(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """纯 numpy 直接 II 型转置实现（scipy 不可用时的回退）。

    逐样本循环，仅用于没有 scipy 的环境。32 kHz 下 1 秒音频约 0.1 s。
    """
    n = len(x)
    y = np.empty(n, dtype=np.float64)
    z1 = z2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for i in range(n):
        xi = float(x[i])
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """优先用 scipy（快），否则回退纯 numpy。"""
    try:
        from scipy.signal import lfilter  # type: ignore
        return lfilter(b, a, x)
    except Exception:  # noqa: BLE001  —— 没装 scipy / 导入异常都走回退
        return _lfilter_numpy(b, a, x)


def apply_brightness(audio, sr: int, gain_db: float = DEFAULT_GAIN_DB,
                     fc: float = DEFAULT_FC, q: float = DEFAULT_Q):
    """对音频施加高通搁架。返回与入参**同 dtype** 的数组。

    gain_db=0 时**原样返回入参**（逐字节不变），不做任何滤波 ——
    这保证「默认关闭」是真正的零风险。

    支持 int16（torch 后端）与 float32/float64（ONNX 后端）输入。
    """
    if gain_db == 0.0:
        return audio                      # 短路：零改动

    arr = np.asarray(audio)
    if arr.size == 0:
        return audio
    if arr.ndim > 1:                       # 多声道：逐声道处理
        chans = [apply_brightness(arr[:, c], sr, gain_db, fc, q)
                 for c in range(arr.shape[1])]
        return np.stack(chans, axis=1).astype(arr.dtype, copy=False)

    is_int = np.issubdtype(arr.dtype, np.integer)
    if is_int:
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        x = arr.astype(np.float64) / scale
    else:
        scale = None
        x = arr.astype(np.float64)

    b, a = shelf_coeffs(sr, fc, gain_db, q)
    y = _lfilter(b, a, x)

    if is_int:
        y = np.clip(y, -1.0, 1.0) * scale
        return np.rint(y).astype(arr.dtype)
    return y.astype(arr.dtype, copy=False)


def describe(gain_db: float = DEFAULT_GAIN_DB, fc: float = DEFAULT_FC,
             q: float = DEFAULT_Q) -> str:
    """一行人类可读描述，用于日志与体检报告。"""
    if gain_db == 0.0:
        return "亮度补偿: 关闭 (0 dB)"
    return f"亮度补偿: +{gain_db:g} dB @ {fc:g} Hz (Q={q:g})"
