# -*- coding: utf-8 -*-
"""亮度补偿（postprocess）单元测试

关键约束：
  · gain_db=0 必须**逐字节不变**（零风险默认）
  · +3dB 只抬 12–16kHz，不能顺带抬 8–12kHz 或低频
  · int16（torch 后端）与 float32（ONNX 后端）都要支持
"""
import numpy as np
import pytest

from zfh_voice import postprocess as pp

SR = 32000


def band_db(x, sr, lo, hi):
    """某频段的平均功率（dB）。"""
    X = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(X[m].mean() + 1e-20)


@pytest.fixture
def noise():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, SR).astype(np.float32)


def test_zero_gain_returns_identical_object(noise):
    """gain_db=0 必须原样返回 —— 默认路径零风险"""
    assert pp.apply_brightness(noise, SR, 0.0) is noise


def test_zero_gain_is_default(noise):
    """默认参数即关闭"""
    assert pp.DEFAULT_GAIN_DB == 0.0
    assert pp.apply_brightness(noise, SR) is noise


def test_three_db_raises_high_band(noise):
    """+3dB 应把 12–16kHz 抬高约 3dB"""
    y = pp.apply_brightness(noise, SR, 3.0)
    d = band_db(y, SR, 12000, 16000) - band_db(noise, SR, 12000, 16000)
    assert 2.0 < d < 4.5, f"12-16k 提升 {d:.2f} dB，超出预期"


def test_three_db_spares_mid_band(noise):
    """8–12kHz 必须基本不动（防过亮）"""
    y = pp.apply_brightness(noise, SR, 3.0)
    d = band_db(y, SR, 8000, 12000) - band_db(noise, SR, 8000, 12000)
    assert d < 1.0, f"8-12k 被抬了 {d:.2f} dB，会过亮"


def test_three_db_spares_low_band(noise):
    """低频不能被动到（否则音色会变薄）"""
    y = pp.apply_brightness(noise, SR, 3.0)
    d = band_db(y, SR, 0, 4000) - band_db(noise, SR, 0, 4000)
    assert abs(d) < 0.5, f"0-4k 变化 {d:.2f} dB"


def test_int16_dtype_preserved(noise):
    """torch 后端给的是 int16，必须原 dtype 返回且不溢出"""
    xi = (noise * 32767).astype(np.int16)
    yi = pp.apply_brightness(xi, SR, 3.0)
    assert yi.dtype == np.int16
    assert yi.min() >= -32768 and yi.max() <= 32767


def test_int16_zero_gain_untouched(noise):
    xi = (noise * 32767).astype(np.int16)
    assert pp.apply_brightness(xi, SR, 0.0) is xi


def test_float32_dtype_preserved(noise):
    y = pp.apply_brightness(noise, SR, 3.0)
    assert y.dtype == np.float32


def test_stereo_supported():
    """多声道逐声道处理，形状不变"""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, (SR, 2)).astype(np.float32)
    y = pp.apply_brightness(x, SR, 3.0)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_empty_input():
    e = np.array([], dtype=np.float32)
    assert pp.apply_brightness(e, SR, 3.0) is e


@pytest.mark.parametrize("fc", [0, -100, 16000, 20000])
def test_invalid_fc_rejected(fc):
    with pytest.raises(ValueError):
        pp.shelf_coeffs(SR, fc, 3.0, 1.0)


@pytest.mark.parametrize("q", [0, -1.0])
def test_invalid_q_rejected(q):
    with pytest.raises(ValueError):
        pp.shelf_coeffs(SR, q=q, fc=12000.0, gain_db=3.0)


def test_invalid_sr_rejected():
    with pytest.raises(ValueError):
        pp.shelf_coeffs(0, 12000.0, 3.0, 1.0)


def test_coeffs_normalized():
    """a[0] 必须是 1，否则滤波结果整体缩放错误"""
    b, a = pp.shelf_coeffs(SR, 12000.0, 3.0, 1.0)
    assert a[0] == pytest.approx(1.0)
    assert len(b) == 3 and len(a) == 3


def test_numpy_fallback_matches_scipy(noise):
    """没有 scipy 时的纯 numpy 回退必须与 scipy 结果一致"""
    b, a = pp.shelf_coeffs(SR, 12000.0, 3.0, 1.0)
    x = noise.astype(np.float64)
    try:
        from scipy.signal import lfilter
    except ImportError:
        pytest.skip("无 scipy，跳过对比")
    ref = lfilter(b, a, x)
    got = pp._lfilter_numpy(b, a, x)
    assert np.allclose(ref, got, atol=1e-6)


def test_describe():
    assert "关闭" in pp.describe(0.0)
    assert "3" in pp.describe(3.0)


# ---------- 按模型版本选补偿量 ----------
#
# 背景（实测）：v2 的 8–12k 偏亮 4.2dB、12–16k 已对齐，所以需要 +3dB；
# 而 v2Pro 的频谱本身就接近原声，加 3dB 会**过冲**（长文本频谱偏差 0.88 → 1.92）。
# 两者起点不同，所以补偿量必须按版本取，不能全局一个值。

def test_gain_v2_needs_compensation():
    assert pp.gain_for_version("v2") == 3.0
    assert pp.gain_for_version("v1") == 3.0


def test_gain_v2pro_needs_none():
    """v2Pro 不该补偿 —— 加了会过冲"""
    assert pp.gain_for_version("v2Pro") == 0.0
    assert pp.gain_for_version("v2ProPlus") == 0.0


def test_gain_unknown_version_is_safe():
    """未知版本保守返回 0 —— 宁可不动，也不要凭猜测加高频"""
    for v in ("v9", "unknown", "", None):
        assert pp.gain_for_version(v) == 0.0, f"{v!r} 应返回 0"


def test_gain_for_version_accepts_default():
    assert pp.gain_for_version("v9", default=1.5) == 1.5
    assert pp.gain_for_version("v2", default=1.5) == 3.0   # 已知版本仍用表


def test_by_version_covers_all_known():
    """表里应覆盖上游全部 6 个版本（否则用户换版本时会静默落到 0）"""
    for v in ("v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"):
        assert v in pp.BY_VERSION, f"{v} 未在 BY_VERSION 里"
