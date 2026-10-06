# -*- coding: utf-8 -*-
"""冒烟测试：验证模型就绪 + 能合成出非静音音频

运行：
    pytest -q                     # 需先下载模型
    或直接 python tests/test_smoke.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import model_status, resolve_model_dir  # noqa: E402
from zfh_voice.paths import ONNX_FILES  # noqa: E402


def _models_ready():
    try:
        resolve_model_dir()
        return True
    except FileNotFoundError:
        return False


requires_models = pytest.mark.skipif(
    not _models_ready(), reason="模型未下载，先运行 python download_models.py")


def test_model_status_reports_all_files():
    d, missing, ok = model_status()
    assert isinstance(d, str)
    if not ok:
        # 未下载模型时，缺失项应覆盖到 ONNX 文件
        assert set(missing) & set(ONNX_FILES)


@requires_models
def test_resolve_model_dir():
    d = resolve_model_dir()
    assert os.path.isdir(d)
    for f in ONNX_FILES:
        assert os.path.exists(os.path.join(d, f)), f


@requires_models
def test_synth_short_text():
    from zfh_voice import TTS

    tts = TTS()
    try:
        r = tts.say("你好。", use_cache=False)
        assert r.sr in (32000, 24000, 44100, 48000)
        assert r.duration > 0.3
        assert np.abs(r.wav).max() > 0.01, "输出近似静音"
    finally:
        tts.close()


@requires_models
def test_cache_hit_is_fast():
    from zfh_voice import TTS

    tts = TTS()
    try:
        text = "缓存测试：这句话应该被缓存。"
        r1 = tts.say(text, use_cache=True)
        r2 = tts.say(text, use_cache=True)
        assert not r1.cached
        assert r2.cached
        assert r1.duration == pytest.approx(r2.duration, abs=1e-6)
    finally:
        tts.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
