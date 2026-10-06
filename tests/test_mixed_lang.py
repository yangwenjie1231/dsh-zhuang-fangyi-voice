# -*- coding: utf-8 -*-
"""中英混排（LangSegmenter 分段）的回归测试

背景：中文文本前端对英文是 `re.sub("[a-zA-Z]+", "", seg)` —— **静默删除**。
上游靠 `LangSegmenter` 按语言分段解决：英文段走 `en` 音素、中文段走 `zh`。
本项目此前直接调 `clean_text(text, "zh")`，绕过了分段，所以英文全丢。

本测试钉死分段行为与关键契约：
  · 英文段必须产出音素（不是被删）
  · 中文段的 word2ph 必须存在且 sum(word2ph) == 音素数
  · 非中文段的 word2ph 必须是 None（上游契约，调用方据此给零 BERT）
  · 分段不可用时优雅降级，不抛
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import frontend, paths  # noqa: E402


def _ready():
    """前端是否可用（需要模型目录里的 G2PW/BERT 辅助文件）"""
    try:
        frontend.prepare()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ready(), reason="文本前端不可用（缺模型目录或辅助文件）")


def test_pure_chinese_single_segment():
    segs = frontend.text_to_segments("今天天气不错。", "zh")
    assert len(segs) == 1
    lang, phones, w2p, norm = segs[0]
    assert lang == "zh"
    assert len(phones) == 13
    assert w2p is not None and sum(w2p) == len(phones)


def test_pure_english_gets_phonemes():
    """★ 核心回归：纯英文必须产出音素（旧路径下是 0 音素）"""
    segs = frontend.text_to_segments("Hello world, this is a test.", "zh")
    assert len(segs) == 1
    lang, phones, w2p, _ = segs[0]
    assert lang == "en"
    assert len(phones) > 0, "英文段必须有音素，否则合成会失败"
    assert w2p is None, "非中文段的 word2ph 应为 None（上游契约）"


def test_mixed_splits_by_language():
    segs = frontend.text_to_segments("GPU 和 AI 都很重要。", "zh")
    langs = [s[0] for s in segs]
    assert "en" in langs and "zh" in langs, f"应中英分段，实际 {langs}"
    # 英文段有音素，中文段有 word2ph
    for lang, phones, w2p, _ in segs:
        assert len(phones) > 0
        if lang == "zh":
            assert w2p is not None and sum(w2p) == len(phones)
        else:
            assert w2p is None


def test_mixed_english_not_deleted():
    """★ 加了英文后音素数必须**变多**（旧路径下不变 → 英文被删）"""
    a = sum(len(p) for _, p, _, _ in frontend.text_to_segments("今天天气不错。", "zh"))
    b = sum(len(p) for _, p, _, _
            in frontend.text_to_segments("今天天气不错，Hello World。", "zh"))
    assert b > a, f"英文应产生额外音素：{a} → {b}"


def test_english_phonemes_are_arpabet():
    """英文段应是 ARPAbet 音素（HH/AH0/OW1…），不是中文拼音"""
    segs = frontend.text_to_segments("Hello", "zh")
    phones = segs[0][1]
    assert any(p in ("HH", "AH0", "L", "OW1") for p in phones), phones


def test_mixed_available_flag_is_bool():
    assert isinstance(frontend.mixed_available(), bool)


def test_split_falls_back_when_unavailable(monkeypatch):
    """分段不可用时降级为单段，不抛"""
    monkeypatch.setattr(frontend, "_lang_segment", None)
    monkeypatch.setattr(frontend, "_ready", True)
    segs = frontend.split_by_lang("GPU 和 AI", "zh")
    assert segs == [("zh", "GPU 和 AI")]


def test_segments_reject_empty():
    """空文本返回 []（不产生空段）"""
    assert frontend.text_to_segments("", "zh") == []
    assert frontend.text_to_segments("   ", "zh") == []
    assert frontend.split_by_lang("", "zh") == []
    # 注意：纯标点**不是**空 —— 上游会把 "。。。" 归一化成一个句号音素
    # （相当于一个停顿），这是合理行为，不该被当成空文本拦掉。
    segs = frontend.text_to_segments("。。。", "zh")
    assert len(segs) == 1 and segs[0][1] == ["."]


def test_unknown_language_mapped_to_default():
    """分段器给出未支持语言时，应回落到默认语言而不是崩"""
    def fake(text, default):
        return [{"lang": "fr", "text": "bonjour"}, {"lang": "zh", "text": "你好"}]
    import types
    orig = frontend._lang_segment
    frontend._lang_segment = fake
    try:
        segs = frontend.split_by_lang("x", "zh")
        assert segs[0][0] == "zh", f"未知语言应回落 zh，实际 {segs[0][0]}"
        assert segs[1][0] == "zh"
    finally:
        frontend._lang_segment = orig


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
