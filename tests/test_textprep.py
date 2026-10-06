# -*- coding: utf-8 -*-
"""英文可念化（textprep）的回归测试

背景：中文文本前端会**静默删掉**英文 ——
`chinese2.py` 第 206 行 `re.sub("[a-zA-Z]+", "", seg)`。
实测「今天天气不错，Hello World。」与「今天天气不错。」音素数完全相同，
纯英文更是直接 0 音素、合成报错。

对一个要念 AI 回答的桌宠来说这是硬伤：回答里的 `GPU`、`Python`
会被静默吃掉，用户只看到"念得怪怪的"，不知道是模型坏了还是输入有问题。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import textprep  # noqa: E402


# ── 基本替换 ────────────────────────────────────────────────────────────

def test_common_words():
    assert textprep.localize("Hello World") == "哈喽 沃尔德" or \
        textprep.localize("Hello World").startswith("哈喽")
    assert "派森" in textprep.localize("这个是 Python 写的程序。")
    assert "欧克" in textprep.localize("OK，我马上就来。")


def test_uppercase_abbrev_spelled_letter_by_letter():
    assert textprep.localize("GPU") == "基皮尤"
    assert textprep.localize("AI") == "诶艾"
    assert "基皮尤" in textprep.localize("GPU 和 AI 都很重要。")


def test_known_abbrev_from_table():
    # 表里有专门读法的优先用表
    assert textprep.localize("API") == "诶皮艾"
    assert textprep.localize("JSON") == "杰森"


def test_chinese_untouched():
    for s in ["今天天气不错。", "管理员，实验数据整理好了！", ""]:
        assert textprep.localize(s) == s


def test_digits_and_symbols_untouched():
    """数字与符号**不能动** —— 前端本来就能正确处理它们

    （`3`→`三`、`￥100`→`幺零零`、`50%`→`百分之五十`）
    """
    for s in ["我有 3 个苹果。", "价格是 ￥100，增长了 50%。", "1+2=3"]:
        assert textprep.localize(s) == s


def test_no_english_left_behind():
    """转换后不该再剩英文字母（否则又会被前端吃掉）"""
    import re
    cases = [
        "GPU 和 AI 都很重要。",
        "Hello World",
        "这个是 Python 写的程序。",
        "API 返回了 JSON 格式的数据。",
        "Debug 一下这个 bug",
        "no problem at all",
    ]
    for c in cases:
        out = textprep.localize(c)
        left = re.findall(r"[A-Za-z]+", out)
        assert not left, f"{c!r} → {out!r} 仍残留英文 {left}"


def test_has_english():
    assert textprep.has_english("GPU") is True
    assert textprep.has_english("Hello 你好") is True
    assert textprep.has_english("纯中文") is False
    assert textprep.has_english("123 ￥") is False
    assert textprep.has_english("") is False
    assert textprep.has_english(None) is False


def test_non_string_passthrough():
    assert textprep.localize(None) is None
    assert textprep.localize(123) == 123


def test_idempotent():
    """转换结果里已无英文 → 再转一次应不变"""
    for c in ["GPU 和 AI", "Hello World", "Python 程序"]:
        once = textprep.localize(c)
        assert textprep.localize(once) == once


def test_word_inflections():
    """复数/词形变化能命中词表"""
    assert "派森" in textprep.localize("Pythons")  # 去 s 命中
    assert textprep.localize("Thanks") == "三克油"


def test_hyphen_and_apostrophe_words():
    """带连字符/撇号的词整词处理，不该被拆碎"""
    out = textprep.localize("don't worry")
    assert "'" not in out, f"撇号应被消掉：{out}"
    assert not any(c.isalpha() and c.isascii() for c in out), out


# ── 与 API 的集成 ───────────────────────────────────────────────────────

def test_api_rejects_text_with_no_spoken_content(tmp_path, monkeypatch):
    """纯英文关掉转换时应明确报错，而不是产出静音"""
    from zfh_voice import api

    monkeypatch.setattr(api.paths, "resolve_model_dir",
                        lambda *a, **k: str(tmp_path))
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "default.wav").write_bytes(b"\0")
    (ref / "default.txt").write_text("参考。", encoding="utf-8")

    t = api.TTS(localize=False)
    with pytest.raises(ValueError) as ei:
        t.say("Hello World")
    assert "可发音" in str(ei.value) or "文本" in str(ei.value)


def test_say_uses_localized_text(tmp_path, monkeypatch):
    """say() 应把转换后的文本送给后端，并记录在 result.spoken"""
    from zfh_voice import api

    monkeypatch.setattr(api.paths, "resolve_model_dir",
                        lambda *a, **k: str(tmp_path))
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "default.wav").write_bytes(b"\0")
    (ref / "default.txt").write_text("参考。", encoding="utf-8")

    seen = {}

    class FakeBackend:
        def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
            seen["text"] = text
            return [0.0] * 3200, 32000

        def close(self):
            pass

    monkeypatch.setattr(api, "make_backend",
                        lambda *a, **k: FakeBackend())

    t = api.TTS(use_cache=False)
    r = t.say("GPU 和 AI 都很重要。")
    assert seen["text"] == "基皮尤 和 诶艾 都很重要。", seen["text"]
    assert r.text == "GPU 和 AI 都很重要。"       # 原始输入保留
    assert r.spoken == "基皮尤 和 诶艾 都很重要。"  # 实际合成文本
    assert not any(c.isascii() and c.isalpha() for c in seen["text"])


def test_localize_can_be_disabled_per_call(tmp_path, monkeypatch):
    from zfh_voice import api

    monkeypatch.setattr(api.paths, "resolve_model_dir",
                        lambda *a, **k: str(tmp_path))
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "default.wav").write_bytes(b"\0")
    (ref / "default.txt").write_text("参考。", encoding="utf-8")

    seen = {}

    class FakeBackend:
        def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
            seen["text"] = text
            return [0.0] * 3200, 32000

        def close(self):
            pass

    monkeypatch.setattr(api, "make_backend", lambda *a, **k: FakeBackend())
    t = api.TTS(use_cache=False)
    t.say("GPU 和 AI", localize=False)
    assert seen["text"] == "GPU 和 AI"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
