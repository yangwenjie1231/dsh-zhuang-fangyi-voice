# -*- coding: utf-8 -*-
"""中文文本前端：文本 → 音素 ID（内嵌 GPT-SoVITS 前端，MIT）

内嵌包位于 _vendor/_gsv_text，已改私有名避免与宿主模块冲突，
并把 G2PW 的硬编码路径改为环境变量驱动（缺失时报错而非联网下载）。

使用前需调用 prepare()，它会：
  1. 解析模型目录
  2. 设置 ZFH_G2PW_DIR / ZFH_BERT_DIR
  3. 把 _vendor 加入 sys.path
"""
import os
import sys
import threading

from . import paths

_lock = threading.Lock()
_ready = False
_clean_text = None
_to_seq = None


def prepare(model_dir=None):
    """初始化文本前端（幂等）"""
    global _ready, _clean_text, _to_seq
    with _lock:
        if _ready:
            return
        mdir = paths.resolve_model_dir(model_dir)
        g2pw = paths.g2pw_dir(mdir)
        bert = paths.bert_dir(mdir)
        if not os.path.isdir(g2pw):
            raise FileNotFoundError(f"G2PW 模型目录不存在: {g2pw}")
        if not os.path.isdir(bert):
            raise FileNotFoundError(f"BERT tokenizer 目录不存在: {bert}")
        # 必须在 import 内嵌前端之前设置（chinese2.py 在 import 时即加载 G2PW）
        os.environ["ZFH_G2PW_DIR"] = g2pw
        os.environ["ZFH_BERT_DIR"] = bert

        vd = paths.vendor_dir()
        if vd not in sys.path:
            sys.path.insert(0, vd)

        from _gsv_text import cleaned_text_to_sequence  # noqa: E402
        from _gsv_text.cleaner import clean_text  # noqa: E402

        _clean_text = clean_text
        _to_seq = cleaned_text_to_sequence
        _ready = True


def text_to_phonemes(text, language="zh", version="v2"):
    """文本 → (phones, word2ph, norm_text)"""
    prepare()
    phones, word2ph, norm_text = _clean_text(text, language, version)
    return phones, word2ph, norm_text


def text_to_ids(text, language="zh", version="v2"):
    """文本 → (ids[list[int]], word2ph, norm_text)"""
    phones, word2ph, norm_text = text_to_phonemes(text, language, version)
    return _to_seq(phones, version), word2ph, norm_text
