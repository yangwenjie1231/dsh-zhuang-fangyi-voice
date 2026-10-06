# -*- coding: utf-8 -*-
"""模型目录解析与完整性检查

解析优先级：
  1. 环境变量 ZFH_MODEL_DIR
  2. 仓库根目录下的 models/
  3. 用户缓存目录 ~/.cache/zfh-voice/models

模型清单固定为 7 个 ONNX + 若干辅助文件，见 ONNX_FILES / AUX_FILES。
"""
import os
import sys

# 推理必需（fp16 为默认精度）
ONNX_FILES = [
    "zfh_t2s_encoder.onnx",
    "zfh_t2s_fsdec.onnx",
    "zfh_t2s_sdec.onnx",
    "zfh_vits.onnx",
    "bert_encoder.onnx",
    "hubert_encoder.onnx",
    "g2pW.onnx",
]

# 文本前端辅助文件
AUX_FILES = [
    os.path.join("G2PWModel", "g2pW.onnx"),      # 与 g2pW.onnx 同源，前端按此路径加载
    os.path.join("G2PWModel", "bopomofo_to_pinyin_wo_tune_dict.json"),
    os.path.join("G2PWModel", "char_bopomofo_dict.json"),
    os.path.join("G2PWModel", "MONOPHONIC_CHARS.txt"),
    os.path.join("G2PWModel", "POLYPHONIC_CHARS.txt"),
    os.path.join("G2PWModel", "config.py"),
    os.path.join("G2PWModel", "version"),
    # fast tokenizer 格式：tokenizer.json + config.json 即可
    os.path.join("chinese-roberta-wwm-ext-large", "tokenizer.json"),
    os.path.join("chinese-roberta-wwm-ext-large", "config.json"),
    os.path.join("ref", "default.wav"),
]


def repo_root():
    """仓库根目录（本文件是 <root>/src/zfh_voice/paths.py）"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "zfh-voice", "models")


def _looks_like_model_dir(d):
    if not d or not os.path.isdir(d):
        return False
    return all(os.path.exists(os.path.join(d, f)) for f in ONNX_FILES)


def resolve_model_dir(model_dir=None, must_exist=True):
    """按优先级找到可用的模型目录"""
    cands = []
    if model_dir:
        cands.append(model_dir)
    if os.environ.get("ZFH_MODEL_DIR"):
        cands.append(os.environ["ZFH_MODEL_DIR"])
    cands.append(os.path.join(repo_root(), "models"))
    cands.append(default_cache_dir())

    for d in cands:
        if _looks_like_model_dir(d):
            return os.path.abspath(d)
    if must_exist:
        raise FileNotFoundError(
            "找不到完整的模型目录。已尝试：\n  " +
            "\n  ".join(os.path.abspath(c) for c in cands) +
            "\n\n请先运行 `python download_models.py` 下载模型，"
            "或设置环境变量 ZFH_MODEL_DIR 指向模型目录。")
    return os.path.abspath(cands[0]) if cands else None


def model_status(model_dir=None):
    """返回 (目录, 缺失文件列表, 已就绪 bool)"""
    d = model_dir or os.environ.get("ZFH_MODEL_DIR") or \
        os.path.join(repo_root(), "models")
    d = os.path.abspath(d)
    missing = []
    for f in ONNX_FILES + AUX_FILES:
        if not os.path.exists(os.path.join(d, f)):
            missing.append(f)
    return d, missing, len(missing) == 0


def g2pw_dir(model_dir):
    return os.path.join(model_dir, "G2PWModel")


def bert_dir(model_dir):
    return os.path.join(model_dir, "chinese-roberta-wwm-ext-large")


def vendor_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
