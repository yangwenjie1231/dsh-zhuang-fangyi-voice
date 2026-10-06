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


def has_onnx(d):
    """ONNX 推理所需的 7 个模型是否齐全"""
    if not d or not os.path.isdir(d):
        return False
    return all(os.path.exists(os.path.join(d, f)) for f in ONNX_FILES)


def has_torch_weights(d, exp_name="zfh"):
    """torch 后端所需的两个权重是否齐全（在 <d>/torch_weights/ 下）"""
    if not d or not os.path.isdir(d):
        return False
    td = os.path.join(d, "torch_weights")
    return (os.path.exists(os.path.join(td, f"{exp_name}-e4.ckpt"))
            and os.path.exists(os.path.join(td, f"{exp_name}_e6_s186.pth")))


def has_aux(d):
    """文本前端辅助文件是否齐全"""
    if not d or not os.path.isdir(d):
        return False
    return (os.path.exists(os.path.join(d, "G2PWModel", "g2pW.onnx"))
            and os.path.exists(os.path.join(d, "chinese-roberta-wwm-ext-large",
                                            "tokenizer.json")))


def has_ref(d):
    return bool(d) and os.path.exists(os.path.join(d, "ref", "default.wav"))


def _matches(d, require):
    if not d or not os.path.isdir(d):
        return False
    if require == "onnx":
        return has_onnx(d)
    if require == "torch":
        return has_torch_weights(d)
    if require == "aux":
        return has_aux(d)
    # require=None：只要目录里有「任一」可识别的模型内容就认
    return has_onnx(d) or has_torch_weights(d) or has_aux(d) or has_ref(d)


def resolve_model_dir(model_dir=None, must_exist=True, require=None):
    """按优先级找到可用的模型目录

    require:
        "onnx"  → 必须齐全 7 个 ONNX（ONNX 后端用）
        "torch" → 必须有 torch_weights/ 下的 .ckpt/.pth
        "aux"   → 必须有文本前端辅助文件
        None    → 只要含任一类内容即可（默认；兼容只用 torch 后端的场景）
    """
    cands = []
    if model_dir:
        cands.append(model_dir)
    if os.environ.get("ZFH_MODEL_DIR"):
        cands.append(os.environ["ZFH_MODEL_DIR"])
    cands.append(os.path.join(repo_root(), "models"))
    cands.append(default_cache_dir())

    for d in cands:
        if _matches(d, require):
            return os.path.abspath(d)
    if must_exist:
        hint = {
            "onnx": "请运行 `python download_models.py` 下载 ONNX 模型",
            "torch": "请运行 `python download_models.py --with-torch` 下载 torch 权重",
            "aux": "请运行 `python download_models.py`（辅助文件随包下载）",
        }.get(require, "请运行 `python download_models.py` 下载模型")
        raise FileNotFoundError(
            f"找不到可用的模型目录（require={require or '任意'!r}）。已尝试：\n  "
            + "\n  ".join(os.path.abspath(c) for c in cands)
            + f"\n\n{hint}，或设置环境变量 ZFH_MODEL_DIR 指向模型目录。")
    return os.path.abspath(cands[0]) if cands else None


def model_status(model_dir=None, backend="onnx"):
    """返回 (目录, 缺失文件列表, 已就绪 bool)

    backend="onnx"  检查 7 个 ONNX + 辅助文件
    backend="torch" 检查 torch 权重 + 辅助文件
    """
    d = model_dir or os.environ.get("ZFH_MODEL_DIR") or \
        os.path.join(repo_root(), "models")
    d = os.path.abspath(d)
    required = list(AUX_FILES if backend == "onnx" else [])
    required += (ONNX_FILES if backend == "onnx"
                 else [os.path.join("torch_weights", "zfh-e4.ckpt"),
                       os.path.join("torch_weights", "zfh_e6_s186.pth")])
    missing = [f for f in required if not os.path.exists(os.path.join(d, f))]
    return d, missing, len(missing) == 0


def g2pw_dir(model_dir):
    return os.path.join(model_dir, "G2PWModel")


def bert_dir(model_dir):
    return os.path.join(model_dir, "chinese-roberta-wwm-ext-large")


def vendor_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
