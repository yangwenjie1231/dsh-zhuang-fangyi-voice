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
    # require=None：只要目录里有「能支撑推理的内容」就认
    # 注意：仅有 ref/ 音频不算——那无法合成，必须有模型或文本前端辅助文件
    return has_onnx(d) or has_torch_weights(d) or has_aux(d)


def _missing_in(d, require):
    """列出该目录下按 require 还缺什么"""
    if not d:
        return []
    if require == "onnx":
        return [f for f in ONNX_FILES
                if not os.path.exists(os.path.join(d, f))]
    if require == "torch":
        td = os.path.join(d, "torch_weights")
        return [f"{f}（torch 后端用）" for f in ("zfh-e4.ckpt", "zfh_e6_s186.pth")
                if not os.path.exists(os.path.join(td, f))]
    if require == "aux":
        return [f for f in AUX_FILES if not os.path.exists(os.path.join(d, f))]
    return ["该目录里没有任何可识别的模型内容"]


def resolve_model_dir(model_dir=None, must_exist=True, require=None):
    """按优先级找到可用的模型目录

    require:
        "onnx"  → 必须齐全 7 个 ONNX（ONNX 后端用）
        "torch" → 必须有 torch_weights/ 下的 .ckpt/.pth
        "aux"   → 必须有文本前端辅助文件
        None    → 只要含任一类内容即可（默认；兼容只用 torch 后端的场景）

    找不到时会抛出一段**可直接照做**的指引，而不是只说一句"找不到"。
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

    if not must_exist:
        return os.path.abspath(cands[0]) if cands else None

    # 构造可执行的指引
    lines = []
    lines.append(f"未找到可用的模型目录（需要：{require or '任意一类模型'}）。")
    lines.append("")
    lines.append("已检查这些位置：")
    for c in cands:
        ap = os.path.abspath(c)
        tag = "存在" if os.path.isdir(ap) else "不存在"
        miss = _missing_in(ap, require)
        extra = f"，缺 {len(miss)} 项" if (os.path.isdir(ap) and miss) else ""
        lines.append(f"  [{tag}{extra}] {ap}")
    lines.append("")
    lines.append("解决办法（任选其一）：")
    if require == "onnx":
        lines.append("  1) 装 ONNX 模型（推荐 fp16，约 1.4GB）")
        lines.append("       python download_models.py")
        lines.append("     磁盘紧张可用 int8（0.87GB）：")
        lines.append("       python download_models.py --precision int8")
        lines.append("  2) 改用 torch 后端（不需要 ONNX，需要 .ckpt/.pth）")
        lines.append("       python download_models.py --torch-only")
        lines.append("       python -m zfh_voice --backend torch "
                     "--gsv-root <GPT-SoVITS 目录> say \"你好\"")
    elif require == "torch":
        lines.append("  下载 torch 后端所需的原始权重（约 229MB）：")
        lines.append("       python download_models.py --with-torch")
        lines.append("     （若只用 ONNX 后端，则不需要这些权重）")
    else:
        lines.append("       python download_models.py")
        lines.append("       python download_models.py --torch-only")
    lines.append(f"  3) 指定已有目录：设环境变量 ZFH_MODEL_DIR，或用 --model-dir")
    lines.append("  4) 不确定该装哪套？先跑环境体检：")
    lines.append("       python -m zfh_voice doctor")
    raise FileNotFoundError("\n".join(lines))


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
