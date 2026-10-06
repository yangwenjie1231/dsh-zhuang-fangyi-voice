# -*- coding: utf-8 -*-
"""下载模型文件（从 GitHub Release）

用法：
    python download_models.py                    # 默认 fp16
    python download_models.py --precision int8
    python download_models.py --precision fp32
    python download_models.py --dir D:\\models    # 指定目录
    python download_models.py --check            # 只检查不下载

产出目录结构：
    models/
    ├─ *.onnx                       7 个推理模型
    ├─ G2PWModel/                   多音字模型与数据
    ├─ chinese-roberta-wwm-ext-large/  BERT tokenizer（只需几个小文件）
    └─ ref/default.wav(.txt)        默认参考音频
"""
import argparse
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "yangwenjie1231/dsh-zhuang-fangyi-voice"
TAG = "models-v1"

ONNX_FILES = [
    "zfh_t2s_encoder.onnx", "zfh_t2s_fsdec.onnx", "zfh_t2s_sdec.onnx",
    "zfh_vits.onnx", "bert_encoder.onnx", "hubert_encoder.onnx", "g2pW.onnx",
]
PRECISIONS = ("fp16", "int8", "fp32")


def target_dir(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    if os.environ.get("ZFH_MODEL_DIR"):
        return os.path.abspath(os.environ["ZFH_MODEL_DIR"])
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def url_for(name):
    return f"https://github.com/{REPO}/releases/download/{TAG}/{name}"


def fetch(url, dst, expect_mb=None):
    """带进度的下载（优先 curl，退回 urllib）"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  已存在，跳过: {os.path.basename(dst)}")
        return True
    print(f"  下载 {os.path.basename(dst)}"
          + (f" (~{expect_mb}MB)" if expect_mb else ""))
    import subprocess
    if sys.platform == "win32":
        cmd = ["curl.exe", "-L", "--fail", "--retry", "3", "-o", dst, url]
    else:
        cmd = ["curl", "-L", "--fail", "--retry", "3", "-o", dst, url]
    try:
        r = subprocess.run(cmd)
        if r.returncode == 0 and os.path.exists(dst):
            return True
    except FileNotFoundError:
        pass
    # 退回 urllib
    try:
        import urllib.request
        with urllib.request.urlopen(url) as resp, open(dst, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        if os.path.exists(dst):
            os.remove(dst)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16", choices=PRECISIONS)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--check", action="store_true", help="只检查不下载")
    a = ap.parse_args()

    d = target_dir(a.dir)
    print(f"目标目录: {d}")
    print(f"精度    : {a.precision}")

    need = [f for f in ONNX_FILES
            if not os.path.exists(os.path.join(d, f))]
    aux_ok = os.path.exists(os.path.join(d, "G2PWModel", "g2pW.onnx")) and \
        os.path.exists(os.path.join(d, "chinese-roberta-wwm-ext-large", "tokenizer.json"))
    ref_ok = os.path.exists(os.path.join(d, "ref", "default.wav"))

    if a.check:
        print(f"\nONNX  : {len(ONNX_FILES) - len(need)}/{len(ONNX_FILES)} 就绪")
        print(f"辅助  : {'就绪' if aux_ok else '缺失'}")
        print(f"参考音: {'就绪' if ref_ok else '缺失'}")
        if need:
            print("缺失: " + ", ".join(need))
        return 0 if (not need and aux_ok and ref_ok) else 1

    os.makedirs(d, exist_ok=True)
    ok = True

    # 1) 辅助文件包（小，先下）
    if not aux_ok or not ref_ok:
        z = os.path.join(d, "_aux.zip")
        if fetch(url_for("zfh-voice-aux.zip"), z, 20):
            print("  解压辅助文件 ...")
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
            os.remove(z)
        else:
            ok = False
    else:
        print("辅助文件已就绪")

    # 2) ONNX 模型（逐个下，便于断点续传）
    if need:
        print(f"\n需下载 {len(need)} 个模型（{a.precision}）")
        for f in need:
            src = f"{a.precision}__{f}"
            tmp = os.path.join(d, f + ".part")
            if not fetch(url_for(src), tmp):
                ok = False
                continue
            os.replace(tmp, os.path.join(d, f))
    else:
        print("\nONNX 模型已就绪")

    # 2b) g2pW.onnx 需要同时存在于 G2PWModel/（文本前端按该路径加载）
    g_src = os.path.join(d, "g2pW.onnx")
    g_dst_dir = os.path.join(d, "G2PWModel")
    g_dst = os.path.join(g_dst_dir, "g2pW.onnx")
    if os.path.exists(g_src) and not os.path.exists(g_dst):
        os.makedirs(g_dst_dir, exist_ok=True)
        try:
            os.link(g_src, g_dst)              # 同盘用硬链接，省一份空间
            print("  g2pW.onnx → G2PWModel/ (硬链接)")
        except OSError:
            import shutil
            shutil.copy2(g_src, g_dst)
            print("  g2pW.onnx → G2PWModel/ (复制)")

    # 3) 校验
    print("\n=== 校验 ===")
    from_check = []
    for f in ONNX_FILES:
        p = os.path.join(d, f)
        if os.path.exists(p):
            print(f"  OK   {f:<28}{os.path.getsize(p)/1024/1024:>8.1f} MB")
        else:
            print(f"  MISS {f}")
            from_check.append(f)
    if from_check:
        ok = False
        print(f"\n仍缺 {len(from_check)} 个文件")
    else:
        print("\n全部就绪。可以用了：")
        print('  python -m zfh_voice say "今天天气不错"')
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
