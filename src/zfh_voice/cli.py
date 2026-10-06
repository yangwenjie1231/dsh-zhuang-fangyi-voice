# -*- coding: utf-8 -*-
"""命令行入口

  python -m zfh_voice say "今天天气不错"              # 合成一句
  python -m zfh_voice say "..." -o out.wav --seed 7
  python -m zfh_voice batch lines.txt -d out/        # 批量（每行一条）
  python -m zfh_voice serve --port 8765              # 起 HTTP 服务
  python -m zfh_voice status                         # 检查模型是否就绪
  python -m zfh_voice cache --clear                  # 查看/清理缓存
"""
import argparse
import os
import sys


def _build_tts(args):
    from .api import TTS
    kw = {}
    if args.backend == "torch":
        if not args.gsv_root:
            print("错误：torch 后端需要 --gsv-root 指定 GPT-SoVITS 检出目录",
                  file=sys.stderr)
            sys.exit(2)
        kw["gsv_root"] = args.gsv_root
        if args.device:
            kw["device"] = args.device
    return TTS(backend=args.backend, model_dir=args.model_dir,
               ref_wav=args.ref, ref_text=args.ref_text,
               use_cache=not args.no_cache, **kw)


def cmd_say(args):
    tts = _build_tts(args)
    text = " ".join(args.text).strip()
    if not text:
        print("错误：文本为空", file=sys.stderr)
        return 2
    out = args.out or os.path.join(args.outdir, "out.wav")
    r = tts.say(text, seed=args.seed, out=out, verbose=True)
    print(f"{r.duration:.2f}s  ->  {os.path.abspath(out)}")
    return 0


def cmd_batch(args):
    tts = _build_tts(args)
    if not os.path.exists(args.file):
        print(f"错误：文件不存在 {args.file}", file=sys.stderr)
        return 2
    texts = [l.strip() for l in open(args.file, encoding="utf-8") if l.strip()]
    print(f"共 {len(texts)} 条 → {args.outdir}")
    rs = tts.say_many(texts, args.outdir, seed=args.seed)
    total = sum(r.duration for r in rs)
    hit = sum(1 for r in rs if r.cached)
    print(f"\n完成 {len(rs)} 条，音频合计 {total:.1f}s，缓存命中 {hit}")
    return 0


def cmd_serve(args):
    from .server import serve
    tts = _build_tts(args)
    serve(tts, host=args.host, port=args.port)
    return 0


def cmd_status(args):
    from .paths import has_onnx, has_torch_weights, model_status, repo_root
    d, missing, ok = model_status(args.model_dir, backend=args.backend)
    print(f"仓库根目录 : {repo_root()}")
    print(f"模型目录   : {d}")
    print(f"后端       : {args.backend}")
    print(f"状态       : {'✓ 就绪' if ok else f'✗ 缺 {len(missing)} 个文件'}")
    # 顺带告知另一类模型是否也在，便于判断能否切换后端
    print(f"  另有 ONNX 模型      : {'是' if has_onnx(d) else '否'}")
    print(f"  另有 torch 权重     : {'是' if has_torch_weights(d) else '否'}")
    if missing:
        for m in missing[:12]:
            print(f"  缺: {m}")
        if len(missing) > 12:
            print(f"  ... 共 {len(missing)} 项")
        if args.backend == "onnx":
            print("\n运行 `python download_models.py` 下载 ONNX 模型")
        else:
            print("\n运行 `python download_models.py --with-torch` 下载 torch 权重")
            print("（torch 后端不要求 ONNX 模型；两者互不通用）")
    return 0 if ok else 1


def cmd_cache(args):
    from .api import TTS
    tts = _build_tts(args)
    if args.clear:
        n = tts.clear_cache()
        print(f"已清理 {n} 个缓存文件")
    else:
        files = [f for f in os.listdir(tts.cache_dir) if f.endswith(".wav")]
        size = sum(os.path.getsize(os.path.join(tts.cache_dir, f))
                   for f in files)
        print(f"缓存目录: {tts.cache_dir}")
        print(f"条目 {len(files)} 个，合计 {size/1024/1024:.1f} MB")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="zfh-voice", description="庄方宜专属音色合成")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"],
                    help="推理后端（默认 onnx）")
    ap.add_argument("--model-dir", default=None, help="模型目录")
    ap.add_argument("--ref", default=None, help="参考音频路径")
    ap.add_argument("--ref-text", default=None, help="参考音频对应文本")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true", help="禁用合成缓存")
    ap.add_argument("--gsv-root", default=None,
                    help="GPT-SoVITS 检出目录（torch 后端需要）")
    ap.add_argument("--device", default=None, help="torch 后端设备: cuda/cpu")
    ap.add_argument("--outdir", default="out", help="默认输出目录")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("say", help="合成一句")
    p.add_argument("text", nargs="+")
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("batch", help="批量合成（每行一条）")
    p.add_argument("file")
    p.add_argument("-d", "--outdir", default="out")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("serve", help="启动 HTTP 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="检查模型是否就绪")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("cache", help="查看/清理缓存")
    p.add_argument("--clear", action="store_true")
    p.set_defaults(func=cmd_cache)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
