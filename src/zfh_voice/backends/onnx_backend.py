# -*- coding: utf-8 -*-
"""ONNX 后端：仅依赖 onnxruntime，不需要 torch / CUDA

推理链路（全部为 ONNX）：
  BERT(ONNX)   文本 → 语义特征
  HuBERT(ONNX) 参考音频 → SSL 内容特征
  encoder      两者 + 音素 → x, prompts
  fsdec        首段解码
  sdec         自回归逐 token 解码（循环 MAX_LOOP 次）
  vits         语义 token + 参考音频 → 波形

精度可切换：fp32 / fp16 / int8（同目录下的三套模型文件命名一致）
"""
import os
import time

import numpy as np

from .. import audio, frontend, paths
from .base import (EARLY_STOP, EOS, MAX_LOOP, SAMPLE_RATE, VERSION,
                   BackendError, SynthBackend)


class OnnxBackend(SynthBackend):
    name = "onnx"

    def __init__(self, model_dir=None, providers=None, threads=None,
                 log_level=3):
        super().__init__(model_dir)
        import onnxruntime as ort
        self.ort = ort
        self.model_dir = paths.resolve_model_dir(model_dir)

        if providers is None:
            # 默认：优先 DirectML（Windows 任意显卡），其次 CPU
            avail = ort.get_available_providers()
            if "DmlExecutionProvider" in avail:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

        so = ort.SessionOptions()
        so.log_severity_level = log_level
        if threads:
            so.intra_op_num_threads = int(threads)

        t0 = time.time()
        self.sess = {}
        for key, fn in [
            ("encoder", "zfh_t2s_encoder.onnx"),
            ("fsdec", "zfh_t2s_fsdec.onnx"),
            ("sdec", "zfh_t2s_sdec.onnx"),
            ("vits", "zfh_vits.onnx"),
            ("bert", "bert_encoder.onnx"),
            ("hubert", "hubert_encoder.onnx"),
        ]:
            p = os.path.join(self.model_dir, fn)
            if not os.path.exists(p):
                raise BackendError(f"缺少模型文件: {p}")
            self.sess[key] = ort.InferenceSession(p, so, providers=providers)
        self.load_seconds = time.time() - t0

        # BERT tokenizer（只需 tokenizer 文件，不需要权重）
        from transformers import AutoTokenizer
        tok_dir = paths.bert_dir(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)

        # 文本前端（会设置 ZFH_G2PW_DIR / ZFH_BERT_DIR）
        frontend.prepare(self.model_dir)

    # ---------------- 子步骤 ----------------
    def _text_ids(self, text):
        """文本 → (ids, word2ph, norm_text, bert)

        **bert 形状必须是 [text_length, 1024]** —— 这是导出的 ONNX 契约
        （`zfh_t2s_encoder.onnx` 的 `text_bert` 输入是 ['text_length', 1024]）。
        上游 torch 侧内部是 [1024, T]，导出时已转置，这里不要照搬上游布局。

        中英混排按语言分段：中文段走 G2PW 并用真 BERT 特征；
        英文段走 g2p_en（ARPAbet 音素）且 BERT 给**全零** ——
        与上游 `TextPreprocessor.get_bert_inf` 一致（BERT 只对中文有意义，
        英文段的 word2ph 本来就是 None）。
        """
        segs = frontend.text_to_segments(text, "zh", VERSION)
        if not segs:
            return (np.zeros((1, 0), dtype=np.int64), None, "",
                    np.zeros((0, 1024), dtype=np.float32))

        ids_all, bert_parts, norm_all, w2p_all = [], [], [], []
        single_lang = len(segs) == 1
        for lang, phones, word2ph, norm in segs:
            ids_all.extend(frontend.cleaned_ids(phones, VERSION))
            norm_all.append(norm)
            if lang == "zh" and word2ph is not None:
                bert_parts.append(self._bert_feature(norm, word2ph))   # [T,1024]
                w2p_all.extend(word2ph)
            else:
                # 非中文：BERT 全零，行数与音素数一致
                bert_parts.append(np.zeros((len(phones), 1024), dtype=np.float32))
                single_lang = False

        bert = np.concatenate(bert_parts, axis=0) if bert_parts else \
            np.zeros((0, 1024), dtype=np.float32)
        w2p = w2p_all if (single_lang and w2p_all) else None
        return (np.array([ids_all], dtype=np.int64), w2p, "".join(norm_all), bert)

    def _bert_feature(self, norm_text, word2ph):
        """中文 BERT 特征 → [T, 1024]（T = sum(word2ph) = 音素数）"""
        enc = self.tokenizer(norm_text, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get(
                "token_type_ids", np.zeros_like(ids)).astype(np.int64),
        }
        hidden = self.sess["bert"].run(None, feed)[0]      # [1, seq, 1024]
        res = hidden[0][1:-1]                              # 去掉首尾特殊 token
        reps = [np.tile(res[i], (word2ph[i], 1)) for i in range(len(word2ph))]
        return np.concatenate(reps, axis=0).astype(np.float32)     # [T, 1024]

    def _ssl_content(self, ref_wav):
        wav, sr = audio.load_wav(ref_wav)
        wav16k = audio.resample(wav, sr, 16000)
        self.check_ref_duration(len(wav) / sr)
        padded = self.zero_pad_for_hubert(wav16k)
        out = self.sess["hubert"].run(
            None, {"wav": padded[None, :].astype(np.float32)})[0]
        return out.astype(np.float32)                      # [1, 768, T]

    def _ref_audio_32k(self, ref_wav):
        wav, sr = audio.load_wav(ref_wav)
        return audio.resample(wav, sr, SAMPLE_RATE)[None, :].astype(np.float32)

    # ---------------- 主流程 ----------------
    def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
        text_seq, text_w2p, text_norm, text_bert = self._text_ids(text)
        ref_seq, ref_w2p, ref_norm, ref_bert = self._text_ids(ref_text)
        ssl = self._ssl_content(ref_wav)
        ref32 = self._ref_audio_32k(ref_wav)

        # 1) encoder
        x, prompts = self.sess["encoder"].run(None, {
            "ref_seq": ref_seq, "text_seq": text_seq,
            "ref_bert": ref_bert, "text_bert": text_bert,
            "ssl_content": ssl})

        # 2) 首段解码
        y, k, v, y_emb, x_example = self.sess["fsdec"].run(
            None, {"x": x, "prompts": prompts})
        prefix_len = prompts.shape[1]

        # 3) 自回归解码
        t0 = time.time()
        n_steps = 0
        for n_steps in range(1, MAX_LOOP):
            y, k, v, y_emb, logits, samples = self.sess["sdec"].run(None, {
                "iy": y, "ik": k, "iv": v,
                "iy_emb": y_emb, "ix_example": x_example})
            if EARLY_STOP != -1 and (y.shape[1] - prefix_len) > EARLY_STOP:
                break
            if (int(np.argmax(logits, axis=-1).reshape(-1)[0]) == EOS
                    or int(np.asarray(samples).reshape(-1)[0]) == EOS):
                break
        y[0, -1] = 0
        pred_semantic = y[:, -n_steps:][None, ...]         # [1,1,T]
        ar_seconds = time.time() - t0

        # 4) 声码器
        wav = self.sess["vits"].run(None, {
            "text_seq": text_seq, "pred_semantic": pred_semantic,
            "ref_audio": ref32})[0]

        if verbose:
            print(f"    [onnx] AR {n_steps} 步 {ar_seconds:.1f}s  "
                  f"总 {ar_seconds:.1f}s")
        return np.asarray(wav).reshape(-1).astype(np.float32), SAMPLE_RATE
