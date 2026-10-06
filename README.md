# dsh-zhuang-fangyi-voice

**庄方宜专属中文音色** —— 说任意中文台词，用她的声音。
作为 [`dsh-zhuang-fangyi-pet`](../dsh-zhuang-fangyi-pet) 的语音子插件。

实测性能见文末[基准数据](#基准数据)。**同类工具中体积最小、依赖最轻**：
不需要 CUDA、不需要 torch（ONNX 后端），一条命令即可出声。

---

## 它提供哪些用法

| 方式 | 适用场景 | 命令/代码 |
|---|---|---|
| **命令行** | 快速生成几条 | `python -m zfh_voice say "台词"` |
| **批量** | 一次生成一整批台词 | `python -m zfh_voice batch lines.txt -d out/` |
| **Python API** | 嵌进别的 Python 程序 | `from zfh_voice import TTS` |
| **HTTP 服务** | **桌宠等外部程序调用** | `python -m zfh_voice serve --port 8765` |
| **合成缓存** | 相同台词秒回，不重复推理 | 自动，无需配置 |

两种推理后端可切换：

| 后端 | 依赖 | 速度 | 适用 |
|---|---|---|---|
| **onnx**（默认） | 仅 onnxruntime | 慢（RTF ≈ 6） | 分发、无显卡机器、跨平台 |
| **torch** | torch + GPT-SoVITS 源码 | **快（RTF ≈ 0.45 @GPU）** | 自己机器、批量生成 |

---

## 快速开始

```bash
git clone https://github.com/yangwenjie1231/dsh-zhuang-fangyi-voice
cd dsh-zhuang-fangyi-voice

pip install -r requirements.txt
python download_models.py            # 下载模型（约 1.4GB）

python -m zfh_voice status           # 确认模型就绪
python -m zfh_voice say "今天天气不错，我们一起出去走走吧。"
```

输出默认落在 `out/out.wav`。

### 模型下载源

`download_models.py` **默认优先 ModelScope**（国内快），失败自动回退 GitHub：

| 源 | 印尼 |
|---|---|
| ModelScope | `https://modelscope.cn/models/yangwenjie1231/dsh-zhuang-fangyi-voice` |
| GitHub | `https://github.com/yangwenjie1231/dsh-zhuang-fangyi-voice/releases` |

也可手动指定：`python download_models.py --source github`。

### 想更快？用 torch 后端

```bash
pip install torch torchaudio
git clone https://github.com/RVC-Boss/GPT-SoVITS third_party/GPT-SoVITS
# 把 Release 里的 .ckpt / .pth 放进对应目录
python -m zfh_voice --backend torch \
    --gsv-root third_party/GPT-SoVITS say "今天天气不错。"
```

---

## HTTP 服务（桌宠对接用）

```bash
python -m zfh_voice serve --host 127.0.0.1 --port 8765
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，返回后端类型与就绪状态 |
| GET | `/voices` | 可用音色列表 |
| POST | `/tts` | body `{"text":"...","seed":null}` → 返回 **audio/wav** |
| POST | `/tts.json` | 同上，但返回 `{ok,duration,wav_base64}` JSON |
| POST | `/cache/clear` | 清空合成缓存 |

示例：

```bash
curl -X POST http://127.0.0.1:8765/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"管理员，今天的实验数据我已经整理好了。"}' \
     -o reply.wav
```

> 服务已开启 CORS，浏览器端可直接 `fetch` 取 `audio/wav` 播放。
> 请求串行处理（推理后端非线程安全），并发请求会排队。

---

## Python API

```python
from zfh_voice import TTS

tts = TTS()                       # 默认 onnx 后端
r = tts.say("管理员，今天的实验数据我已经整理好了。")
print(r.duration, r.sr, r.cached)
r.save("reply.wav")

# 批量
texts = ["早上好。", "该休息一下了。", "任务完成，辛苦啦。"]
tts.say_many(texts, "out/")

# 直接拿字节（发给别处播放）
wav_bytes = r.to_wav_bytes()
```

---

## 配置

### 模型目录解析顺序

1. 环境变量 `ZFH_MODEL_DIR`
2. 仓库下的 `models/`
3. `~/.cache/zfh-voice/models`

### 参考音频

音色由**参考音频**驱动。默认使用模型包里的 `ref/default.wav`（约 8 秒）。
想换参考：

```python
tts = TTS(ref_wav="my_ref.wav", ref_text="这段音频里说的那句话")
```

**硬性要求**（与上游一致）：

- 时长 **3~10 秒**，超出直接报错
- `ref_text` 必须与音频内容**逐字一致**，否则音色会跑偏
- 优先选干净干声；带背景音乐的参考会把底噪一起学进输出

### 精度切换

`download_models.py --precision {fp16,int8,fp32}`

| 精度 | 体积 | 说明 |
|---|---:|---|
| fp16（默认） | 1.40 GB | 数值基本无损 |
| int8 | 0.87 GB | 更小，但音质需自行试听确认 |
| fp32 | 2.79 GB | 基准 |

### 用任意显卡加速

把 `onnxruntime` 换成 `onnxruntime-directml`，代码会自动优先选择
`DmlExecutionProvider`，Windows 上 NVIDIA / AMD / Intel 都能加速，无需 CUDA。

---

## 装成 DSH 插件（推荐）

本仓库除了是 Python 库，**也是一个 DSH 插件**：`index.js` 是它的宿主半边，
负责托管推理服务的进程（启动/保活/空闲释放显存），并把合成能力挂成宿主服务
**`zfhVoice`** —— 桌宠插件用它把 AI 的回答念出来。

```powershell
# 1) 装 Python 依赖与模型（见「快速开始」）
# 2) 复制插件到 DSH profile
powershell -NoProfile -ExecutionPolicy Bypass -File tools\install-dsh-plugin.ps1
# 3) 重启 DSH（ESM 模块有缓存，disable→enable 对入口文件不够）
```

装好后宿主日志会出现「庄方宜语音：已挂载 zfhVoice 服务」。

**给 AI 助手用的安装提示词**：`docs/安装提示词.md` —— 整段复制给
Claude Code / Cursor / DSH 自己，它会按步骤配好环境（含每一步的确认标志）。

### 插件配置（全部可选，留空即自动）

| 键 | 默认 | 说明 |
|---|---|---|
| `python` | 自动 | 找 `.venv` → `ZFH_VOICE_PYTHON` → PATH |
| `repoDir` | 插件自身目录 | 插件知道自己装在哪，**不需要绝对路径** |
| `backend` | `auto` | `auto` / `torch`（快，占显存）/ `onnx`（慢，省显存） |
| `gsvRoot` | 自动 | GPT-SoVITS 检出目录（torch 后端需要） |
| `modelDir` | 自动 | 等价 `ZFH_MODEL_DIR` |
| `host` / `port` | `127.0.0.1` / `8765` | 本地 HTTP 服务 |
| `resident` | `false` | **常驻**：一直占显存但合成立刻开始；关闭则空闲后释放 |
| `idleStopSec` | `300` | 非常驻时空闲多久停掉（0 = 不停） |

### 两条纪律（有测试盯着）

- **只停自己起的**：用户在终端手动 `python -m zfh_voice serve` 时，
  插件只连接、绝不终止；
- **秒退不重启**：依赖没装/模型缺失时自动重启会变成重启风暴 ——
  10 秒内退出只记错误，等用户显式操作。

自测：`node tools/test-plugin.mjs`（30 项，覆盖 python 解析/命令组装/
决策矩阵/监管器）。

---

## 与桌宠集成

桌宠侧只需调用 HTTP 接口，无需关心本仓库的实现细节：

```
桌宠 → POST http://127.0.0.1:8765/tts {"text": "..."} → audio/wav → 播放
```

若不想常驻服务，也可以直接调 CLI 生成到文件后播放：

```bash
python -m zfh_voice say "台词" -o cache/line.wav
```

固定台词建议走 CLI + 缓存，临时台词走 HTTP。

---

## 基准数据

测试机：RTX 5060 Laptop 8GB / 32 核 CPU / Windows

| 方案 | 模型体积 | RTF | 一句话（6 秒音频）耗时 |
|---|---:|---:|---:|
| torch + CUDA | ~14 GB（含环境） | **0.45** | ~3 s |
| torch + CPU | ~2.6 GB | 1.60 | ~10 s |
| **onnx + CPU**（默认） | **1.5 GB** | 6.1 | ~37 s |

说明：

- **ONNX 的瓶颈是自回归解码**：逐 token 生成（约 130 步），
  每步都要把 24 层 KV cache 传进传出，单步约 170ms。
- 精度对速度几乎无影响（fp32 RTF 6.12 / fp16 6.09）——瓶颈不在算力。
- 首次调用有预热开销（约 10s），之后走缓存或连续批量明显更快。
- **适合批量预生成，不适合逐字实时交互。**

### 质量指标

| 指标 | 数值 |
|---|---|
| 内容准确率（ASR 反查 CER） | 0.000 ~ 0.045 |
| 与 torch 版音色一致度（频谱包络相关） | 0.968 ~ 0.974 |

---

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `找不到完整的模型目录` | 先跑 `python download_models.py`，或设 `ZFH_MODEL_DIR` |
| `参考音频时长超出 3~10s` | 裁剪参考音频，或换一段 |
| 输出听起来不像本人 | 参考音频不合适：换干净的 3~10 秒单句 |
| 下载模型很慢 | GitHub 直连慢时可挂代理，或用 `--check` 确认已下好的部分 |
| `ZFH_G2PW_DIR 未设置` | 直接用 `TTS()` 会自动设置；手动 import 内嵌前端才会遇到 |

---

## 协议

代码 MIT。内嵌的文本前端来自 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT），
改动已在代码中标注。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

音色模型基于游戏角色语音微调，**仅供个人学习与同人创作**，
商业使用请自行取得授权。
