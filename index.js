/**
 * dsh-zhuang-fangyi-voice · DSH 插件（宿主半边）
 *
 * ── 它是什么 ──────────────────────────────────────────────────────────────
 *
 * 这个仓库的 Python 部分（`src/zfh_voice`）是**推理库**：给一句中文，出 WAV。
 * 本文件是它的 **DSH 插件外壳**，负责三件事：
 *
 *   1. **托管 Python 服务的进程**（启动 / 保活 / 空闲释放显存）；
 *   2. 把合成能力挂成宿主服务 **`zfhVoice`**（其它插件 `ctx.get('zfhVoice')` 就能用）；
 *   3. 给用户一个可配的接缝（后端、python、端口、常驻策略）。
 *
 * 桌宠插件就是这么用的：它拿到 `zfhVoice` 后把 AI 的回答合成出来播放。
 * **没装本插件时桌宠仍然能用固定台词** —— 两个插件彼此独立。
 *
 * ── 为什么不需要任何绝对路径 ──────────────────────────────────────────────
 *
 * 插件**知道自己装在哪**（`import.meta.url`）。所以：
 *   · 仓库目录默认 = 本文件所在目录；
 *   · python 默认 = 仓库里的 `.venv` → 环境变量 → PATH 上的 `python`；
 *   · 模型目录 = Python 侧的 `ZFH_MODEL_DIR` / `<repo>/models` / `~/.cache`。
 *
 * 于是同一份代码在任何人机器上都能跑，配置项全部是**可选覆盖**。
 *
 * ── 常驻 vs 释放（为什么这是个开关而不是常量）─────────────────────────────
 *
 * 启动服务要加载模型：onnx 约 10 秒；torch+CUDA 要把权重搬进显存，更慢，
 * 而且加载完长期占着显存。两个诉求冲突，所以让用户选：
 *
 *   · `resident: true` —— 一直活着；合成立刻开始。代价：显存常占。
 *   · `resident: false` —— 需要时才起，空闲 `idleStopSec` 秒后停掉释放显存。
 *
 * ── 两条纪律 ──────────────────────────────────────────────────────────────
 *
 *   · **只停自己起的**：用户在终端手动 `python -m zfh_voice serve` 时，
 *     插件只连接、绝不终止（`ours` 标志就是这件事的账本）。
 *   · **秒退不重启**：依赖没装/模型缺失/端口占用时自动重启会变成重启风暴。
 *     10 秒内退出只记 `lastError`，等用户显式操作。
 *
 * @module dsh-zhuang-fangyi-voice
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

/** 本插件的安装目录 —— 一切相对路径的锚点（因此不需要绝对路径配置）。 */
const HERE = path.dirname(fileURLToPath(import.meta.url))

export const name = 'dsh-zhuang-fangyi-voice'

/** 服务名（桌宠通过它取合成能力）。 */
export const SERVICE_NAME = 'zfhVoice'

/** 空闲多久停掉服务释放显存（秒）—— 默认 5 分钟。 */
export const DEFAULT_IDLE_STOP_SEC = 300

/** 启动后等健康检查的最长时间（CUDA 下加载模型可能几十秒）。 */
export const DEFAULT_START_TIMEOUT_MS = 120000

/** 短命退出判定：低于它就认为"起不来"，不再自动重启。 */
export const RAPID_EXIT_MS = 10000

/** 健康探测间隔。 */
export const HEALTH_POLL_MS = 15000

/** 合成结果体积上限（WAV 未压缩；6 秒 ≈ 380KB）。 */
export const MAX_SPEECH_BYTES = 8 * 1024 * 1024

// ─────────────────────────────────────────────────────────────────────────────
// 纯函数部分（可单测，见 tools/test-plugin.mjs）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 找一个能用的 python（**全部相对插件自身**，不含任何绝对路径硬编码）。
 *
 * 顺序：显式配置 → 环境变量 → 插件目录下的虚拟环境 → PATH。
 *
 * @param {object} o
 * @param {string} [o.python] - 显式配置
 * @param {string} [o.repoDir] - 插件目录
 * @param {(p: string) => boolean} [o.exists]
 * @param {object} [o.env]
 * @returns {string} 可执行文件名或路径（找不到就返回 `python`，让 PATH 兜底）
 */
export function resolvePython (o = {}) {
  const explicit = String(o.python ?? '').trim()
  if (explicit !== '') return explicit
  const env = o.env ?? (typeof process !== 'undefined' ? process.env : {})
  const fromEnv = String(env.ZFH_VOICE_PYTHON ?? '').trim()
  if (fromEnv !== '') return fromEnv
  const repoDir = String(o.repoDir ?? '').trim()
  const exists = typeof o.exists === 'function' ? o.exists : (() => false)
  if (repoDir !== '') {
    // 常见虚拟环境位置（Windows 与 POSIX 都试）——**相对插件目录**，可移植。
    for (const rel of [
      ['.venv', 'Scripts', 'python.exe'], ['.venv', 'bin', 'python'],
      ['venv', 'Scripts', 'python.exe'], ['venv', 'bin', 'python']
    ]) {
      const p = path.join(repoDir, ...rel)
      if (exists(p)) return p
    }
  }
  // 交给 PATH（名字由平台决定）
  return process.platform === 'win32' ? 'python' : 'python3'
}

/**
 * 组装启动命令。
 *
 * CLI 形状（本仓库 `cli.py`）——**全局参数必须在子命令前面**：
 *
 *     python -m zfh_voice --backend torch --gsv-root <dir> serve --host 127.0.0.1 --port 8765
 *
 * @param {object} o
 * @param {string} o.python
 * @param {string} o.repoDir
 * @param {string} [o.backend] - `auto` | `torch` | `onnx`
 * @param {string} [o.gsvRoot]
 * @param {string} [o.modelDir]
 * @param {string} [o.host]
 * @param {number} [o.port]
 * @returns {{argv: string[], cwd: string, env: Record<string,string>}|null}
 */
export function buildServeCommand (o = {}) {
  const python = String(o.python ?? '').trim()
  const repoDir = String(o.repoDir ?? '').trim()
  if (python === '' || repoDir === '') return null
  const host = String(o.host ?? '').trim() || '127.0.0.1'
  const port = Number.isFinite(Number(o.port)) && Number(o.port) > 0
    ? Math.floor(Number(o.port))
    : 8765
  const argv = [python, '-m', 'zfh_voice']
  const backend = String(o.backend ?? '').trim()
  // `auto` = 不传，交给 Python 侧自己选（它有二选一的逻辑）。
  if (backend === 'torch' || backend === 'onnx') argv.push('--backend', backend)
  const gsv = String(o.gsvRoot ?? '').trim()
  if (gsv !== '') argv.push('--gsv-root', gsv)
  const model = String(o.modelDir ?? '').trim()
  if (model !== '') argv.push('--model-dir', model)
  argv.push('serve', '--host', host, '--port', String(port))
  return {
    argv,
    cwd: repoDir,
    // `src/` 布局：从仓库直接跑必须把 `src` 放进 PYTHONPATH，否则
    // `python -m zfh_voice` 报 ModuleNotFoundError（没 pip install 时）。
    // 这一条让用户**不必先安装**就能跑起来。
    env: { PYTHONPATH: path.join(repoDir, 'src') }
  }
}

/**
 * **纯决策**：这一刻该对服务做什么。
 *
 * 穷举"常驻/非常驻 × 健康/不健康 × 谁起的 × 空闲多久"——这些组合在真实
 * 时序里很难复现，但两个方向的错都很贵且都发生在用户看不见的地方：
 * 该释放时不释放（显存白占）、不该杀时杀了（别人的进程）。
 *
 * @returns {{action: 'none'|'stop'|'restart', why: string}}
 */
export function nextServiceAction (s) {
  const resident = s.resident === true
  const ours = s.ours === true
  const healthy = s.healthy === true
  const launchable = s.launchable === true
  const idleMs = Number.isFinite(s.idleMs) ? s.idleMs : 0
  const idleLimitMs = Number.isFinite(s.idleLimitMs) ? s.idleLimitMs : 0

  // ① 不是我们起的 → 永不碰它（用户自己开的服务）。
  if (!ours) return { action: 'none', why: 'not-ours' }

  // ② 我们起的但不健康 → 可能崩了。
  if (!healthy) {
    if (!resident) return { action: 'none', why: 'down-not-resident' }
    if (!launchable) return { action: 'none', why: 'down-no-launch' }
    if (s.rapidExit === true) return { action: 'none', why: 'down-rapid-exit' }
    return { action: 'restart', why: 'down-resident' }
  }

  // ③ 健康 + 我们起的：常驻留着；非常驻空闲够了就停。
  if (resident) return { action: 'none', why: 'resident-keep' }
  if (idleLimitMs > 0 && idleMs >= idleLimitMs) return { action: 'stop', why: 'idle-stop' }
  return { action: 'none', why: 'in-use' }
}

/** 解析 WAV 时长（ms）—— `POST /tts` 只给字节，时长得自己从头部读。 */
export function wavDurationMs (bytes) {
  try {
    if (bytes === null || bytes === undefined) return 0
    const b = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes)
    if (b.length < 44) return 0
    if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') return 0
    let byteRate = 0
    let dataLen = 0
    let off = 12
    while (off + 8 <= b.length) {
      const id = b.toString('ascii', off, off + 4)
      const size = b.readUInt32LE(off + 4)
      const body = off + 8
      if (id === 'fmt ' && body + 16 <= b.length) byteRate = b.readUInt32LE(body + 8)
      else if (id === 'data') dataLen = Math.min(size, b.length - body)
      off = body + size + (size % 2)
    }
    if (byteRate <= 0 || dataLen <= 0) return 0
    return Math.round((dataLen / byteRate) * 1000)
  } catch {
    return 0
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 配置
// ─────────────────────────────────────────────────────────────────────────────

/** 插件配置默认值（全空 = 自动，见文件头"为什么不需要绝对路径"）。 */
export function defaultConfig () {
  return {
    /** python 可执行文件。留空 = .venv → ZFH_VOICE_PYTHON → PATH。 */
    python: '',
    /** 本仓库目录。留空 = 插件自身目录。 */
    repoDir: '',
    /** 推理后端：`auto`（默认）/ `torch`（快，占显存）/ `onnx`（慢，省显存）。 */
    backend: 'auto',
    /** GPT-SoVITS 检出目录（torch 后端需要）。留空 = 交给 Python 侧找。 */
    gsvRoot: '',
    /** 模型目录（等价 ZFH_MODEL_DIR）。留空 = <repo>/models → ~/.cache。 */
    modelDir: '',
    host: '127.0.0.1',
    port: 8765,
    /** 常驻：true = 一直占着（快）；false = 空闲后释放显存。 */
    resident: false,
    /** 非常驻时，空闲多少秒停掉（0 = 不停）。 */
    idleStopSec: DEFAULT_IDLE_STOP_SEC,
    /** 等模型加载的上限。 */
    startTimeoutMs: DEFAULT_START_TIMEOUT_MS
  }
}

/** 归一化配置（只认已知键、夹取数值、过滤非法后端）。 */
export function normalizeConfig (input, base = defaultConfig()) {
  const out = { ...base }
  if (input === null || typeof input !== 'object') return out
  for (const key of ['python', 'repoDir', 'gsvRoot', 'modelDir']) {
    if (typeof input[key] === 'string' && input[key].length <= 500) out[key] = input[key].trim()
  }
  if (input.backend === 'auto' || input.backend === 'torch' || input.backend === 'onnx') {
    out.backend = input.backend
  }
  if (typeof input.host === 'string' && /^[0-9a-zA-Z.:-]{1,64}$/.test(input.host)) {
    out.host = input.host
  }
  const port = Number(input.port)
  if (Number.isFinite(port) && port > 0 && port < 65536) out.port = Math.floor(port)
  if (typeof input.resident === 'boolean') out.resident = input.resident
  const idle = Number(input.idleStopSec)
  if (Number.isFinite(idle) && idle >= 0 && idle <= 86400) out.idleStopSec = Math.floor(idle)
  const timeout = Number(input.startTimeoutMs)
  if (Number.isFinite(timeout) && timeout >= 1000 && timeout <= 600000) {
    out.startTimeoutMs = Math.floor(timeout)
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// 运行时（HTTP 客户端 + 进程监管）
// ─────────────────────────────────────────────────────────────────────────────

/** 建 TTS HTTP 客户端（只依赖 Python 侧已文档化的契约）。 */
function createClient (cfg, logger) {
  const baseUrl = `http://${cfg.host}:${cfg.port}`
  const fetchImpl = typeof fetch === 'function' ? fetch : null

  const json = async (url, init, timeoutMs) => {
    if (fetchImpl === null) return null
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), timeoutMs)
    try {
      const res = await fetchImpl(url, { ...init, signal: ctrl.signal })
      if (res === null || res.ok !== true) return null
      return await res.json()
    } catch { return null } finally { clearTimeout(timer) }
  }

  return {
    baseUrl,
    async health () {
      const body = await json(`${baseUrl}/health`, { method: 'GET' }, 5000)
      if (body?.ok !== true) return null
      return {
        ok: true,
        backend: typeof body.backend === 'string' ? body.backend : null,
        version: typeof body.version === 'string' ? body.version : null
      }
    },
    async synthesize (text) {
      if (fetchImpl === null) return null
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 60000)
      try {
        const res = await fetchImpl(`${baseUrl}/tts`, {
          method: 'POST',
          signal: ctrl.signal,
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ text })
        })
        if (res === null || res.ok !== true) return null
        const buf = Buffer.from(await res.arrayBuffer())
        if (buf.length === 0 || buf.length > MAX_SPEECH_BYTES) return null
        if (buf.toString('ascii', 0, 4) !== 'RIFF') {
          logger?.warn?.('zhuang-fangyi-voice: /tts 返回的不是 RIFF/WAV，已丢弃')
          return null
        }
        return { bytes: buf, durationMs: wavDurationMs(buf) }
      } catch (error) {
        logger?.warn?.(`zhuang-fangyi-voice: 合成失败：${error?.message ?? error}`)
        return null
      } finally { clearTimeout(timer) }
    }
  }
}

/**
 * 进程监管器（启动/保活/空闲释放）。
 *
 * @param {object} o
 * @param {(spec: object) => object} o.spawn - 受管子进程接缝
 * @param {() => Promise<object|null>} o.health
 * @param {object} [o.logger]
 * @param {() => number} [o.now]
 */
export function createSupervisor (o = {}) {
  const spawn = typeof o.spawn === 'function' ? o.spawn : null
  const health = typeof o.health === 'function' ? o.health : async () => null
  const logger = o.logger ?? null
  const now = typeof o.now === 'function' ? o.now : () => Date.now()

  let cfg = defaultConfig()
  let launch = null
  let handle = null
  let startedAt = 0
  let exitedAt = 0
  let lastUseAt = now()
  let lastError = null
  let backend = null
  let starting = null

  let _spawnCount = 0

  const launchable = () => launch !== null && spawn !== null

  const probe = async () => {
    const h = await health()
    if (h !== null && h !== undefined && h.ok === true) {
      backend = h.backend ?? backend
      return true
    }
    return false
  }

  const ensure = async (opts = {}) => {
    lastUseAt = now()
    if (await probe()) { exitedAt = 0; return true }
    if (!launchable()) return false
    if (starting !== null) return starting
    starting = (async () => {
      try {
        logger?.info?.('庄方宜语音：启动推理服务…')
        handle = spawn({
          argv: launch.argv,
          cwd: launch.cwd,
          env: launch.env,
          stdio: 'ignore',
          graceMs: 2000
        })
        _spawnCount += 1
        startedAt = now()
        exitedAt = 0
        lastError = null
        try {
          handle?.done?.then(value => {
            exitedAt = now()
            const ms = startedAt > 0 ? exitedAt - startedAt : 0
            if (ms < RAPID_EXIT_MS) {
              lastError = `服务启动后 ${Math.round(ms / 1000)} 秒就退出（exit=${value?.exitCode ?? '?'}）` +
                ' —— 多半是依赖没装或模型没下载，见 docs/安装提示词.md'
              logger?.warn?.(`庄方宜语音：${lastError}`)
            }
            handle = null
          }).catch(() => {})
        } catch { /* done 不是 promise 也无妨 */ }
        if (opts.wait === false) return true
        const deadline = now() + cfg.startTimeoutMs
        while (now() < deadline) {
          if (await probe()) {
            logger?.info?.(`庄方宜语音：服务就绪（${Math.round((now() - startedAt) / 1000)}s，backend=${backend ?? '?'}）`)
            return true
          }
          if (handle === null && exitedAt > 0) return false
          await new Promise(r => setTimeout(r, 1000))
        }
        lastError = `等 ${Math.round(cfg.startTimeoutMs / 1000)} 秒仍未就绪（模型可能太大）`
        return false
      } catch (error) {
        lastError = String(error?.message ?? error)
        logger?.warn?.(`庄方宜语音：启动服务失败：${lastError}`)
        return false
      } finally {
        starting = null
      }
    })()
    return starting
  }

  const stop = () => {
    if (handle === null) return false
    try {
      handle.terminate?.()
      logger?.info?.('庄方宜语音：已停止服务（释放显存）')
      handle = null
      startedAt = 0
      return true
    } catch (error) {
      logger?.warn?.(`庄方宜语音：停止服务失败：${error?.message ?? error}`)
      return false
    }
  }

  const tick = async () => {
    const healthy = await probe().catch(() => false)
    const decision = nextServiceAction({
      resident: cfg.resident,
      ours: handle !== null || (exitedAt > 0 && startedAt > 0),
      healthy,
      idleMs: now() - lastUseAt,
      idleLimitMs: Math.max(0, Number(cfg.idleStopSec) || 0) * 1000,
      launchable: launchable(),
      rapidExit: lastError !== null && lastError.includes('就退出')
    })
    if (decision.action === 'stop') stop()
    else if (decision.action === 'restart') await ensure()
    return decision
  }

  return {
    setConfig (next, nextLaunch) {
      cfg = next
      if (nextLaunch !== undefined) launch = nextLaunch
    },
    ensure,
    stop,
    tick,
    markUse: () => { lastUseAt = now() },
    status () {
      return {
        resident: cfg.resident,
        launchable: launchable(),
        ours: handle !== null,
        pid: handle?.pid ?? null,
        ready: handle !== null,
        backend,
        lastError,
        idleMs: now() - lastUseAt,
        idleStopSec: cfg.idleStopSec,
        launch: launch === null ? null : { argv: launch.argv, cwd: launch.cwd },
        spawnCount: _spawnCount
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 插件入口
// ─────────────────────────────────────────────────────────────────────────────

/**
 * cordis 插件入口。
 *
 * @param {object} ctx - cordis Context
 * @param {object} [rawConfig] - 插件配置（cordis.patch.yml 或设置页给的）
 */
export function apply (ctx, rawConfig) {
  const logger = ctx.logger ?? console
  const cfg = normalizeConfig(rawConfig)

  const repoDir = cfg.repoDir !== '' ? cfg.repoDir : HERE
  const python = resolvePython({
    python: cfg.python,
    repoDir,
    exists: p => { try { return fs.existsSync(p) } catch { return false } }
  })
  const launch = buildServeCommand({
    python,
    repoDir,
    backend: cfg.backend,
    gsvRoot: cfg.gsvRoot,
    modelDir: cfg.modelDir,
    host: cfg.host,
    port: cfg.port
  })

  const client = createClient(cfg, logger)
  const supervisor = createSupervisor({
    spawn: spec => {
      let sub = null
      try {
        sub = typeof ctx.get === 'function' ? ctx.get('subprocess') : undefined
      } catch { sub = null }
      if (sub === null || sub === undefined || typeof sub.spawn !== 'function') {
        throw new Error('宿主未提供 ctx.subprocess，无法托管推理服务进程')
      }
      return sub.spawn(spec)
    },
    health: () => client.health(),
    logger
  })
  supervisor.setConfig(cfg, launch)

  // 周期检查：常驻保活 / 非常驻释放显存。
  ctx.effect(() => {
    const timer = setInterval(() => { void supervisor.tick().catch(() => {}) }, HEALTH_POLL_MS)
    timer.unref?.()
    return () => {
      clearInterval(timer)
      try { supervisor.stop() } catch { /* 已退出 */ }
    }
  }, 'zhuang-fangyi-voice: service supervisor')

  /**
   * 对外服务：其它插件用它把一句话变成语音。
   *
   * 桌宠的用法：`const v = ctx.get('zfhVoice'); const r = await v.synthesize(text)`。
   */
  const service = {
    /** 合成一句（自动确保服务在跑；失败返回 null，绝不抛）。 */
    async synthesize (text) {
      if (typeof text !== 'string' || text.trim() === '') return null
      const ok = await supervisor.ensure()
      if (!ok) return null
      const r = await client.synthesize(text)
      supervisor.markUse()
      return r === null ? null : { ...r, backend: supervisor.status().backend }
    },
    /** 健康检查（不启动服务）。 */
    health: () => client.health(),
    /** 只读状态。 */
    status: () => ({
      ...supervisor.status(),
      baseUrl: client.baseUrl,
      python,
      repoDir,
      backend_pref: cfg.backend
    }),
    /** 显式启动（会等模型加载完）。 */
    start: () => supervisor.ensure(),
    /** 停掉**本插件起的**服务（用户自己起的那份不会被动）。 */
    stop: () => supervisor.stop(),
    /** 由宿主插件（如桌宠）驱动配置 —— 让开关只有一处。 */
    configure (patch) {
      const next = normalizeConfig({ ...cfg, ...(patch ?? {}) }, cfg)
      const nextRepo = next.repoDir !== '' ? next.repoDir : HERE
      const nextLaunch = buildServeCommand({
        python: resolvePython({
          python: next.python,
          repoDir: nextRepo,
          exists: p => { try { return fs.existsSync(p) } catch { return false } }
        }),
        repoDir: nextRepo,
        backend: next.backend,
        gsvRoot: next.gsvRoot,
        modelDir: next.modelDir,
        host: next.host,
        port: next.port
      })
      Object.assign(cfg, next)
      supervisor.setConfig(cfg, nextLaunch)
      void supervisor.tick().catch(() => {})
      return this.status()
    }
  }

  ctx.provide(SERVICE_NAME, service)
  ctx.effect(() => () => {
    try { supervisor.stop() } catch { /* 已退出 */ }
  }, 'zhuang-fangyi-voice: dispose service')

  logger.info?.(`庄方宜语音：已挂载 ${SERVICE_NAME} 服务（repo=${repoDir}，python=${python}，` +
    `resident=${cfg.resident}）`)
  return service
}

export default { name, apply }
