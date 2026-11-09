import type { WorkerMessage } from './pyWorker'

export interface RunHandlers {
  onStatus: (text: string) => void
  onOutput: (text: string, stream: 'stdout' | 'stderr') => void
}

export interface RunResult {
  ok: boolean
  error?: string
  figures: string[]
  seconds?: number
}

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, { handlers: RunHandlers; resolve: (r: RunResult) => void }>()

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./pyWorker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const msg = e.data
      const job = pending.get(msg.id)
      if (!job) return
      if (msg.type === 'status') job.handlers.onStatus(msg.text)
      else if (msg.type === 'stdout' || msg.type === 'stderr') job.handlers.onOutput(msg.text, msg.type)
      else if (msg.type === 'done') {
        pending.delete(msg.id)
        job.resolve({ ok: true, figures: msg.figures, seconds: msg.seconds })
      } else if (msg.type === 'error') {
        pending.delete(msg.id)
        job.resolve({ ok: false, error: msg.text, figures: msg.figures })
      }
    }
  }
  return worker
}

export function runPython(code: string, handlers: RunHandlers): Promise<RunResult> {
  const id = nextId++
  return new Promise((resolve) => {
    pending.set(id, { handlers, resolve })
    getWorker().postMessage({ id, code })
  })
}

/** Kill a long-running script. The next run starts a fresh Python. */
export function stopPython() {
  worker?.terminate()
  worker = null
  for (const [, job] of pending) job.resolve({ ok: false, error: 'Stopped.', figures: [] })
  pending.clear()
}
