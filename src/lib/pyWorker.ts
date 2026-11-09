/// <reference lib="webworker" />
/**
 * Runs Python in a Web Worker with Pyodide, so the page stays responsive.
 * Packages (numpy, scikit-learn, matplotlib, ...) are loaded on demand from
 * the imports found in the code. Matplotlib figures are returned as PNGs.
 */

export const PYODIDE_VERSION = '0.29.3'
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

export type WorkerRequest = { id: number; code: string }
export type WorkerMessage =
  | { id: number; type: 'status'; text: string }
  | { id: number; type: 'stdout' | 'stderr'; text: string }
  | { id: number; type: 'done'; figures: string[]; seconds: number }
  | { id: number; type: 'error'; text: string; figures: string[] }

interface PyodideLike {
  loadPackagesFromImports(code: string, opts?: { messageCallback?: (m: string) => void }): Promise<void>
  loadPackage(names: string[]): Promise<void>
  runPythonAsync(code: string): Promise<unknown>
  setStdout(opts: { batched: (s: string) => void }): void
  setStderr(opts: { batched: (s: string) => void }): void
  globals: { get(name: string): { toJs(): string[]; destroy?: () => void } | undefined }
}

const ctx = self as unknown as DedicatedWorkerGlobalScope
let pyodidePromise: Promise<PyodideLike> | null = null
let currentId = 0

const SETUP = `
import os, sys, io, base64, warnings
os.environ["MPLBACKEND"] = "AGG"
warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg.*")
`

const COLLECT_FIGURES = `
_mlp_figs = []
if "matplotlib.pyplot" in sys.modules:
    import matplotlib.pyplot as _plt
    for _n in _plt.get_fignums():
        _buf = io.BytesIO()
        _plt.figure(_n).savefig(_buf, format="png", dpi=110, bbox_inches="tight")
        _mlp_figs.append(base64.b64encode(_buf.getvalue()).decode())
    _plt.close("all")
`

async function getPyodide(id: number): Promise<PyodideLike> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      ctx.postMessage({ id, type: 'status', text: 'Downloading Python (first run only)…' })
      const mod = await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`)
      const py = (await mod.loadPyodide({ indexURL: INDEX_URL })) as PyodideLike
      py.setStdout({ batched: (text) => ctx.postMessage({ id: currentId, type: 'stdout', text }) })
      py.setStderr({ batched: (text) => ctx.postMessage({ id: currentId, type: 'stderr', text }) })
      await py.runPythonAsync(SETUP)
      return py
    })()
    pyodidePromise.catch(() => {
      pyodidePromise = null
    })
  }
  return pyodidePromise
}

async function collectFigures(py: PyodideLike): Promise<string[]> {
  try {
    await py.runPythonAsync(COLLECT_FIGURES)
    const figs = py.globals.get('_mlp_figs')
    const list = figs ? figs.toJs() : []
    figs?.destroy?.()
    return list
  } catch {
    return []
  }
}

ctx.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const { id, code } = event.data
  currentId = id
  let py: PyodideLike
  try {
    py = await getPyodide(id)
  } catch (err) {
    ctx.postMessage({
      id,
      type: 'error',
      text: `Python could not be loaded: ${String(err)}\nCheck your internet connection, or copy the code and run it on your computer.`,
      figures: [],
    })
    return
  }

  const start = performance.now()
  try {
    ctx.postMessage({ id, type: 'status', text: 'Loading packages…' })
    // scikit-learn needs pandas for DataFrame output, but never imports it by name.
    if (/as_frame\s*=\s*True|set_output|fetch_openml/.test(code)) {
      await py.loadPackage(['pandas'])
    }
    await py.loadPackagesFromImports(code, {
      messageCallback: (m) => {
        if (m.startsWith('Loading')) ctx.postMessage({ id, type: 'status', text: m.replace(/,.*$/, '…') })
      },
    })
    ctx.postMessage({ id, type: 'status', text: 'Running…' })
    await py.runPythonAsync(code)
    const figures = await collectFigures(py)
    ctx.postMessage({ id, type: 'done', figures, seconds: (performance.now() - start) / 1000 })
  } catch (err) {
    const figures = await collectFigures(py)
    const text = String((err as Error)?.message ?? err)
    ctx.postMessage({ id, type: 'error', text: cleanTraceback(text), figures })
  }
}

/** Drop Pyodide's internal frames so the traceback starts at the learner's code. */
function cleanTraceback(text: string): string {
  const marker = text.indexOf('File "<exec>"')
  if (marker === -1) return text
  return 'Traceback (most recent call last):\n  ' + text.slice(marker)
}
