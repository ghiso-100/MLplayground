import { useRef, useState, type KeyboardEvent } from 'react'
import { runPython, stopPython } from '../lib/python'

interface Props {
  code: string
  starter: string
  local?: boolean
  onChange: (code: string | null) => void
}

type Line = { text: string; stream: 'stdout' | 'stderr' | 'error' }

export function CodeRunner({ code, starter, local, onChange }: Props) {
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('')
  const [lines, setLines] = useState<Line[]>([])
  const [figures, setFigures] = useState<string[]>([])
  const [copied, setCopied] = useState(false)
  const area = useRef<HTMLTextAreaElement>(null)
  const edited = code !== starter

  const run = async () => {
    setRunning(true)
    setLines([])
    setFigures([])
    setStatus('Starting Python…')
    const result = await runPython(code, {
      onStatus: setStatus,
      onOutput: (text, stream) => setLines((l) => [...l, { text, stream }]),
    })
    setFigures(result.figures)
    if (!result.ok && result.error) setLines((l) => [...l, { text: result.error!, stream: 'error' }])
    setStatus(result.ok ? `Finished in ${result.seconds?.toFixed(1)} s` : 'Stopped with an error')
    setRunning(false)
  }

  const stop = () => {
    stopPython()
    setRunning(false)
    setStatus('Stopped. The next run starts a fresh Python.')
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      area.current?.select()
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      if (!running && !local) run()
      return
    }
    if (e.key === 'Tab' && !e.shiftKey) {
      e.preventDefault()
      const { selectionStart: s, selectionEnd: end } = el
      const next = code.slice(0, s) + '    ' + code.slice(end)
      onChange(next)
      requestAnimationFrame(() => el.setSelectionRange(s + 4, s + 4))
    }
  }

  const lineCount = code.split('\n').length

  return (
    <div className="runner">
      <div className="runner-bar">
        <span className="runner-lang">Python</span>
        {edited && <span className="runner-edited">edited</span>}
        <div className="runner-actions">
          {edited && (
            <button className="btn-quiet" onClick={() => onChange(null)}>
              Reset code
            </button>
          )}
          <button className="btn-quiet" onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
          {local ? null : running ? (
            <button className="btn-run is-stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button className="btn-run" onClick={run} title="Run (Ctrl/Cmd + Enter)">
              Run code
            </button>
          )}
        </div>
      </div>

      <div className="editor">
        <pre className="gutter" aria-hidden="true">
          {Array.from({ length: lineCount }, (_, i) => i + 1).join('\n')}
        </pre>
        <textarea
          ref={area}
          value={code}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          aria-label="Python code"
          rows={lineCount}
          onChange={(e) => onChange(e.target.value === starter ? null : e.target.value)}
          onKeyDown={onKeyDown}
        />
      </div>

      {local && (
        <p className="runner-note">
          This exercise downloads a dataset, which the browser can't do. Copy the code and run it in your
          virtual environment with <code>python exercise.py</code>.
        </p>
      )}

      {(status || lines.length > 0 || figures.length > 0) && (
        <div className="output" aria-live="polite">
          {status && <p className={`output-status ${running ? 'is-busy' : ''}`}>{status}</p>}
          {lines.length > 0 && (
            <pre className="output-text">
              {lines.map((l, i) => (
                <span key={i} className={`out-${l.stream}`}>
                  {l.text}
                  {'\n'}
                </span>
              ))}
            </pre>
          )}
          {figures.map((src, i) => (
            <img key={i} className="output-fig" src={`data:image/png;base64,${src}`} alt={`Plot ${i + 1}`} />
          ))}
        </div>
      )}
    </div>
  )
}
