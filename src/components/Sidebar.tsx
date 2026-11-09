import { useRef, useState } from 'react'
import { PHASES } from '../content'
import type { Progress, ProgressState } from '../lib/progress'
import { STEPS } from '../types'

interface Props {
  activeTopic: string
  progress: Progress
  onSelect: (topicId: string) => void
  theme: 'light' | 'dark'
  onToggleTheme: () => void
}

const TOTAL_STEPS = PHASES.reduce((n, p) => n + p.topics.length, 0) * STEPS.length

export function Sidebar({ activeTopic, progress, onSelect, theme, onToggleTheme }: Props) {
  const activePhase = PHASES.find((p) => p.topics.some((t) => t.id === activeTopic))?.id
  const [open, setOpen] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(PHASES.map((p) => [p.id, p.id === activePhase])),
  )
  const fileInput = useRef<HTMLInputElement>(null)
  const [message, setMessage] = useState('')

  const doneSteps = Object.values(progress.state.done).reduce((n, d) => n + d.length, 0)
  const pct = Math.round((doneSteps / TOTAL_STEPS) * 100)

  const exportProgress = () => {
    const blob = new Blob([JSON.stringify(progress.state, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'ml-playground-progress.json'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const importProgress = async (file: File) => {
    try {
      const data = JSON.parse(await file.text()) as ProgressState
      if (typeof data !== 'object' || !data || typeof data.done !== 'object') throw new Error()
      progress.replaceAll(data)
      setMessage('Progress imported.')
    } catch {
      setMessage('That file is not a progress export. Choose the .json file you exported earlier.')
    }
  }

  return (
    <nav className="sidebar" aria-label="Curriculum">
      <header className="brand">
        <svg viewBox="0 0 64 64" className="brand-mark" aria-hidden="true">
          <circle cx="32" cy="32" r="20" fill="none" strokeWidth="6" strokeDasharray="94 32" strokeLinecap="round" />
          <circle cx="32" cy="12" r="6" />
        </svg>
        <div>
          <h1>ML Playground</h1>
          <p>Learning scikit-learn, one loop at a time</p>
        </div>
      </header>

      <div className="overall" aria-label={`${doneSteps} of ${TOTAL_STEPS} steps done`}>
        <div className="overall-bar">
          <span style={{ width: `${pct}%` }} />
        </div>
        <p>
          <strong>{doneSteps}</strong> of {TOTAL_STEPS} steps done
        </p>
      </div>

      <ol className="phases">
        {PHASES.map((phase, pi) => {
          const phaseDone = phase.topics.reduce((n, t) => n + (progress.state.done[t.id]?.length ?? 0), 0)
          const phaseTotal = phase.topics.length * STEPS.length
          const isOpen = open[phase.id]
          return (
            <li key={phase.id} className={`phase ${isOpen ? 'is-open' : ''}`}>
              <button
                className="phase-head"
                aria-expanded={isOpen}
                onClick={() => setOpen((o) => ({ ...o, [phase.id]: !o[phase.id] }))}
              >
                <span className="phase-num">{pi}</span>
                <span className="phase-title">
                  {phase.title}
                  <small>{phase.weeks}</small>
                </span>
                <span className="phase-count">
                  {phaseDone}/{phaseTotal}
                </span>
              </button>
              {isOpen && (
                <ul className="topics">
                  {phase.topics.map((t) => {
                    const done = progress.state.done[t.id] ?? []
                    const complete = done.length === STEPS.length
                    return (
                      <li key={t.id}>
                        <button
                          className={`topic ${t.id === activeTopic ? 'is-active' : ''} ${complete ? 'is-complete' : ''}`}
                          aria-current={t.id === activeTopic ? 'page' : undefined}
                          onClick={() => onSelect(t.id)}
                        >
                          <span className="topic-title">{t.title}</span>
                          <span className="pips" aria-label={`${done.length} of 5 steps done`}>
                            {STEPS.map((s) => (
                              <i key={s.id} className={done.includes(s.id) ? 'on' : ''} />
                            ))}
                          </span>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              )}
            </li>
          )
        })}
      </ol>

      <footer className="sidebar-foot">
        <button onClick={exportProgress}>Export progress</button>
        <button onClick={() => fileInput.current?.click()}>Import</button>
        <button onClick={onToggleTheme} aria-label="Toggle color theme">
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="application/json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) importProgress(f)
            e.target.value = ''
          }}
        />
        {message && (
          <p className="foot-msg" role="status">
            {message}
          </p>
        )}
      </footer>
    </nav>
  )
}
