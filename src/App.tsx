import { useEffect, useMemo, useState } from 'react'
import { LoopTrack } from './components/LoopTrack'
import { Sidebar } from './components/Sidebar'
import { StepContent } from './components/StepContent'
import { ALL_TOPICS, TOPIC_BY_ID } from './content'
import { stepKey, useProgress } from './lib/progress'
import { STEPS, type StepId } from './types'

type Theme = 'light' | 'dark'

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem('mlplayground:theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* ignore */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function readHash(): { topic?: string; step?: StepId } {
  const [topic, step] = window.location.hash.replace(/^#\/?/, '').split('/')
  return {
    topic: TOPIC_BY_ID.has(topic) ? topic : undefined,
    step: STEPS.some((s) => s.id === step) ? (step as StepId) : undefined,
  }
}

export default function App() {
  const progress = useProgress()
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [navOpen, setNavOpen] = useState(false)

  const [location, setLocation] = useState(() => {
    const h = readHash()
    const last = progress.state.last
    return {
      topic: h.topic ?? last?.topic ?? ALL_TOPICS[0].topic.id,
      step: h.step ?? (h.topic ? 'intuition' : last?.step) ?? ('intuition' as StepId),
    }
  })

  const flat = TOPIC_BY_ID.get(location.topic) ?? ALL_TOPICS[0]
  const { topic, phase, phaseIndex } = flat
  const stepIndex = STEPS.findIndex((s) => s.id === location.step)
  const stepMeta = STEPS[stepIndex]

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('mlplayground:theme', theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  const { setLast } = progress
  useEffect(() => {
    const hash = `#/${location.topic}/${location.step}`
    if (window.location.hash !== hash) history.replaceState(null, '', hash)
    setLast(location.topic, location.step)
  }, [location, setLast])

  useEffect(() => {
    const onHash = () => {
      const h = readHash()
      if (h.topic) setLocation({ topic: h.topic, step: h.step ?? 'intuition' })
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const go = (topicId: string, step: StepId = 'intuition') => {
    setLocation({ topic: topicId, step })
    setNavOpen(false)
    document.querySelector('.main')?.scrollTo({ top: 0 })
  }

  const goStep = (step: StepId) => {
    setLocation((l) => ({ ...l, step }))
    document.querySelector('.step-head')?.scrollIntoView({ block: 'nearest' })
  }

  const isDone = (s: StepId) => progress.isDone(topic.id, s)
  const done = isDone(location.step)
  const allDone = STEPS.every((s) => isDone(s.id))
  const nextTopic = ALL_TOPICS[flat.index + 1]
  const prevStep = STEPS[stepIndex - 1]
  const nextStep = STEPS[stepIndex + 1]

  const completeAndContinue = () => {
    if (!done) progress.toggleDone(topic.id, location.step)
    if (nextStep) goStep(nextStep.id)
  }

  const note = progress.state.notes[stepKey(topic.id, location.step)] ?? ''
  const topicNumber = useMemo(() => phase.topics.findIndex((t) => t.id === topic.id) + 1, [phase, topic])

  return (
    <div className={`app ${navOpen ? 'nav-open' : ''}`}>
      <button className="nav-toggle" onClick={() => setNavOpen((o) => !o)} aria-expanded={navOpen}>
        {navOpen ? 'Close' : 'Topics'}
      </button>
      <div className="scrim" onClick={() => setNavOpen(false)} />

      <Sidebar
        key={phase.id}
        activeTopic={topic.id}
        progress={progress}
        onSelect={(id) => go(id)}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
      />

      <main className="main">
        <div className="sheet">
          <header className="topic-head">
            <p className="crumb">
              Phase {phaseIndex}: {phase.title}
              <span>
                Topic {topicNumber} of {phase.topics.length}
              </span>
            </p>
            <h2>{topic.title}</h2>
            <p className="summary">{topic.summary}</p>
            <a className="guide-link" href={topic.guide.url} target="_blank" rel="noreferrer">
              {topic.guide.label}
            </a>
          </header>

          <LoopTrack current={location.step} isDone={isDone} onSelect={goStep} />

          <section className="step" aria-labelledby="step-title">
            <div className="step-head">
              <h3 id="step-title">
                <span className="step-index">{stepIndex + 1}</span>
                {stepMeta.verb}
              </h3>
              <p className="step-hint">{stepMeta.hint}</p>
            </div>

            <div className="step-body">
              <StepContent key={`${topic.id}:${location.step}`} topic={topic} step={location.step} progress={progress} />
            </div>

            <label className="notes">
              <span>My notes for this step</span>
              <textarea
                value={note}
                placeholder={
                  location.step === 'intuition'
                    ? 'The core idea in one sentence, in your own words…'
                    : location.step === 'real'
                      ? 'What does the result mean in the real world?'
                      : 'What clicked, what is still unclear…'
                }
                onChange={(e) => progress.setNote(topic.id, location.step, e.target.value)}
              />
            </label>

            <div className="step-nav">
              <button className="btn-quiet" onClick={() => prevStep && goStep(prevStep.id)} disabled={!prevStep}>
                Previous step
              </button>
              <button className={`btn-done ${done ? 'is-done' : ''}`} onClick={() => progress.toggleDone(topic.id, location.step)} aria-pressed={done}>
                {done ? 'Step done' : 'Mark step done'}
              </button>
              {nextStep ? (
                <button className="btn-primary" onClick={completeAndContinue}>
                  {done ? 'Next step' : 'Done, next step'}
                </button>
              ) : nextTopic ? (
                <button
                  className="btn-primary"
                  onClick={() => {
                    if (!done) progress.toggleDone(topic.id, location.step)
                    go(nextTopic.topic.id)
                  }}
                >
                  {allDone || done ? 'Next topic' : 'Done, next topic'}
                </button>
              ) : (
                <span className="finish">Last topic of the plan</span>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  )
}
