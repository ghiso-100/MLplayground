import { useState } from 'react'
import type { Progress } from '../lib/progress'
import { stepKey } from '../lib/progress'
import type { StepId, Topic, WidgetId } from '../types'
import { GdWidget } from '../widgets/GdWidget'
import { KmeansWidget } from '../widgets/KmeansWidget'
import { KnnWidget } from '../widgets/KnnWidget'
import { OlsWidget } from '../widgets/OlsWidget'
import { CodeRunner } from './CodeRunner'
import { Rich } from './Rich'

const WIDGETS: Record<WidgetId, () => React.JSX.Element> = {
  ols: OlsWidget,
  gd: GdWidget,
  knn: KnnWidget,
  kmeans: KmeansWidget,
}

interface Props {
  topic: Topic
  step: StepId
  progress: Progress
}

export function StepContent({ topic, step, progress }: Props) {
  const [showSolution, setShowSolution] = useState(false)

  if (step === 'intuition') {
    const Widget = topic.widget ? WIDGETS[topic.widget] : null
    return (
      <>
        <Rich text={topic.intuition} />
        {Widget && <Widget />}
      </>
    )
  }

  if (step === 'math') {
    return (
      <>
        <Rich text={topic.math} />
        <p className="aside">
          Now open the{' '}
          <a href={topic.guide.url} target="_blank" rel="noreferrer">
            {topic.guide.label}
          </a>{' '}
          and find the same formula there. Copy it onto paper and label every symbol.
        </p>
      </>
    )
  }

  if (step === 'paper') {
    return (
      <>
        <div className="exercise">
          <Rich text={topic.paper.task} />
        </div>
        <p className="aside">Grab a pen. Work it out completely before you reveal the solution.</p>
        <button className="btn-reveal" aria-expanded={showSolution} onClick={() => setShowSolution((s) => !s)}>
          {showSolution ? 'Hide solution' : 'Reveal solution'}
        </button>
        {showSolution && (
          <div className="solution">
            <Rich text={topic.paper.solution} />
          </div>
        )}
      </>
    )
  }

  const data = step === 'code' ? topic.code : topic.real
  const saved = progress.state.code[stepKey(topic.id, step)]
  return (
    <>
      <Rich text={data.task} />
      <CodeRunner
        key={`${topic.id}:${step}`}
        code={saved ?? data.starter}
        starter={data.starter}
        local={data.local}
        onChange={(c) => progress.setCode(topic.id, step, c)}
      />
    </>
  )
}
