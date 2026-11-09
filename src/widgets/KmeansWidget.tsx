import { useMemo, useState } from 'react'
import { Playground, gaussian, mulberry32, scale } from './shared'

const W = 600
const H = 360
const PAD = 20
const x = scale(0, 10, PAD, W - PAD)
const y = scale(0, 6, H - PAD, PAD)
const K = 3
const CLASSES = ['pt-a', 'pt-b', 'pt-c']

type Pt = { x: number; y: number }

function makeData(): Pt[] {
  const r = mulberry32(21)
  const centers = [
    [2.4, 4.2],
    [5.6, 1.8],
    [7.8, 4.4],
  ]
  return centers.flatMap(([cx, cy]) =>
    Array.from({ length: 22 }, () => ({ x: cx + gaussian(r) * 0.85, y: cy + gaussian(r) * 0.7 })),
  ).filter((p) => p.x > 0.25 && p.x < 9.75 && p.y > 0.25 && p.y < 5.75)
}

function randomCenters(seed: number): Pt[] {
  const r = mulberry32(seed)
  return Array.from({ length: K }, () => ({ x: 0.8 + r() * 8.4, y: 0.6 + r() * 4.8 }))
}

export function KmeansWidget() {
  const data = useMemo(makeData, [])
  const [seed, setSeed] = useState(3)
  const [centers, setCenters] = useState<Pt[]>(() => randomCenters(3))
  const [labels, setLabels] = useState<number[] | null>(null)
  const [phase, setPhase] = useState<'assign' | 'update'>('assign')
  const [iter, setIter] = useState(0)
  const [converged, setConverged] = useState(false)

  const assign = () => {
    const next = data.map((p) => {
      let best = 0
      let bd = Infinity
      centers.forEach((c, i) => {
        const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2
        if (d < bd) {
          bd = d
          best = i
        }
      })
      return best
    })
    if (labels && next.every((l, i) => l === labels[i]) && iter > 0) setConverged(true)
    setLabels(next)
    setPhase('update')
  }

  const update = () => {
    if (!labels) return
    setCenters((cs) =>
      cs.map((c, k) => {
        const members = data.filter((_, i) => labels[i] === k)
        if (!members.length) return c
        return {
          x: members.reduce((s, p) => s + p.x, 0) / members.length,
          y: members.reduce((s, p) => s + p.y, 0) / members.length,
        }
      }),
    )
    setIter((n) => n + 1)
    setPhase('assign')
  }

  const restart = () => {
    const s = seed + 1
    setSeed(s)
    setCenters(randomCenters(s))
    setLabels(null)
    setPhase('assign')
    setIter(0)
    setConverged(false)
  }

  const inertia = labels
    ? data.reduce((s, p, i) => s + (p.x - centers[labels[i]].x) ** 2 + (p.y - centers[labels[i]].y) ** 2, 0)
    : null

  return (
    <Playground
      title="K-means playground"
      instructions="Alternate the two steps yourself and watch the crosses (centers) walk toward the groups."
      readout={
        <>
          <span>
            iteration <b>{iter}</b>
          </span>
          <span>
            inertia <b>{inertia === null ? '–' : inertia.toFixed(1)}</b>
          </span>
          <span>{converged ? <b>Converged: assignments stopped changing.</b> : `Next: ${phase === 'assign' ? 'assign points to the nearest center' : 'move centers to the mean of their points'}`}</span>
        </>
      }
      controls={
        <>
          <button className="btn-quiet" onClick={assign} disabled={phase !== 'assign' || converged}>
            Assign
          </button>
          <button className="btn-quiet" onClick={update} disabled={phase !== 'update' || converged}>
            Update centers
          </button>
          <button className="btn-quiet" onClick={restart}>
            New random start
          </button>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="plot">
        <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} className="frame" />
        {labels &&
          data.map((p, i) => (
            <line key={`l${i}`} x1={x(p.x)} y1={y(p.y)} x2={x(centers[labels[i]].x)} y2={y(centers[labels[i]].y)} className="km-link" />
          ))}
        {data.map((p, i) => (
          <circle key={i} cx={x(p.x)} cy={y(p.y)} r={5.5} className={`pt ${labels ? CLASSES[labels[i]] : 'pt-none'}`} />
        ))}
        {centers.map((c, k) => (
          <g key={k} className={`km-center ${CLASSES[k]}`} style={{ transform: `translate(${x(c.x)}px, ${y(c.y)}px)` }}>
            <circle r={13} className="km-center-bg" />
            <path d="M -7 -7 L 7 7 M 7 -7 L -7 7" />
          </g>
        ))}
      </svg>
    </Playground>
  )
}
