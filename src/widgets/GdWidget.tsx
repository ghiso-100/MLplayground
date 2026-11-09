import { useEffect, useRef, useState } from 'react'
import { Axes, Playground, scale } from './shared'

// Same tiny dataset as the paper exercises: (1,2), (2,4), (3,5); model y = w x
const XS = [1, 2, 3]
const YS = [2, 4, 5]
const loss = (w: number) => XS.reduce((s, x, i) => s + (YS[i] - w * x) ** 2, 0) / XS.length
const grad = (w: number) => (2 / XS.length) * XS.reduce((s, x, i) => s + (w * x - YS[i]) * x, 0)
const W_BEST = 25 / 14

const W = 600
const H = 340
const PAD = 40
const xs = scale(-1, 4.5, PAD, W - PAD)
const ys = scale(0, 40, H - PAD, PAD)
const Y_MAX = 40

const curve = Array.from({ length: 111 }, (_, i) => -1 + (i * 5.5) / 110)
  .map((w, i) => `${i ? 'L' : 'M'} ${xs(w).toFixed(1)} ${ys(loss(w)).toFixed(1)}`)
  .join(' ')

const START_W = -0.5

export function GdWidget() {
  const [lr, setLr] = useState(0.05)
  const [path, setPath] = useState<number[]>([START_W])
  const [auto, setAuto] = useState(false)
  const timer = useRef<number | null>(null)
  const w = path[path.length - 1]
  const diverged = Math.abs(w) > 20 || !Number.isFinite(w)

  const step = () => setPath((p) => {
    const last = p[p.length - 1]
    if (Math.abs(last) > 20) return p
    return [...p, last - lr * grad(last)]
  })

  useEffect(() => {
    if (!auto) return
    timer.current = window.setInterval(step, 380)
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  })

  useEffect(() => {
    if (auto && (path.length > 40 || diverged || Math.abs(grad(w)) < 1e-3)) setAuto(false)
  }, [auto, path.length, diverged, w])

  const reset = () => {
    setAuto(false)
    setPath([START_W])
  }

  const visible = path.filter((v) => v >= -1 && v <= 4.5)

  return (
    <Playground
      title="Gradient descent playground"
      instructions="Each step moves w against the slope. Change the learning rate and watch how the steps behave."
      readout={
        <>
          <span>
            step <b>{path.length - 1}</b>
          </span>
          <span>
            w = <b>{diverged ? '∞' : w.toFixed(3)}</b>
          </span>
          <span>
            loss <b>{diverged ? '∞' : loss(w).toFixed(3)}</b>
          </span>
          <span>
            slope <b>{diverged ? '–' : grad(w).toFixed(3)}</b>
          </span>
          {diverged && <span className="warn">Diverged: the learning rate is too large.</span>}
        </>
      }
      controls={
        <>
          <label className="slider">
            learning rate <b>{lr.toFixed(3)}</b>
            <input
              type="range"
              min={0.005}
              max={0.3}
              step={0.005}
              value={lr}
              onChange={(e) => {
                setLr(Number(e.target.value))
                reset()
              }}
            />
          </label>
          <button className="btn-quiet" onClick={step} disabled={diverged}>
            One step
          </button>
          <button className="btn-quiet" onClick={() => setAuto((a) => !a)} disabled={diverged}>
            {auto ? 'Pause' : 'Run'}
          </button>
          <button className="btn-quiet" onClick={reset}>
            Reset
          </button>
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="plot">
        <Axes width={W} height={H} pad={PAD} xTicks={[-1, 0, 1, 2, 3, 4]} yTicks={[0, 10, 20, 30, 40]} x={xs} y={ys} xLabel="w" yLabel="loss L(w)" />
        <path d={curve} className="curve" />
        <line x1={xs(W_BEST)} x2={xs(W_BEST)} y1={PAD} y2={H - PAD} className="best-line" />
        <text x={xs(W_BEST) + 6} y={PAD + 14} className="tick">
          minimum w = 1.786
        </text>
        {visible.slice(1).map((v, i) => {
          const prev = visible[i]
          return (
            <line key={i} x1={xs(prev)} y1={ys(Math.min(loss(prev), Y_MAX))} x2={xs(v)} y2={ys(Math.min(loss(v), Y_MAX))} className="gd-jump" />
          )
        })}
        {visible.map((v, i) => (
          <circle
            key={i}
            cx={xs(v)}
            cy={ys(Math.min(loss(v), Y_MAX))}
            r={i === visible.length - 1 ? 8 : 4}
            className={i === visible.length - 1 ? 'pt pt-marker' : 'pt pt-trail'}
          />
        ))}
      </svg>
    </Playground>
  )
}
