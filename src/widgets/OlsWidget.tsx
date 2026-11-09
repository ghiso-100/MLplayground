import { useRef, useState, type PointerEvent } from 'react'
import { Axes, Playground, scale, svgPoint } from './shared'

const W = 600
const H = 360
const PAD = 40
const x = scale(0, 10, PAD, W - PAD)
const y = scale(0, 10, H - PAD, PAD)

const START = [
  { x: 1.2, y: 2.1 },
  { x: 2.5, y: 2.6 },
  { x: 3.6, y: 4.4 },
  { x: 5.0, y: 4.1 },
  { x: 6.3, y: 6.2 },
  { x: 7.4, y: 6.0 },
  { x: 8.6, y: 8.1 },
]

function fit(pts: { x: number; y: number }[]) {
  const n = pts.length
  if (n < 2) return null
  const mx = pts.reduce((s, p) => s + p.x, 0) / n
  const my = pts.reduce((s, p) => s + p.y, 0) / n
  const sxx = pts.reduce((s, p) => s + (p.x - mx) ** 2, 0)
  if (sxx < 1e-9) return null
  const sxy = pts.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0)
  const b1 = sxy / sxx
  const b0 = my - b1 * mx
  const sse = pts.reduce((s, p) => s + (p.y - (b0 + b1 * p.x)) ** 2, 0)
  return { b0, b1, sse, mse: sse / n }
}

const clamp = (v: number) => Math.min(9.9, Math.max(0.1, v))

export function OlsWidget() {
  const [pts, setPts] = useState(START)
  const [squares, setSquares] = useState(true)
  const [drag, setDrag] = useState<number | null>(null)
  const svg = useRef<SVGSVGElement>(null)
  const model = fit(pts)

  const toData = (e: PointerEvent) => {
    const p = svgPoint(svg.current!, e.clientX, e.clientY)
    return { x: clamp(x.invert(p.x)), y: clamp(y.invert(p.y)) }
  }

  const onBackground = (e: PointerEvent<SVGRectElement>) => {
    if (pts.length >= 30) return
    setPts((ps) => [...ps, toData(e)])
  }

  return (
    <Playground
      title="Least squares playground"
      instructions="Click to add a point, drag points to move them, double-click a point to remove it."
      readout={
        model ? (
          <>
            <span>
              ŷ = <b>{model.b0.toFixed(2)}</b> + <b>{model.b1.toFixed(2)}</b> x
            </span>
            <span>
              sum of squared errors <b>{model.sse.toFixed(2)}</b>
            </span>
            <span>
              MSE <b>{model.mse.toFixed(3)}</b>
            </span>
          </>
        ) : (
          <span>Add at least two points with different x values.</span>
        )
      }
      controls={
        <>
          <label className="check">
            <input type="checkbox" checked={squares} onChange={(e) => setSquares(e.target.checked)} />
            Show squared errors
          </label>
          <button className="btn-quiet" onClick={() => setPts(START)}>
            Reset points
          </button>
          <button className="btn-quiet" onClick={() => setPts([])}>
            Clear
          </button>
        </>
      }
    >
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="plot"
        onPointerMove={(e) => {
          if (drag === null) return
          const p = toData(e)
          setPts((ps) => ps.map((q, i) => (i === drag ? p : q)))
        }}
        onPointerUp={() => setDrag(null)}
        onPointerLeave={() => setDrag(null)}
      >
        <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} className="hit" onPointerDown={onBackground} />
        <Axes width={W} height={H} pad={PAD} xTicks={[0, 2, 4, 6, 8, 10]} yTicks={[0, 2, 4, 6, 8, 10]} x={x} y={y} xLabel="size" yLabel="price" />

        {model &&
          squares &&
          pts.map((p, i) => {
            const yHat = model.b0 + model.b1 * p.x
            const side = Math.abs(y(p.y) - y(yHat))
            const top = Math.min(y(p.y), y(yHat))
            return <rect key={`sq${i}`} x={x(p.x)} y={top} width={side} height={side} className="sq" />
          })}

        {model && (
          <line
            x1={x(0)}
            y1={y(model.b0)}
            x2={x(10)}
            y2={y(model.b0 + model.b1 * 10)}
            className="fit-line"
            clipPath="url(#ols-clip)"
          />
        )}
        <clipPath id="ols-clip">
          <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} />
        </clipPath>

        {model &&
          pts.map((p, i) => (
            <line key={`r${i}`} x1={x(p.x)} x2={x(p.x)} y1={y(p.y)} y2={y(model.b0 + model.b1 * p.x)} className="residual" />
          ))}

        {pts.map((p, i) => (
          <circle
            key={i}
            cx={x(p.x)}
            cy={y(p.y)}
            r={drag === i ? 9 : 7}
            className="pt pt-a draggable"
            onPointerDown={(e) => {
              e.stopPropagation()
              ;(e.target as Element).setPointerCapture?.(e.pointerId)
              setDrag(i)
            }}
            onDoubleClick={() => setPts((ps) => ps.filter((_, j) => j !== i))}
          />
        ))}
      </svg>
    </Playground>
  )
}
