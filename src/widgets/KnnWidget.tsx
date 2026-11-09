import { useMemo, useRef, useState, type PointerEvent } from 'react'
import { Playground, gaussian, mulberry32, scale, svgPoint } from './shared'

const W = 600
const H = 360
const PAD = 20
const x = scale(0, 10, PAD, W - PAD)
const y = scale(0, 6, H - PAD, PAD)

type P = { x: number; y: number; c: 0 | 1 }

function makeData(): P[] {
  const r = mulberry32(7)
  const pts: P[] = []
  const blobs: [number, number, 0 | 1][] = [
    [2.8, 2.2, 0],
    [4.6, 4.3, 0],
    [6.6, 2.0, 1],
    [7.6, 4.4, 1],
  ]
  for (const [cx, cy, c] of blobs)
    for (let i = 0; i < 14; i++)
      pts.push({ x: cx + gaussian(r) * 1.0, y: cy + gaussian(r) * 0.75, c })
  return pts.filter((p) => p.x > 0.2 && p.x < 9.8 && p.y > 0.2 && p.y < 5.8)
}

function vote(data: P[], qx: number, qy: number, k: number) {
  const sorted = data
    .map((p, i) => ({ i, d: Math.hypot(p.x - qx, p.y - qy) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k)
  const ones = sorted.filter((s) => data[s.i].c === 1).length
  return { nearest: sorted, ones, zeros: k - ones, pred: (ones > k / 2 ? 1 : 0) as 0 | 1 }
}

const COLS = 40
const ROWS = 24

export function KnnWidget() {
  const data = useMemo(makeData, [])
  const [k, setK] = useState(5)
  const [q, setQ] = useState({ x: 5.4, y: 3.1 })
  const [regions, setRegions] = useState(true)
  const [dragging, setDragging] = useState(false)
  const svg = useRef<SVGSVGElement>(null)

  const result = vote(data, q.x, q.y, k)
  const radius = result.nearest[result.nearest.length - 1].d

  const grid = useMemo(() => {
    if (!regions) return []
    const cells: { cx: number; cy: number; c: 0 | 1 }[] = []
    for (let i = 0; i < COLS; i++)
      for (let j = 0; j < ROWS; j++) {
        const cx = ((i + 0.5) / COLS) * 10
        const cy = ((j + 0.5) / ROWS) * 6
        cells.push({ cx, cy, c: vote(data, cx, cy, k).pred })
      }
    return cells
  }, [data, k, regions])

  const move = (e: PointerEvent) => {
    const p = svgPoint(svg.current!, e.clientX, e.clientY)
    setQ({ x: Math.min(9.9, Math.max(0.1, x.invert(p.x))), y: Math.min(5.9, Math.max(0.1, y.invert(p.y))) })
  }

  const cw = (W - 2 * PAD) / COLS
  const ch = (H - 2 * PAD) / ROWS

  return (
    <Playground
      title="Nearest neighbors playground"
      instructions="Drag the square (or click anywhere) to move the new point. The k nearest neighbors vote."
      readout={
        <>
          <span>
            votes: <b className="c-a">{result.zeros} teal</b> vs <b className="c-b">{result.ones} blue</b>
          </span>
          <span>
            prediction <b className={result.pred ? 'c-b' : 'c-a'}>{result.pred ? 'blue' : 'teal'}</b>
          </span>
        </>
      }
      controls={
        <>
          <label className="slider">
            k <b>{k}</b>
            <input type="range" min={1} max={25} step={2} value={k} onChange={(e) => setK(Number(e.target.value))} />
          </label>
          <label className="check">
            <input type="checkbox" checked={regions} onChange={(e) => setRegions(e.target.checked)} />
            Show decision regions
          </label>
        </>
      }
    >
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className="plot"
        onPointerDown={(e) => {
          move(e)
          setDragging(true)
          svg.current?.setPointerCapture?.(e.pointerId)
        }}
        onPointerMove={(e) => dragging && move(e)}
        onPointerUp={() => setDragging(false)}
      >
        <rect x={PAD} y={PAD} width={W - 2 * PAD} height={H - 2 * PAD} className="frame" />
        <g className="regions" shapeRendering="crispEdges">
          {grid.map((g, i) => (
            <rect key={i} x={x(g.cx) - cw / 2} y={y(g.cy) - ch / 2} width={cw + 0.6} height={ch + 0.6} className={g.c ? 'region-b' : 'region-a'} />
          ))}
        </g>
        <circle cx={x(q.x)} cy={y(q.y)} r={radius * (x(1) - x(0))} className="knn-radius" />
        {result.nearest.map((n) => (
          <line key={n.i} x1={x(q.x)} y1={y(q.y)} x2={x(data[n.i].x)} y2={y(data[n.i].y)} className="knn-link" />
        ))}
        {data.map((p, i) => (
          <circle key={i} cx={x(p.x)} cy={y(p.y)} r={6} className={`pt ${p.c ? 'pt-b' : 'pt-a'}`} />
        ))}
        <rect x={x(q.x) - 9} y={y(q.y) - 9} width={18} height={18} rx={3} className={`query ${result.pred ? 'pt-b' : 'pt-a'}`} />
      </svg>
    </Playground>
  )
}
