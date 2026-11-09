import type { ReactNode } from 'react'

/** Small seeded random generator so the playgrounds look the same on every visit. */
export function mulberry32(seed: number) {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-9)
  const v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/** Linear scale helper. */
export function scale(d0: number, d1: number, r0: number, r1: number) {
  const f = (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)
  f.invert = (p: number) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0)
  return f
}

/** Convert a pointer event to SVG viewBox coordinates. */
export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number) {
  const pt = svg.createSVGPoint()
  pt.x = clientX
  pt.y = clientY
  const m = svg.getScreenCTM()
  if (!m) return { x: 0, y: 0 }
  const p = pt.matrixTransform(m.inverse())
  return { x: p.x, y: p.y }
}

export function Playground({
  title,
  instructions,
  controls,
  readout,
  children,
}: {
  title: string
  instructions: string
  controls?: ReactNode
  readout?: ReactNode
  children: ReactNode
}) {
  return (
    <figure className="playground">
      <figcaption>
        <strong>{title}</strong>
        <span>{instructions}</span>
      </figcaption>
      <div className="playground-stage">{children}</div>
      {(controls || readout) && (
        <div className="playground-foot">
          {readout && <div className="readout">{readout}</div>}
          {controls && <div className="controls">{controls}</div>}
        </div>
      )}
    </figure>
  )
}

/** Light grid and axes for a plot area. */
export function Axes({
  width,
  height,
  pad,
  xTicks,
  yTicks,
  x,
  y,
  xLabel,
  yLabel,
}: {
  width: number
  height: number
  pad: number
  xTicks: number[]
  yTicks: number[]
  x: (v: number) => number
  y: (v: number) => number
  xLabel?: string
  yLabel?: string
}) {
  return (
    <g className="axes">
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          <line x1={x(t)} x2={x(t)} y1={pad} y2={height - pad} className="grid" />
          <text x={x(t)} y={height - pad + 16} textAnchor="middle" className="tick">
            {t}
          </text>
        </g>
      ))}
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={pad} x2={width - pad} y1={y(t)} y2={y(t)} className="grid" />
          <text x={pad - 8} y={y(t) + 4} textAnchor="end" className="tick">
            {t}
          </text>
        </g>
      ))}
      {xLabel && (
        <text x={width - pad} y={height - pad - 6} textAnchor="end" className="axis-label">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text x={pad + 6} y={pad + 12} className="axis-label">
          {yLabel}
        </text>
      )}
    </g>
  )
}
