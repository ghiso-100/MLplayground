import { STEPS, type StepId } from '../types'

interface Props {
  current: StepId
  isDone: (step: StepId) => boolean
  onSelect: (step: StepId) => void
}

const X = [110, 305, 500, 695, 890]
const TOP = 58
const BOTTOM = 156

// A stadium-shaped loop: the five stations sit on the top straight,
// the bottom straight is the way back to step 1 for the next topic.
const TRACK = `M ${X[0]} ${TOP} L ${X[4]} ${TOP} C ${X[4] + 92} ${TOP}, ${X[4] + 92} ${BOTTOM}, ${X[4]} ${BOTTOM} L ${X[0]} ${BOTTOM} C ${X[0] - 92} ${BOTTOM}, ${X[0] - 92} ${TOP}, ${X[0]} ${TOP} Z`

export function LoopTrack({ current, isDone, onSelect }: Props) {
  const currentIndex = STEPS.findIndex((s) => s.id === current)
  const doneCount = STEPS.filter((s) => isDone(s.id)).length

  return (
    <div className="loop">
      <svg viewBox="0 0 1000 184" className="loop-svg" role="group" aria-label="Study loop">
        <path d={TRACK} className="loop-track" />
        <path d={TRACK} className="loop-track-dash" />

        {/* Filled track between consecutive completed stations */}
        {STEPS.slice(0, -1).map((s, i) =>
          isDone(s.id) && isDone(STEPS[i + 1].id) ? (
            <line key={s.id} x1={X[i]} y1={TOP} x2={X[i + 1]} y2={TOP} className="loop-track-done" />
          ) : null,
        )}

        <text x={500} y={BOTTOM + 6} className="loop-return" textAnchor="middle">
          {doneCount === 5 ? 'Loop complete. On to the next topic' : 'then loop again with the next topic'}
        </text>

        {/* The marker that travels to the active station */}
        <g className="loop-cursor" style={{ transform: `translateX(${X[currentIndex]}px)` }}>
          <circle cx={0} cy={TOP} r={34} className="loop-cursor-halo" />
        </g>

        {STEPS.map((s, i) => {
          const done = isDone(s.id)
          const active = s.id === current
          return (
            <g
              key={s.id}
              className={`station ${done ? 'is-done' : ''} ${active ? 'is-active' : ''}`}
              role="button"
              tabIndex={0}
              aria-pressed={active}
              aria-label={`Step ${i + 1}: ${s.name}${done ? ', done' : ''}`}
              onClick={() => onSelect(s.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelect(s.id)
                }
              }}
            >
              <circle cx={X[i]} cy={TOP} r={44} className="station-hit" />
              <circle cx={X[i]} cy={TOP} r={22} className="station-dot" />
              {done ? (
                <path
                  d={`M ${X[i] - 9} ${TOP} l 6 6 l 12 -13`}
                  className="station-check"
                />
              ) : (
                <text x={X[i]} y={TOP + 8} textAnchor="middle" className="station-num">
                  {i + 1}
                </text>
              )}
              <text x={X[i]} y={TOP + 56} textAnchor="middle" className="station-name">
                {s.name}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
