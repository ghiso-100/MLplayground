import { useCallback, useEffect, useState } from 'react'
import type { StepId } from '../types'

const KEY = 'mlplayground:v1'

export interface ProgressState {
  /** Completed steps per topic id. */
  done: Record<string, StepId[]>
  /** Notes per "topicId:stepId". */
  notes: Record<string, string>
  /** Edited code per "topicId:stepId". */
  code: Record<string, string>
  /** Where the learner was last. */
  last: { topic: string; step: StepId } | null
}

const EMPTY: ProgressState = { done: {}, notes: {}, code: {}, last: null }

function load(): ProgressState {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return EMPTY
    const parsed = JSON.parse(raw) as Partial<ProgressState>
    return { ...EMPTY, ...parsed }
  } catch {
    return EMPTY
  }
}

function save(state: ProgressState) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    /* storage unavailable (private mode): progress lives in memory only */
  }
}

export const stepKey = (topic: string, step: StepId) => `${topic}:${step}`

export function useProgress() {
  const [state, setState] = useState<ProgressState>(load)

  useEffect(() => save(state), [state])

  const isDone = useCallback(
    (topic: string, step: StepId) => state.done[topic]?.includes(step) ?? false,
    [state.done],
  )

  const toggleDone = useCallback((topic: string, step: StepId) => {
    setState((s) => {
      const current = s.done[topic] ?? []
      const next = current.includes(step) ? current.filter((x) => x !== step) : [...current, step]
      return { ...s, done: { ...s.done, [topic]: next } }
    })
  }, [])

  const setNote = useCallback((topic: string, step: StepId, text: string) => {
    setState((s) => ({ ...s, notes: { ...s.notes, [stepKey(topic, step)]: text } }))
  }, [])

  const setCode = useCallback((topic: string, step: StepId, text: string | null) => {
    setState((s) => {
      const code = { ...s.code }
      if (text === null) delete code[stepKey(topic, step)]
      else code[stepKey(topic, step)] = text
      return { ...s, code }
    })
  }, [])

  const setLast = useCallback((topic: string, step: StepId) => {
    setState((s) =>
      s.last?.topic === topic && s.last.step === step ? s : { ...s, last: { topic, step } },
    )
  }, [])

  const replaceAll = useCallback((next: ProgressState) => setState({ ...EMPTY, ...next }), [])

  return { state, isDone, toggleDone, setNote, setCode, setLast, replaceAll }
}

export type Progress = ReturnType<typeof useProgress>
