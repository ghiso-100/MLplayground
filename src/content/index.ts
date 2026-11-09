import type { Phase, Topic } from '../types'
import { phase0 } from './phase0'
import { phase1 } from './phase1'
import { phase2 } from './phase2'
import { phase3 } from './phase3'
import { phase4 } from './phase4'

export const PHASES: Phase[] = [phase0, phase1, phase2, phase3, phase4]

export interface FlatTopic {
  topic: Topic
  phase: Phase
  phaseIndex: number
  index: number
}

export const ALL_TOPICS: FlatTopic[] = PHASES.flatMap((phase, phaseIndex) =>
  phase.topics.map((topic) => ({ topic, phase, phaseIndex, index: 0 })),
).map((t, index) => ({ ...t, index }))

export const TOPIC_BY_ID = new Map(ALL_TOPICS.map((t) => [t.topic.id, t]))
