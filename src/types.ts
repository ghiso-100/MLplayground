/**
 * Content model for the curriculum.
 *
 * Text fields use a tiny markup language (see components/Rich.tsx):
 *   paragraphs separated by blank lines, "- " bullet lists,
 *   | tables |, **bold**, «code», $inline math$ and $$display math$$ (KaTeX).
 */

export type StepId = 'intuition' | 'math' | 'paper' | 'code' | 'real'

export type WidgetId = 'ols' | 'gd' | 'knn' | 'kmeans'

export interface CodeStep {
  /** What to do, in markup. */
  task: string
  /** Python starter code. */
  starter: string
  /** True when the code needs a network download and must run locally. */
  local?: boolean
}

export interface Topic {
  id: string
  title: string
  /** One-line summary shown under the title. */
  summary: string
  /** Link into the scikit-learn user guide (or another reference for math). */
  guide: { label: string; url: string }
  /** Optional interactive visual shown in the intuition step. */
  widget?: WidgetId
  intuition: string
  math: string
  paper: { task: string; solution: string }
  code: CodeStep
  real: CodeStep
}

export interface Phase {
  id: string
  title: string
  /** Short description of what this phase is for. */
  goal: string
  weeks: string
  topics: Topic[]
}

export interface StepMeta {
  id: StepId
  name: string
  verb: string
  hint: string
}

export const STEPS: StepMeta[] = [
  {
    id: 'intuition',
    name: 'Intuition',
    verb: 'Get the idea',
    hint: 'What problem does this solve? Say the core idea in one sentence before touching formulas.',
  },
  {
    id: 'math',
    name: 'Math',
    verb: 'Read the formula',
    hint: 'Write the formula by hand and explain every symbol in your own words.',
  },
  {
    id: 'paper',
    name: 'Paper',
    verb: 'Calculate by hand',
    hint: 'Work a tiny example with 3 to 5 numbers on paper. This is where understanding happens.',
  },
  {
    id: 'code',
    name: 'From scratch',
    verb: 'Build it in NumPy',
    hint: 'Implement a simple version yourself, then check it against scikit-learn.',
  },
  {
    id: 'real',
    name: 'Real data',
    verb: 'Apply and interpret',
    hint: 'Use a real dataset and explain what the result means in the real world.',
  },
]
