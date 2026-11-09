import katex from 'katex'
import 'katex/dist/katex.min.css'
import { Fragment, memo, type ReactNode } from 'react'

/**
 * Renders the curriculum's tiny markup:
 *   blank-line separated blocks, "- " and "1. " lists, | tables |,
 *   $$display math$$, $inline math$, **bold** and «code».
 */

const mathCache = new Map<string, string>()

function tex(src: string, display: boolean): string {
  const key = (display ? 'D' : 'I') + src
  let html = mathCache.get(key)
  if (!html) {
    html = katex.renderToString(src, { displayMode: display, throwOnError: false, strict: false })
    mathCache.set(key, html)
  }
  return html
}

function Inline({ text }: { text: string }) {
  const parts = text.split(/(\$[^$]+\$|\*\*[^*]+\*\*|«[^»]+»)/g)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('$') && part.endsWith('$') && part.length > 2) {
          return <span key={i} dangerouslySetInnerHTML={{ __html: tex(part.slice(1, -1), false) }} />
        }
        if (part.startsWith('**') && part.endsWith('**')) {
          return (
            <strong key={i}>
              <Inline text={part.slice(2, -2)} />
            </strong>
          )
        }
        if (part.startsWith('«') && part.endsWith('»')) {
          return <code key={i}>{part.slice(1, -1)}</code>
        }
        return <Fragment key={i}>{part}</Fragment>
      })}
    </>
  )
}

const LIST_RE = /^(- |\d+\. )/

function renderBlock(block: string, i: number): ReactNode {
  const trimmed = block.trim()

  if (trimmed.startsWith('$$') && trimmed.endsWith('$$')) {
    return (
      <div
        key={i}
        className="math-block"
        dangerouslySetInnerHTML={{ __html: tex(trimmed.slice(2, -2).trim(), true) }}
      />
    )
  }

  const lines = trimmed.split('\n')

  if (lines.every((l) => l.trim().startsWith('|'))) {
    const rows = lines
      .filter((l) => !/^\|\s*-/.test(l.trim()))
      .map((l) =>
        l
          .trim()
          .replace(/^\||\|$/g, '')
          .split('|')
          .map((c) => c.trim()),
      )
    const [head, ...body] = rows
    return (
      <div key={i} className="table-wrap">
        <table>
          <thead>
            <tr>
              {head.map((c, j) => (
                <th key={j}>
                  <Inline text={c} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((r, k) => (
              <tr key={k}>
                {r.map((c, j) => (
                  <td key={j}>
                    <Inline text={c} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  // Leading paragraph lines, then list items (with indented continuation lines).
  const firstList = lines.findIndex((l) => LIST_RE.test(l))
  if (firstList !== -1) {
    const intro = lines.slice(0, firstList).join(' ')
    const items: string[] = []
    for (const line of lines.slice(firstList)) {
      if (LIST_RE.test(line)) items.push(line.replace(LIST_RE, ''))
      else if (items.length) items[items.length - 1] += '\n' + line.trim()
    }
    const ordered = /^\d+\. /.test(lines[firstList])
    const List = ordered ? 'ol' : 'ul'
    return (
      <Fragment key={i}>
        {intro && (
          <p>
            <Inline text={intro} />
          </p>
        )}
        <List>
          {items.map((item, j) => (
            <li key={j}>
              {item.split('\n').map((seg, k) => (
                <Fragment key={k}>
                  {k > 0 && <br />}
                  <Inline text={seg} />
                </Fragment>
              ))}
            </li>
          ))}
        </List>
      </Fragment>
    )
  }

  return (
    <p key={i}>
      <Inline text={lines.join(' ')} />
    </p>
  )
}

export const Rich = memo(function Rich({ text, className }: { text: string; className?: string }) {
  const blocks = text.split(/\n\s*\n/)
  return <div className={`rich ${className ?? ''}`}>{blocks.map(renderBlock)}</div>
})
