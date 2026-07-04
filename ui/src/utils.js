// Copy text to the clipboard, returning true on success. Prefers the async
// Clipboard API, but that requires a secure context (HTTPS or localhost) — when
// the app is served over plain HTTP to a remote host, navigator.clipboard is
// undefined (or writeText rejects), so we fall back to a hidden <textarea> +
// execCommand('copy'), which still works in insecure contexts.
export async function copyText(value) {
  const text = String(value ?? '')
  if (navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // fall through to the legacy path
    }
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    // Keep it out of view and non-interactive, but still selectable.
    ta.style.position = 'fixed'
    ta.style.top = '-9999px'
    ta.setAttribute('readonly', '')
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

// A skill that produces a downloadable file reports its path through the query
// runner's save_artifact tool as
//   /applications/<app_name>/documents/<artifact_id>/download
// The runner only knows the app's internal id, so it emits the literal
// `<app_name>` placeholder and a path relative to the API — neither of which is
// clickable in the chat as-is (a relative path targets the UI origin, and the
// placeholder isn't a real name). Rewrite every such occurrence into an
// absolute URL against the API origin with the real app name, so remark-gfm
// autolinks it and the click hits the download endpoint. Leaves everything else
// untouched; idempotent on already-absolute URLs.
const ARTIFACT_PATH_RE =
  /(?:https?:\/\/[^/\s)]+)?\/applications\/([^/\s)]+)\/documents\/([^/\s)]+)\/download/g

// The LLM often reports that path inside a markdown code span (backticks). A
// code span never renders as a link, so these are lifted into a real labeled
// link instead. The character class excludes the backtick so the span boundary
// is respected.
const ARTIFACT_CODE_SPAN_RE =
  /`\s*(?:https?:\/\/[^/\s`]+)?\/applications\/([^/\s`]+)\/documents\/([^/\s`]+)\/download\s*`/g

export function resolveArtifactLinks(text, apiUrl, appName) {
  if (!text) return text
  const base = String(apiUrl || '').replace(/\/$/, '')
  const toUrl = (name, id) => {
    const app = name === '<app_name>' ? appName : decodeURIComponent(name)
    return `${base}/applications/${encodeURIComponent(app || name)}/documents/${id}/download`
  }
  return String(text)
    // Backtick-wrapped path -> a labeled markdown link (code spans don't click).
    .replace(ARTIFACT_CODE_SPAN_RE, (_m, name, id) => `[${artifactLabel(id)}](${toUrl(name, id)})`)
    // Bare or already-linked occurrences -> resolve the URL in place; a bare URL
    // then autolinks, and the `a` override wires up the download click.
    .replace(ARTIFACT_PATH_RE, (_m, name, id) => toUrl(name, id))
}

// Human-facing filename for a generated artifact id:
// `saas-001-amended__06467960.docx` -> `saas-001-amended.docx` (drop the short
// hex hash save_artifact appends to disambiguate).
export function artifactLabel(id) {
  return decodeURIComponent(String(id)).replace(/__[0-9a-f]{6,}(?=\.[^.]+$|$)/i, '')
}

export function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

export function fmtBytes(b) {
  if (b < 1024) return b + ' B'
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB'
  return (b / 1048576).toFixed(1) + ' MB'
}

// Compact relative time ("just now", "5m", "3h", "2d") from an ISO timestamp,
// falling back to a locale date for anything older than a week.
export function fmtRelTime(iso) {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const sec = Math.max(0, Math.floor((Date.now() - then) / 1000))
  if (sec < 45) return 'just now'
  const min = Math.floor(sec / 60)
  if (min < 60) return min + 'm'
  const hr = Math.floor(min / 60)
  if (hr < 24) return hr + 'h'
  const day = Math.floor(hr / 24)
  if (day < 7) return day + 'd'
  return new Date(then).toLocaleDateString()
}

export function previewText(text, maxLines = 6, maxChars = 320) {
  const lines = String(text || '').trim().split(/\r?\n/)
  let out = lines.slice(0, maxLines).join('\n')
  if (lines.length > maxLines) out += '\n…'
  if (out.length > maxChars) out = out.slice(0, maxChars) + '…'
  return out
}

export function metaText(meta) {
  const entries = Object.entries(meta || {})
  if (!entries.length) return 'no metadata'
  return entries.map(([k, v]) => `${k}: ${v}`).join(' • ')
}

export function schemaTypeStr(def) {
  if (def.$ref) return def.$ref.split('/').pop()
  if (def.anyOf) {
    const parts = def.anyOf
      .map(t => (t.$ref ? t.$ref.split('/').pop() : t.type || null))
      .filter(Boolean)
      .filter(t => t !== 'null')
    return parts.join('|') || '?'
  }
  if (def.type === 'array') {
    const items = def.items || {}
    const itemType = items.$ref ? items.$ref.split('/').pop() : items.type || 'object'
    return 'list[' + itemType + ']'
  }
  return def.type || '?'
}

export function simplifyExtractionSchemas(yamlText) {
  return yamlText.replace(
    /^([ \t]*)(extraction_schema:|schema:)([ \t]*)('(?:[^']|'')*'|"(?:[^"\\]|\\.)*")/gm,
    function (match, indent, key, _sp, quoted) {
      let inner = quoted.slice(1, -1)
      if (quoted[0] === "'") inner = inner.replace(/\n[ \t]*/g, ' ').replace(/''/g, "'")
      try {
        const schema = JSON.parse(inner)
        const props = schema.properties || {}
        const entries = Object.entries(props)
        if (!entries.length) return match
        const fieldLines = entries
          .map(([name, def]) => {
            const t = schemaTypeStr(def)
            const d = def.description
              ? ' — ' + (def.description.length > 60 ? def.description.slice(0, 60) + '…' : def.description)
              : ''
            return indent + '#   ' + name + ': ' + t + d
          })
          .join('\n')
        return indent + key + '  # ' + entries.length + ' fields\n' + fieldLines
      } catch (_) {
        return match
      }
    }
  )
}

export async function* streamSSE(response) {
  const reader = response.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop()
    for (const line of lines) {
      if (!line.startsWith('data:')) continue
      const payload = line.slice(5).trim()
      if (payload === '[DONE]') return
      try { yield JSON.parse(payload) } catch {}
    }
  }
}

export async function waitForTasks(apiUrl, appName, taskIds, { pollInterval = 2000, timeout = 300000, onProgress } = {}) {
  const deadline = Date.now() + timeout
  const pending = new Set(taskIds)
  const results = {}
  while (pending.size > 0) {
    if (Date.now() > deadline) {
      for (const tid of pending) results[tid] = { task_id: tid, doc_id: null, status: 'failed', error: 'timeout' }
      break
    }
    await new Promise(r => setTimeout(r, pollInterval))
    for (const tid of [...pending]) {
      try {
        const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(appName)}/tasks/${encodeURIComponent(tid)}`)
        if (resp.ok) {
          const task = await resp.json()
          if (task.status === 'done' || task.status === 'failed') {
            results[tid] = task
            pending.delete(tid)
            if (onProgress) onProgress(taskIds.length - pending.size, taskIds.length)
          }
        }
      } catch {}
    }
  }
  return taskIds.map(tid => results[tid])
}
