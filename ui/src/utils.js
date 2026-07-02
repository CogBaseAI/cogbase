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
