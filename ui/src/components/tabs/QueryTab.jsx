import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { streamSSE, copyText } from '../../utils'

export default function QueryTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const { t } = useT()
  const [msgs, setMsgs] = useState([{ role: 'sys', text: t('query.intro') }])
  const [input, setInput] = useState('')
  const [querying, setQuerying] = useState(false)
  const [chunks, setChunks] = useState([])
  const [structuredRecords, setStructuredRecords] = useState([])
  const msgsRef = useRef(null)
  const sessionIdRef = useRef(null)
  const textareaRef = useRef(null)
  const hasApp = !!currentApp
  const prevAppRef = useRef(currentApp)

  useEffect(() => {
    if (currentApp !== prevAppRef.current) {
      const prevApp = prevAppRef.current
      prevAppRef.current = currentApp
      closeSession(prevApp)
      // Drop the previous app's retrieved references so they don't linger.
      setChunks([])
      setStructuredRecords([])
      if (currentApp) {
        setMsgs(prev => [...prev, { role: 'sys', text: t('query.connected', { app: currentApp }) }])
      }
    }
  }, [currentApp])

  // Close the session bound to `appName`, fire-and-forget, and clear the local handle.
  function closeSession(appName) {
    const sid = sessionIdRef.current
    sessionIdRef.current = null
    if (!sid || !appName) return
    fetch(`${apiUrl}/applications/${encodeURIComponent(appName)}/sessions/${encodeURIComponent(sid)}/close`, {
      method: 'POST',
    }).catch(() => {})
  }

  // Open a session for the current app if one isn't already open; returns its id.
  async function ensureSession() {
    if (sessionIdRef.current) return sessionIdRef.current
    const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    if (!resp.ok) throw new Error(`failed to start session: HTTP ${resp.status}`)
    const data = await resp.json()
    sessionIdRef.current = data.session_id
    return data.session_id
  }

  function scrollMsgs() { if (msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight }

  function autoResize(el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 110) + 'px'
  }

  async function sendQuery() {
    if (querying || !currentApp) return
    const text = input.trim()
    if (!text) return
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = ''
    setQuerying(true)

    const userMsg = { role: 'user', text }
    const botMsg = { role: 'bot', text: t('query.thinking'), thinking: true }
    setMsgs(prev => [...prev, userMsg, botMsg])
    setTimeout(scrollMsgs, 0)

    let answer = ''
    let started = false
    try {
      const sessionId = await ensureSession()
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, session_id: sessionId }),
      })
      if (!resp.ok) {
        const errText = await resp.text()
        setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('query.httpErr', { status: resp.status, msg: errText }), error: true }])
        return
      }

      for await (const d of streamSSE(resp)) {
        if (d.token) {
          if (!started) { started = true }
          answer += d.token
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer }])
          setTimeout(scrollMsgs, 0)
        } else if (d.result) {
          if (!started) started = true
          if (d.result.passthrough && d.result.structured_records) {
            answer = JSON.stringify(d.result.structured_records, null, 2)
            setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer, mono: true }])
          } else if (d.result.answer) {
            answer = d.result.answer
            setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer }])
          }
          setChunks(d.result.chunks || [])
          setStructuredRecords(d.result.structured_records || [])
        } else if (d.error) {
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('common.error', { msg: d.error }), error: true }])
        }
      }

      if (!started) setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('query.noResponse'), muted: true }])
    } catch (e) {
      setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('common.networkError', { msg: e.message }), error: true }])
    } finally {
      setQuerying(false)
    }
  }

  // Refresh: close the current session (triggering distillation) and start fresh.
  // A new session is opened lazily on the next question.
  function refreshSession() {
    closeSession(currentApp)
    setChunks([])
    setStructuredRecords([])
    setMsgs([{ role: 'sys', text: t('query.refreshed') + (currentApp ? ' ' + t('query.connected', { app: currentApp }) : '') }])
  }

  const totalRefs = chunks.length + structuredRecords.length

  return (
    <>
      {!hasApp && <div className="warn-bar show">{t('common.noAppWarn')}</div>}
      <div className="chat-layout">
        <div className="chat-col">
          <div className="msgs" ref={msgsRef}>
            {msgs.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                {m.role === 'user' && <div className="msg-who">{t('query.you')}</div>}
                {m.role === 'bot' && <div className="msg-who">{t('query.bot')}</div>}
                <div className="msg-body" style={{
                  color: m.error ? 'var(--red)' : m.thinking || m.muted ? 'var(--muted)' : undefined,
                  fontFamily: m.mono ? 'monospace' : undefined,
                  fontSize: m.mono ? 11 : undefined,
                }}>
                  {m.role === 'bot' && !m.mono && !m.thinking
                    ? <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents(t)}>{m.text}</ReactMarkdown></div>
                    : m.text}
                </div>
                {(m.role === 'user' || m.role === 'bot') && !m.thinking && !m.error && !m.muted && (
                  <div className="msg-actions">
                    <CopyButton text={m.text} title={t('query.copy')} copiedTitle={t('query.copied')} />
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="chat-input">
            <textarea
              ref={textareaRef}
              value={input}
              placeholder={t('query.placeholder')}
              rows={1}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery() } }}
              onChange={e => { setInput(e.target.value); autoResize(e.target) }}
              disabled={querying || !hasApp}
            />
            <button className="btn btn-ghost" title={t('query.refreshSession')} onClick={refreshSession}>↺</button>
            <button className="btn btn-primary" disabled={querying || !hasApp} onClick={sendQuery}>{t('common.send')}</button>
          </div>
        </div>
        <div className="chat-aside">
          <div className="aside-hd">
            <h3>{t('query.references')}</h3>
            <span style={{ fontSize: 11, color: 'var(--muted)' }}>{totalRefs ? (totalRefs !== 1 ? t('query.refs', { n: totalRefs }) : t('query.ref', { n: totalRefs })) : '—'}</span>
          </div>
          <div className="aside-body">
            {!totalRefs && <div style={{ padding: '30px 10px', textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>{t('query.refsEmpty')}</div>}
            {structuredRecords.length > 0 && (
              <>
                <div className="ref-section-hd">{t('query.structuredRecords', { n: structuredRecords.length })}</div>
                {structuredRecords.map((rec, i) => (
                  <div key={i} className="ref-card"><pre className="ref-code">{JSON.stringify(rec, null, 2)}</pre></div>
                ))}
              </>
            )}
            {chunks.length > 0 && (
              <>
                <div className="ref-section-hd">{t('query.passages', { n: chunks.length })}</div>
                {chunks.map((ch, i) => <RefChunk key={i} chunk={ch} t={t} />)}
              </>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

// Clipboard / check glyphs, sized to the current font.
function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  )
}
function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  )
}

// An icon copy button that flips to a check on success. `text` may be a string
// or a function returning the text to copy (deferred so the DOM can be read at
// click time). Always visible, styled like ChatGPT/Claude.
function CopyButton({ text, title, copiedTitle, className = 'icon-copy-btn' }) {
  const [copied, setCopied] = useState(false)
  async function copy(e) {
    e.stopPropagation()
    const value = typeof text === 'function' ? text() : text
    if (await copyText(value)) {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }
  }
  return (
    <button
      type="button"
      className={`${className}${copied ? ' copied' : ''}`}
      onClick={copy}
      title={copied ? copiedTitle : title}
      aria-label={copied ? copiedTitle : title}
    >
      {copied ? <CheckIcon /> : <CopyIcon />}
    </button>
  )
}

// Serialize a rendered <table> DOM node to tab-separated rows, so it pastes
// cleanly into spreadsheets.
function tableToTSV(table) {
  if (!table) return ''
  return [...table.querySelectorAll('tr')]
    .map(tr => [...tr.querySelectorAll('th,td')].map(c => c.textContent.trim()).join('\t'))
    .join('\n')
}

// Wrap each markdown table with a floating "Copy table" button.
function CopyTableWrapper({ children, t }) {
  const ref = useRef(null)
  return (
    <div className="md-table-wrap">
      <CopyButton
        className="md-table-copy-btn"
        text={() => tableToTSV(ref.current)}
        title={t('query.copyTable')}
        copiedTitle={t('query.copied')}
      />
      <table ref={ref}>{children}</table>
    </div>
  )
}

// ReactMarkdown component overrides. Memoized per-`t` so the object is stable.
function mdComponents(t) {
  return {
    table: ({ children }) => <CopyTableWrapper t={t}>{children}</CopyTableWrapper>,
  }
}

function RefChunk({ chunk, t }) {
  const [expanded, setExpanded] = useState(false)
  const score = chunk.metadata?.score != null ? Number(chunk.metadata.score).toFixed(3) : null
  const collection = chunk.metadata?.collection || null
  const label = chunk.chunk_id || chunk.doc_id
  const long = chunk.text.length > 300
  return (
    <div className="ref-card" onClick={() => long && setExpanded(v => !v)}>
      <div className="ref-meta">
        <span className="ref-docid">{label}</span>
        {collection && <span className="ref-coll">{collection}</span>}
        {score !== null && <span className="ref-score">{score}</span>}
      </div>
      <div className={`ref-text${long && !expanded ? ' collapsed' : ''}`}>{chunk.text}</div>
      {long && <div className="ref-expand-hint">{expanded ? t('query.collapse') : t('query.expand')}</div>}
    </div>
  )
}
