import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { streamSSE, copyText, fmtRelTime, resolveArtifactLinks } from '../../utils'

export default function QueryTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const { t } = useT()
  const [msgs, setMsgs] = useState([{ role: 'sys', text: t('query.intro') }])
  const [input, setInput] = useState('')
  const [querying, setQuerying] = useState(false)
  // Index into `msgs` of the answer whose references fill the pane. Each bot
  // answer carries its own `refs`, so clicking a past answer re-points the pane
  // at that turn's evidence; -1 shows the empty state.
  const [selectedRefIdx, setSelectedRefIdx] = useState(-1)
  const [sessions, setSessions] = useState([])
  const [activeSid, setActiveSid] = useState(null)
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
      // Drop the previous app's retrieved references + session pointer so they
      // don't linger across the switch.
      setSelectedRefIdx(-1)
      setActiveSid(null)
      if (currentApp) {
        setMsgs(prev => [...prev, { role: 'sys', text: t('query.connected', { app: currentApp }) }])
      }
    }
  }, [currentApp])

  // Load the app's chat history whenever the selected app changes (including
  // mount). Kept separate from the switch-cleanup effect so it also runs on the
  // initial render, when prevAppRef already equals currentApp.
  useEffect(() => {
    if (currentApp) loadSessions()
    else setSessions([])
  }, [currentApp])

  // Fetch the session list from the index. Best-effort: a failure just leaves
  // the sidebar empty rather than interrupting the chat.
  async function loadSessions() {
    if (!currentApp) return
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/sessions`)
      if (!resp.ok) return
      const data = await resp.json()
      setSessions(data.sessions || [])
    } catch {}
  }

  // Open a past session: load its transcript and make it the active session so
  // the next message resumes it (the server resumes an existing session_id).
  async function openSession(sid) {
    if (querying || sid === sessionIdRef.current) return
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/sessions/${encodeURIComponent(sid)}`)
      if (!resp.ok) return
      const data = await resp.json()
      const rawMsgs = data.messages || []
      // Carry each assistant turn's references onto its message so clicking any
      // past answer re-points the pane at that turn's evidence.
      const loaded = rawMsgs.map(m => ({
        role: m.role === 'assistant' ? 'bot' : 'user',
        text: m.content,
        ...(m.role === 'assistant' ? { refs: m.references || {} } : {}),
      }))
      sessionIdRef.current = sid
      setActiveSid(sid)
      // Default the pane to the latest answer, matching a live turn's behavior.
      let lastBot = -1
      for (let i = loaded.length - 1; i >= 0; i--) {
        if (loaded[i].role === 'bot') { lastBot = i; break }
      }
      setSelectedRefIdx(lastBot)
      setMsgs(loaded.length ? loaded : [{ role: 'sys', text: t('query.emptySession') }])
      setTimeout(scrollMsgs, 0)
    } catch {}
  }

  // Permanently delete a past session (drops its episodic log + index row). If it
  // was the active chat, reset the view to a fresh, unopened state.
  async function deleteSession(sid, e) {
    e.stopPropagation()
    if (querying || !currentApp) return
    if (!window.confirm(t('query.confirmDeleteChat'))) return
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/sessions/${encodeURIComponent(sid)}`, {
        method: 'DELETE',
      })
      if (!resp.ok) return
    } catch { return }
    if (sid === sessionIdRef.current) {
      sessionIdRef.current = null
      setActiveSid(null)
      setSelectedRefIdx(-1)
      setMsgs([{ role: 'sys', text: t('query.refreshed') + (currentApp ? ' ' + t('query.connected', { app: currentApp }) : '') }])
    }
    loadSessions()
  }

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
    // The bot answer stays at this fixed index for the whole turn (token updates
    // replace the last message in place), so the final result can attach its
    // references there and select it.
    let botIdx = -1
    setMsgs(prev => { botIdx = prev.length + 1; return [...prev, userMsg, botMsg] })
    setTimeout(scrollMsgs, 0)

    let answer = ''
    let started = false
    try {
      const sessionId = await ensureSession()
      setActiveSid(sessionId)
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
          const refs = d.result.references || {}
          let mono = false
          if (d.result.passthrough && refs.structured_records) {
            answer = JSON.stringify(refs.structured_records, null, 2)
            mono = true
          } else if (d.result.answer) {
            answer = d.result.answer
          }
          // Replace the streamed placeholder with the final answer, carrying its
          // references, then point the pane at this turn.
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer, mono: mono || undefined, refs }])
          setSelectedRefIdx(botIdx)
        } else if (d.error) {
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('common.error', { msg: d.error }), error: true }])
        }
      }

      if (!started) setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('query.noResponse'), muted: true }])
    } catch (e) {
      setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('common.networkError', { msg: e.message }), error: true }])
    } finally {
      setQuerying(false)
      // The turn just updated the session index (title/activity); refresh the
      // sidebar so the current chat appears and re-sorts to the top.
      loadSessions()
    }
  }

  // Start a new chat: close the current session (triggering distillation) and
  // clear the view. A fresh session is opened lazily on the next question.
  function newChat() {
    closeSession(currentApp)
    setActiveSid(null)
    setSelectedRefIdx(-1)
    setMsgs([{ role: 'sys', text: t('query.refreshed') + (currentApp ? ' ' + t('query.connected', { app: currentApp }) : '') }])
    loadSessions()
  }

  // The pane reflects the selected answer's references (live turns auto-select
  // the newest; a loaded transcript defaults to its latest answer).
  const selectedRefs = msgs[selectedRefIdx]?.refs || {}
  const chunks = selectedRefs.chunks || []
  const structuredRecords = selectedRefs.structured_records || []
  const totalRefs = chunks.length + structuredRecords.length

  return (
    <>
      {!hasApp && <div className="warn-bar show">{t('common.noAppWarn')}</div>}
      <div className="chat-layout">
        <div className="chat-history">
          <div className="aside-hd">
            <h3>{t('query.chats')}</h3>
            <button className="btn btn-ghost btn-sm" disabled={!hasApp} onClick={newChat}>{t('query.newChat')}</button>
          </div>
          <div className="chat-history-body">
            {!sessions.length && (
              <div style={{ padding: '24px 12px', textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>
                {t('query.noChats')}
              </div>
            )}
            {sessions.map(s => (
              <div
                key={s.session_id}
                className={`chat-history-item${s.session_id === activeSid ? ' active' : ''}`}
                onClick={() => openSession(s.session_id)}
                title={s.title || t('query.untitledChat')}
              >
                <div className="chat-history-title">{s.title || t('query.untitledChat')}</div>
                <div className="chat-history-meta">
                  <span>{fmtRelTime(s.updated_at)}</span>
                  <span>{s.message_count !== 1 ? t('query.msgs', { n: s.message_count }) : t('query.msg', { n: s.message_count })}</span>
                </div>
                <button
                  className="chat-history-del"
                  title={t('query.deleteChat')}
                  aria-label={t('query.deleteChat')}
                  disabled={querying}
                  onClick={e => deleteSession(s.session_id, e)}
                >×</button>
              </div>
            ))}
          </div>
        </div>
        <div className="chat-col">
          <div className="msgs" ref={msgsRef}>
            {msgs.map((m, i) => {
              const selectable = m.role === 'bot' && !!m.refs
              return (
              <div
                key={i}
                className={`msg ${m.role}${selectable ? ' selectable' : ''}${selectable && i === selectedRefIdx ? ' ref-selected' : ''}`}
                onClick={selectable ? () => setSelectedRefIdx(i) : undefined}
                title={selectable ? t('query.showRefs') : undefined}
              >
                {m.role === 'user' && <div className="msg-who">{t('query.you')}</div>}
                {m.role === 'bot' && <div className="msg-who">{t('query.bot')}</div>}
                <div className="msg-body" style={{
                  color: m.error ? 'var(--red)' : m.thinking || m.muted ? 'var(--muted)' : undefined,
                  fontFamily: m.mono ? 'monospace' : undefined,
                  fontSize: m.mono ? 11 : undefined,
                }}>
                  {m.role === 'bot' && !m.mono && !m.thinking
                    ? <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents(t)}>{resolveArtifactLinks(m.text, apiUrl, currentApp)}</ReactMarkdown></div>
                    : m.text}
                </div>
                {(m.role === 'user' || m.role === 'bot') && !m.thinking && !m.error && !m.muted && (
                  <div className="msg-actions">
                    <CopyButton text={m.text} title={t('query.copy')} copiedTitle={t('query.copied')} />
                  </div>
                )}
              </div>
              )
            })}
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
            <button className="btn btn-ghost" title={t('query.newChat')} onClick={newChat}>↺</button>
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
    // Generated-artifact links get a dedicated downloader; everything else opens
    // in a new tab.
    a: ({ href, children }) => {
      const isDownload = /\/documents\/[^/]+\/download(?:[?#]|$)/.test(href || '')
      if (isDownload) return <DownloadLink href={href} t={t}>{children}</DownloadLink>
      return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
    },
  }
}

// A download link for a generated artifact. Two problems make a plain anchor
// unreliable here: the click bubbles to the enclosing message's ref-select
// handler, and a bare navigation to the download endpoint is flaky across
// origins/new-tab popup rules. So we stop propagation and fetch the file into a
// blob, then save it via a throwaway object-URL anchor — which downloads with
// the server's filename regardless of origin (the API allows all origins). If
// the fetch fails, fall back to opening the URL directly (the endpoint's
// Content-Disposition: attachment still triggers a download).
function DownloadLink({ href, children, t }) {
  const [busy, setBusy] = useState(false)
  async function onClick(e) {
    e.preventDefault()
    e.stopPropagation()
    if (busy) return
    setBusy(true)
    try {
      const resp = await fetch(href)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const blob = await resp.blob()
      const cd = resp.headers.get('Content-Disposition') || ''
      const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd)
      // The artifact id (second-to-last path segment) is the filename fallback.
      const name = m ? decodeURIComponent(m[1]) : decodeURIComponent(href.split('/').slice(-2, -1)[0] || 'download')
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = name
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch {
      window.open(href, '_blank', 'noopener')
    } finally {
      setBusy(false)
    }
  }
  return (
    <a href={href} onClick={onClick} title={t('query.download')} aria-busy={busy || undefined}>
      {children}
    </a>
  )
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
