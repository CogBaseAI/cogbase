import React, { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { streamSSE, copyText, fmtRelTime, resolveArtifactLinks, latestDocxArtifact, artifactLabel, filenameFromContentDisposition } from '../../utils'

export default function QueryTab({ active, pendingQuery, onPendingConsumed, navSlot }) {
  const { apiUrl, appBase, authFetch, currentApp, apps } = useApp()
  const { t } = useT()
  // Chat log holds only real user/bot turns (plus the rare opened-empty-session
  // notice). The "select an app" prompt and app-intro live outside the log as
  // empty-state / banner, so they never linger as stale messages once a chat
  // starts.
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [querying, setQuerying] = useState(false)
  // Index into `msgs` of the answer whose Sources panel is open; -1 = closed.
  // Only one panel is open at a time, so a single floating overlay serves all
  // turns.
  const [sourcesIdx, setSourcesIdx] = useState(-1)
  // Document panel: renders the latest .docx artifact a bot answer produced
  // (redline etc.). Opens automatically when a new document appears; the user can
  // hide it. `docHidden` is the manual hide; the panel only exists when a doc does.
  const [docHidden, setDocHidden] = useState(false)
  const [sessions, setSessions] = useState([])
  const [activeSid, setActiveSid] = useState(null)
  const msgsRef = useRef(null)
  // Whether the message pane is pinned to the bottom. Streaming token updates
  // only auto-scroll while pinned, so a user who scrolls up to read the start of
  // a long answer isn't yanked back down on every token. Scrolling back to the
  // bottom re-pins.
  const atBottomRef = useRef(true)
  const sessionIdRef = useRef(null)
  const textareaRef = useRef(null)
  const hasApp = !!currentApp
  const prevAppRef = useRef(currentApp)

  // The most recent .docx artifact produced anywhere in this conversation.
  const currentDoc = latestDocxArtifact(msgs, apiUrl, currentApp)
  const prevDocIdRef = useRef(null)
  // Auto-reveal the panel when a new document appears (e.g. a refined redline).
  useEffect(() => {
    const id = currentDoc?.id || null
    if (id && id !== prevDocIdRef.current) {
      setDocHidden(false)
    }
    prevDocIdRef.current = id
  }, [currentDoc?.id])

  useEffect(() => {
    if (currentApp !== prevAppRef.current) {
      const prevApp = prevAppRef.current
      prevAppRef.current = currentApp
      closeSession(prevApp)
      // Drop the previous app's chat and session pointer so nothing lingers
      // across the switch. The new app's empty state (intro banner + starter
      // chips) renders from the cleared log.
      setActiveSid(null)
      setSourcesIdx(-1)
      setMsgs([])
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
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/sessions`)
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
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/sessions/${encodeURIComponent(sid)}`)
      if (!resp.ok) return
      const data = await resp.json()
      const rawMsgs = data.messages || []
      // Carry each assistant turn's references onto its message so its inline
      // Sources drawer can show that turn's evidence.
      const loaded = rawMsgs.map(m => ({
        role: m.role === 'assistant' ? 'bot' : 'user',
        text: m.content,
        ...(m.role === 'assistant' ? { refs: m.references || {} } : {}),
      }))
      sessionIdRef.current = sid
      setActiveSid(sid)
      setSourcesIdx(-1)
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
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/sessions/${encodeURIComponent(sid)}`, {
        method: 'DELETE',
      })
      if (!resp.ok) return
    } catch { return }
    if (sid === sessionIdRef.current) {
      sessionIdRef.current = null
      setActiveSid(null)
      setSourcesIdx(-1)
      setMsgs([])
    }
    loadSessions()
  }

  // Close the session bound to `appName`, fire-and-forget, and clear the local handle.
  function closeSession(appName) {
    const sid = sessionIdRef.current
    sessionIdRef.current = null
    if (!sid || !appName) return
    authFetch(`${appBase}/${encodeURIComponent(appName)}/sessions/${encodeURIComponent(sid)}/close`, {
      method: 'POST',
    }).catch(() => {})
  }

  // Open a session for the current app if one isn't already open; returns its id.
  async function ensureSession() {
    if (sessionIdRef.current) return sessionIdRef.current
    const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    if (!resp.ok) throw new Error(`failed to start session: HTTP ${resp.status}`)
    const data = await resp.json()
    sessionIdRef.current = data.session_id
    return data.session_id
  }

  // Force the pane to the bottom and re-pin. Used when the user drives an action
  // that should follow the newest content (sending a query, opening a chat).
  function scrollMsgs() {
    if (msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight
    atBottomRef.current = true
  }

  // Scroll to the bottom only if the user hasn't scrolled up. Used during token
  // streaming so it follows new output without fighting a manual scroll-up.
  function scrollMsgsIfPinned() {
    if (atBottomRef.current && msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight
  }

  // Re-evaluate the pin on manual scroll: pinned when within a small slack of the
  // bottom, so tiny rounding gaps still count as "at bottom".
  function onMsgsScroll() {
    const el = msgsRef.current
    if (el) atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }

  function autoResize(el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 110) + 'px'
  }

  async function sendQuery(override) {
    if (querying || !currentApp) return
    const text = (typeof override === 'string' ? override : input).trim()
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
      setActiveSid(sessionId)
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/query/stream`, {
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
          setTimeout(scrollMsgsIfPinned, 0)
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
          // references onto the message for its inline Sources drawer.
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer, mono: mono || undefined, refs }])
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
    setSourcesIdx(-1)
    setMsgs([])
    loadSessions()
  }

  // Consume a query handed off from the Ingest tab's post-upload CTA: once this
  // tab is active and an app is selected, either auto-send it (single-doc review)
  // or just prefill the input so the user can pick which contract, then send
  // (multi-doc review). Always in a fresh chat — a review shouldn't thread onto
  // whatever conversation was open. Fires once — the parent clears it on consume.
  useEffect(() => {
    if (!active || !pendingQuery || !currentApp) return
    const { text, send } = pendingQuery
    if (send && querying) return // wait for the in-flight answer; harmless to defer
    onPendingConsumed?.()
    newChat() // close the current session, clear the view, start clean
    if (send) {
      sendQuery(text)
    } else {
      setInput(text)
      // Focus and size the box, cursor at the end, so editing which contract to
      // review is a keystroke away.
      setTimeout(() => {
        const el = textareaRef.current
        if (!el) return
        el.focus()
        autoResize(el)
        const n = el.value.length
        el.setSelectionRange(n, n)
      }, 0)
    }
  }, [active, pendingQuery, currentApp])

  // A fresh chat (only system notices, no user/bot turns) has nothing yet; used
  // to gate the starter chips.
  const hasChat = msgs.some(m => m.role === 'user' || m.role === 'bot')
  // Config-driven starter prompts for the current app (from config.yaml, carried
  // on the app list payload). Shown in the empty state to teach a new user what
  // this app can do; clicking one runs it.
  const appConfig = apps.find(a => a.name === currentApp)?.config
  const exampleQueries = (appConfig?.example_queries || []).filter(q => typeof q === 'string' && q.trim())
  const queryIntro = typeof appConfig?.query_intro === 'string' ? appConfig.query_intro.trim() : ''
  // The intro banner persists above the conversation (visible during and after
  // an answer); the example chips are a starter shown only until the first turn.
  const showIntro = hasApp && !!queryIntro
  const showStarter = hasApp && !hasChat && exampleQueries.length > 0

  // The chats list. It lives in the app sidebar's lower (contextual) slot — this
  // tab renders it there via a portal when a slot is provided (see App.jsx), and
  // inline as a fallback otherwise (e.g. standalone tests). Hiding is now the
  // sidebar-level control, so there's no per-tab collapse toggle here.
  const chatHistory = (
    <div className="chat-history">
      <div className="aside-hd">
        <h3 style={{ flex: 1 }}>{t('query.chats')}</h3>
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
  )

  return (
    <>
      {!hasApp && <div className="warn-bar show">{t('common.noAppWarn')}</div>}
      {/* Portal the chats list into the sidebar's contextual slot, but only while
          this tab is active — the panel stays mounted when hidden, so an
          unconditional portal would stack the chats onto whatever the active tab
          shows there. Inline (no slot) is the standalone-test fallback. */}
      {navSlot ? (active && createPortal(chatHistory, navSlot)) : chatHistory}
      <div className="chat-layout">
        <div className="chat-col">
          {showIntro && <div className="query-intro-banner">{queryIntro}</div>}
          <div className="msgs" ref={msgsRef} onScroll={onMsgsScroll}>
            {!hasApp && !hasChat && (
              <div className="query-empty">{t('query.intro')}</div>
            )}
            {showStarter && (
              <div className="query-starter">
                <div className="query-starter-label">{t('query.tryAsking')}</div>
                <div className="chips">
                  {exampleQueries.map((q, i) => (
                    <button
                      key={i}
                      className="chip chip-btn"
                      disabled={querying}
                      onClick={() => sendQuery(q)}
                    >{q}</button>
                  ))}
                </div>
              </div>
            )}
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
                    ? <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents(t)}>{resolveArtifactLinks(m.text, apiUrl, currentApp)}</ReactMarkdown></div>
                    : m.text}
                </div>
                {m.role === 'bot' && m.refs && (
                  <SourcesToggle
                    refs={m.refs}
                    open={sourcesIdx === i}
                    onToggle={() => setSourcesIdx(sourcesIdx === i ? -1 : i)}
                    t={t}
                  />
                )}
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
            <button className="btn btn-ghost" title={t('query.newChat')} onClick={newChat}>↺</button>
            <button className="btn btn-primary" disabled={querying || !hasApp} onClick={sendQuery}>{t('common.send')}</button>
          </div>
        </div>
        {currentDoc && docHidden && (
          <div className="chat-aside-min">
            <button className="aside-toggle" title={t('query.showDoc')} aria-label={t('query.showDoc')} onClick={() => setDocHidden(false)}><DocIcon /></button>
          </div>
        )}
        {currentDoc && !docHidden && (
          <DocPanel doc={currentDoc} t={t} onHide={() => setDocHidden(true)} />
        )}
        {sourcesIdx >= 0 && msgs[sourcesIdx]?.refs && (
          <SourcesPanel refs={msgs[sourcesIdx].refs} t={t} onClose={() => setSourcesIdx(-1)} />
        )}
      </div>
    </>
  )
}

// A right-hand panel that renders the latest generated .docx (redline etc.) with
// tracked changes visible, via the docx-preview library — lazily imported so the
// dependency is code-split and only loaded once a document actually appears.
// Persisted doc-panel width. Best-effort: any storage failure (private mode,
// quota, disabled) just falls back to the flex default.
const DOC_WIDTH_KEY = 'cogbase.docPanelWidth'
function readStoredDocWidth() {
  try {
    const v = parseInt(localStorage.getItem(DOC_WIDTH_KEY), 10)
    return Number.isFinite(v) ? Math.min(Math.max(v, 360), 1200) : null
  } catch { return null }
}
function writeStoredDocWidth(w) {
  try { localStorage.setItem(DOC_WIDTH_KEY, String(w)) } catch {}
}

function DocPanel({ doc, t, onHide }) {
  const { authFetch } = useApp()
  const bodyRef = useRef(null)
  const panelRef = useRef(null)
  const [status, setStatus] = useState('loading')  // 'loading' | 'ready' | 'error'
  // null → grow to the CSS default (flex up to max-width); a number → explicit
  // width the user dragged the panel to. Persisted across mounts/reloads so the
  // panel reopens at the width the user last chose.
  const [width, setWidth] = useState(readStoredDocWidth)
  const [dragging, setDragging] = useState(false)

  // Drag the left-edge handle: the panel is right-anchored, so width is the gap
  // between the panel's right edge and the cursor. Clamped to a sane range.
  function startResize(e) {
    e.preventDefault()
    const rightEdge = panelRef.current.getBoundingClientRect().right
    setDragging(true)
    document.body.style.userSelect = 'none'
    const onMove = ev => setWidth(Math.min(Math.max(rightEdge - ev.clientX, 360), 1200))
    const onUp = () => {
      setDragging(false)
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      setWidth(w => { writeStoredDocWidth(w); return w })
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  useEffect(() => {
    let cancelled = false
    const container = bodyRef.current
    if (!container) return
    setStatus('loading')
    container.innerHTML = ''
    async function render() {
      try {
        const [{ renderAsync }, resp] = await Promise.all([
          import('docx-preview'),
          authFetch(doc.url),
        ])
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const blob = await resp.blob()
        if (cancelled || !bodyRef.current) return
        bodyRef.current.innerHTML = ''
        await renderAsync(blob, bodyRef.current, undefined, {
          inWrapper: true,
          renderChanges: true,          // show w:ins / w:del tracked changes
          ignoreLastRenderedPageBreak: true,
        })
        if (!cancelled) setStatus('ready')
      } catch {
        if (!cancelled) setStatus('error')
      }
    }
    render()
    return () => { cancelled = true }
  }, [doc.url])

  return (
    <div
      className="chat-doc-panel"
      ref={panelRef}
      style={width != null ? { flex: 'none', width, maxWidth: 'none' } : undefined}
    >
      <div
        className={`chat-doc-resizer${dragging ? ' dragging' : ''}`}
        onMouseDown={startResize}
        title={t('query.resizeDoc')}
        role="separator"
        aria-orientation="vertical"
      />
      <div className="aside-hd">
        <button className="aside-toggle" title={t('query.hideDoc')} aria-label={t('query.hideDoc')} onClick={onHide}><PanelIcon /></button>
        <h3 className="chat-doc-title" title={artifactLabel(doc.id)}>{artifactLabel(doc.id)}</h3>
        <DownloadLink href={doc.url} t={t}><DownloadIcon /></DownloadLink>
      </div>
      {status === 'loading' && <div className="chat-doc-note">{t('query.docLoading')}</div>}
      {status === 'error' && <div className="chat-doc-note chat-doc-err">{t('query.docError')}</div>}
      <div className="chat-doc-body" ref={bodyRef} style={{ display: status === 'ready' ? 'block' : 'none' }} />
    </div>
  )
}

// Document glyph for the doc-panel reopen rail.
function DocIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </svg>
  )
}

// Small download glyph for the doc-panel header.
function DownloadIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 10l5 5 5-5" />
      <path d="M12 15V3" />
    </svg>
  )
}

// Right-side panel toggle, matching the sidebar-collapse glyph ChatGPT/Claude
// use: a framed rectangle with a divided right column standing in for the pane.
function PanelIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <line x1="15" y1="3" x2="15" y2="21" />
    </svg>
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
  const { authFetch } = useApp()
  const [busy, setBusy] = useState(false)
  async function onClick(e) {
    e.preventDefault()
    e.stopPropagation()
    if (busy) return
    setBusy(true)
    try {
      const resp = await authFetch(href)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const blob = await resp.blob()
      const cd = resp.headers.get('Content-Disposition') || ''
      const name = filenameFromContentDisposition(cd, href)
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

// The inline trigger under a bot turn. Clicking it opens (or, when already this
// turn's, closes) the floating Sources panel; the caret rotates while its panel
// is open. Renders nothing when the turn cited nothing.
function SourcesToggle({ refs, open, onToggle, t }) {
  const chunks = refs.chunks || []
  const structuredRecords = refs.structured_records || []
  const total = chunks.length + structuredRecords.length
  if (!total) return null
  return (
    <div className={`msg-sources${open ? ' open' : ''}`}>
      <button className="msg-sources-toggle" onClick={onToggle} aria-expanded={open}>
        <ChevronIcon />
        {t('query.sources', { n: total })}
      </button>
    </div>
  )
}

// Per-answer citations shown in a right-anchored slide-over that floats over the
// chat (and the docx panel) on demand — matching the ChatGPT/Claude pattern
// rather than a standing column. Dismisses via its ✕, Escape, or a click
// outside the panel.
function SourcesPanel({ refs, t, onClose }) {
  const panelRef = useRef(null)
  const chunks = refs.chunks || []
  const structuredRecords = refs.structured_records || []
  const total = chunks.length + structuredRecords.length

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose() }
    // Close on an outside mousedown, but leave the Sources triggers alone: their
    // own onClick owns toggling/swapping, and acting here too would race it
    // (close-then-reopen).
    function onDown(e) {
      if (panelRef.current && !panelRef.current.contains(e.target) && !e.target.closest('.msg-sources-toggle')) onClose()
    }
    window.addEventListener('keydown', onKey)
    // Defer the outside-click listener a tick so the click that opened the panel
    // (which lands outside it) doesn't immediately close it again.
    const id = setTimeout(() => window.addEventListener('mousedown', onDown), 0)
    return () => {
      clearTimeout(id)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('mousedown', onDown)
    }
  }, [onClose])

  return (
    <div className="sources-panel" ref={panelRef} role="dialog" aria-label={t('query.sources', { n: total })}>
      <div className="aside-hd">
        <h3 style={{ flex: 1 }}>{t('query.sources', { n: total })}</h3>
        <button className="aside-toggle" title={t('query.close')} aria-label={t('query.close')} onClick={onClose}><CloseIcon /></button>
      </div>
      <div className="aside-body">
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
  )
}

// Disclosure caret for the Sources trigger; rotates when its panel is open (CSS).
function ChevronIcon() {
  return (
    <svg className="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 18l6-6-6-6" />
    </svg>
  )
}

// ✕ glyph for the Sources panel's close control.
function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M18 6 6 18" /><path d="M6 6l12 12" />
    </svg>
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
