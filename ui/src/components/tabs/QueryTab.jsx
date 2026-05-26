import React, { useState, useRef, useEffect } from 'react'
import { useApp } from '../../context'
import { streamSSE } from '../../utils'

export default function QueryTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const [msgs, setMsgs] = useState([{ role: 'sys', text: 'Select an app, then start asking questions in natural language.' }])
  const [input, setInput] = useState('')
  const [querying, setQuerying] = useState(false)
  const [chunks, setChunks] = useState([])
  const [structuredRecords, setStructuredRecords] = useState([])
  const msgsRef = useRef(null)
  const historyRef = useRef([])
  const textareaRef = useRef(null)
  const hasApp = !!currentApp
  const prevAppRef = useRef(currentApp)

  useEffect(() => {
    if (currentApp !== prevAppRef.current) {
      prevAppRef.current = currentApp
      historyRef.current = []
      if (currentApp) {
        setMsgs(prev => [...prev, { role: 'sys', text: `Connected to "${currentApp}".` }])
      }
    }
  }, [currentApp])

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
    const botMsg = { role: 'bot', text: 'Thinking…', thinking: true }
    setMsgs(prev => [...prev, userMsg, botMsg])
    setTimeout(scrollMsgs, 0)

    let answer = ''
    let started = false
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, history: historyRef.current }),
      })
      if (!resp.ok) {
        const errText = await resp.text()
        setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: `HTTP ${resp.status}: ${errText}`, error: true }])
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
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: 'Error: ' + d.error, error: true }])
        }
      }

      historyRef.current.push({ role: 'user', content: text })
      historyRef.current.push({ role: 'assistant', content: answer })
      if (!started) setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: '(no response)', muted: true }])
    } catch (e) {
      setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: 'Network error: ' + e.message, error: true }])
    } finally {
      setQuerying(false)
    }
  }

  function clearHistory() {
    historyRef.current = []
    setChunks([])
    setStructuredRecords([])
    setMsgs([{ role: 'sys', text: 'History cleared.' + (currentApp ? ` Connected to "${currentApp}".` : '') }])
  }

  const totalRefs = chunks.length + structuredRecords.length

  return (
    <>
      {!hasApp && <div className="warn-bar show">⚠ No app selected — go to Apps and click Use on an app first.</div>}
      <div className="chat-layout">
        <div className="chat-col">
          <div className="msgs" ref={msgsRef}>
            {msgs.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                {m.role === 'user' && <div className="msg-who">You</div>}
                {m.role === 'bot' && <div className="msg-who">CogBase</div>}
                <div className="msg-body" style={{
                  color: m.error ? 'var(--red)' : m.thinking || m.muted ? 'var(--muted)' : undefined,
                  fontFamily: m.mono ? 'monospace' : undefined,
                  fontSize: m.mono ? 11 : undefined,
                }}>{m.text}</div>
              </div>
            ))}
          </div>
          <div className="chat-input">
            <textarea
              ref={textareaRef}
              value={input}
              placeholder="Ask a question…"
              rows={1}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery() } }}
              onChange={e => { setInput(e.target.value); autoResize(e.target) }}
              disabled={querying || !hasApp}
            />
            <button className="btn btn-ghost" title="Clear history" onClick={clearHistory}>↺</button>
            <button className="btn btn-primary" disabled={querying || !hasApp} onClick={sendQuery}>Send</button>
          </div>
        </div>
        <div className="chat-aside">
          <div className="aside-hd">
            <h3>References</h3>
            <span style={{ fontSize: 11, color: 'var(--muted)' }}>{totalRefs ? `${totalRefs} ref${totalRefs !== 1 ? 's' : ''}` : '—'}</span>
          </div>
          <div className="aside-body">
            {!totalRefs && <div style={{ padding: '30px 10px', textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>References will appear here after a query.</div>}
            {structuredRecords.length > 0 && (
              <>
                <div className="ref-section-hd">Structured Records ({structuredRecords.length})</div>
                {structuredRecords.map((rec, i) => (
                  <div key={i} className="ref-card"><pre className="ref-code">{JSON.stringify(rec, null, 2)}</pre></div>
                ))}
              </>
            )}
            {chunks.length > 0 && (
              <>
                <div className="ref-section-hd">Passages ({chunks.length})</div>
                {chunks.map((ch, i) => <RefChunk key={i} chunk={ch} />)}
              </>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function RefChunk({ chunk }) {
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
      {long && <div className="ref-expand-hint">{expanded ? '▲ collapse' : '▼ expand'}</div>}
    </div>
  )
}
