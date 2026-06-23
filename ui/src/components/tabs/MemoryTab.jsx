import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'

// Behaviour-affecting kinds are the ones gated to pending_review; the rest
// (preferences, retrieval hints) go active without review. Filter accordingly.
const KIND_FILTERS = [
  { value: '', label: 'All kinds' },
  { value: 'fact', label: 'Facts' },
  { value: 'correction', label: 'Corrections' },
]

export default function MemoryTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const [memories, setMemories] = useState(null)  // null=loading, []|[...]=loaded
  const [error, setError] = useState(null)
  const [kind, setKind] = useState('')
  const [busyId, setBusyId] = useState(null)       // memory_id with an in-flight verdict
  const [msg, setMsg] = useState({ text: '', cls: '' })

  async function loadPending() {
    if (!currentApp) { setMemories([]); setError(null); return }
    setMemories(null); setError(null)
    try {
      const qs = kind ? `?kind=${encodeURIComponent(kind)}` : ''
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/memory/pending${qs}`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { memories: list = [] } = await resp.json()
      setMemories(list)
    } catch (e) { setError(e.message) }
  }

  useEffect(() => { if (active) loadPending() }, [active, currentApp, kind])

  // Send a single accept/reject verdict, then drop the row on success.
  async function review(memory, decision) {
    setBusyId(memory.memory_id)
    setMsg({ text: '', cls: '' })
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/memory/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decisions: [{ memory_id: memory.memory_id, decision }] }),
      })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        setMsg({ text: (Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || `Error ${resp.status}`, cls: 'err' })
        return
      }
      const { results = [] } = await resp.json()
      const outcome = results[0]?.outcome || 'done'
      setMemories(prev => (prev || []).filter(m => m.memory_id !== memory.memory_id))
      setMsg({ text: outcome === 'accepted' ? 'Accepted — now active.' : outcome === 'rejected' ? 'Rejected — superseded.' : `Outcome: ${outcome}`, cls: 'ok' })
    } catch (e) {
      setMsg({ text: 'Network error: ' + e.message, cls: 'err' })
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div className="page">
      <div className="page-hd">
        <h2>Memory</h2>
        <button className="btn btn-ghost" onClick={loadPending}>⟳ Refresh</button>
      </div>

      <p className="sub" style={{ marginBottom: 14 }}>
        Behaviour-affecting memories (<strong>facts</strong> and <strong>corrections</strong>) distilled from
        sessions are gated here as <code>pending_review</code> — they stay out of recall until accepted.
        {currentApp
          ? <> Reviewing for <strong>{currentApp}</strong>.</>
          : <> Select an app in the Apps tab to review its pending memories.</>}
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <select
          className="select"
          value={kind}
          disabled={!currentApp}
          onChange={e => setKind(e.target.value)}
        >
          {KIND_FILTERS.map(k => <option key={k.value} value={k.value}>{k.label}</option>)}
        </select>
        <span className={`settings-msg${msg.cls ? ' ' + msg.cls : ''}`}>{msg.text}</span>
      </div>

      {!currentApp && <div className="empty"><div className="ei">🧠</div><p>No app selected. Choose one in the Apps tab.</p></div>}
      {currentApp && !memories && !error && <div className="empty"><p><span className="spinning">⟳</span> Loading…</p></div>}
      {error && <div className="empty"><p style={{ color: 'var(--red)' }}>Failed: {error}</p></div>}
      {currentApp && memories && memories.length === 0 && !error && (
        <div className="empty"><div className="ei">🧠</div><p>Nothing to review. Pending memories appear here after sessions are distilled.</p></div>
      )}

      {currentApp && memories && memories.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {memories.map(m => (
            <MemoryCard key={m.memory_id} memory={m} busy={busyId === m.memory_id} onReview={review} />
          ))}
        </div>
      )}
    </div>
  )
}

function MemoryCard({ memory, busy, onReview }) {
  const [showEvidence, setShowEvidence] = useState(false)
  const conf = memory.confidence != null ? Number(memory.confidence).toFixed(2) : null
  const hasEvidence = (memory.source_event_ids?.length || 0) > 0 || (memory.evidence_snapshot && Object.keys(memory.evidence_snapshot).length > 0)
  return (
    <div className="ref-card" style={{ cursor: 'default' }}>
      <div className="ref-meta" style={{ marginBottom: 8 }}>
        <span className="badge b-init">{memory.kind}</span>
        {conf !== null && <span className="ref-score" title="confidence">conf {conf}</span>}
        {(memory.entities || []).map(e => <span key={e} className="chip">{e}</span>)}
      </div>

      <div style={{ fontSize: 13, lineHeight: 1.5, marginBottom: 10 }}>{memory.content}</div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button className="btn btn-green btn-sm" disabled={busy} onClick={() => onReview(memory, 'accept')}>
          {busy ? '⟳' : '✓'} Accept
        </button>
        <button className="btn btn-red btn-sm" disabled={busy} onClick={() => onReview(memory, 'reject')}>
          ✕ Reject
        </button>
        {hasEvidence && (
          <button className="btn btn-ghost btn-sm" onClick={() => setShowEvidence(v => !v)}>
            {showEvidence ? '▲ Hide evidence' : '▼ Evidence'}
          </button>
        )}
      </div>

      {showEvidence && (
        <pre className="ref-code" style={{ marginTop: 10 }}>
          {JSON.stringify({ source_event_ids: memory.source_event_ids, evidence_snapshot: memory.evidence_snapshot }, null, 2)}
        </pre>
      )}
    </div>
  )
}
