import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'

// Behaviour-affecting kinds are the ones gated to pending_review; the rest
// (preferences, retrieval hints) go active without review. Filter accordingly.
const KIND_FILTERS = [
  { value: '', label: 'All kinds' },
  { value: 'fact', label: 'Facts' },
  { value: 'correction', label: 'Corrections' },
]

// The full kind set for the (read-only) records browser, which spans every kind.
const ALL_KIND_FILTERS = [
  { value: '', label: 'All kinds' },
  { value: 'fact', label: 'Facts' },
  { value: 'correction', label: 'Corrections' },
  { value: 'preference', label: 'Preferences' },
  { value: 'retrieval_hint', label: 'Retrieval hints' },
]

const STATUS_FILTERS = [
  { value: 'active', label: 'Active' },
  { value: 'all', label: 'All statuses' },
  { value: 'pending_review', label: 'Pending review' },
  { value: 'superseded', label: 'Superseded' },
]

// Map a distillation task status to the shared status-badge class.
const STATUS_BADGE = { done: 'b-active', failed: 'b-error', running: 'b-init', pending: 'b-init' }
// Map a memory record's lifecycle status to a status-badge class.
const MEM_STATUS_BADGE = { active: 'b-active', superseded: 'b-error', pending_review: 'b-init' }

export default function MemoryTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const [mode, setMode] = useState('review')       // 'review' (gate) | 'records' (browse)
  const [memories, setMemories] = useState(null)   // pending records; null=loading
  const [error, setError] = useState(null)
  const [kind, setKind] = useState('')
  const [busyId, setBusyId] = useState(null)       // memory_id with an in-flight verdict
  const [msg, setMsg] = useState({ text: '', cls: '' })
  const [runs, setRuns] = useState(null)           // distillation tasks; null=loading
  const [showRuns, setShowRuns] = useState(false)
  const [records, setRecords] = useState(null)     // browsed records; null=loading
  const [recError, setRecError] = useState(null)
  const [recStatus, setRecStatus] = useState('active')

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

  // Load the distillation task runs that feed this review queue (newest first).
  async function loadRuns() {
    if (!currentApp) { setRuns([]); return }
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/tasks?task_type=distill`)
      if (!resp.ok) { setRuns([]); return }
      const { tasks = [] } = await resp.json()
      tasks.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
      setRuns(tasks)
    } catch { setRuns([]) }
  }

  // Browse stored records (most-recently-observed first), filtered by status/kind.
  async function loadRecords() {
    if (!currentApp) { setRecords([]); setRecError(null); return }
    setRecords(null); setRecError(null)
    try {
      const params = new URLSearchParams({ status: recStatus })
      if (kind) params.set('kind', kind)
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/memory?${params}`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { memories: list = [] } = await resp.json()
      setRecords(list)
    } catch (e) { setRecError(e.message) }
  }

  function refresh() {
    if (mode === 'records') loadRecords()
    else { loadPending(); loadRuns() }
  }

  useEffect(() => { if (active && mode === 'review') { loadPending(); loadRuns() } }, [active, currentApp, kind, mode])
  useEffect(() => { if (active && mode === 'records') loadRecords() }, [active, currentApp, kind, recStatus, mode])

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
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="seg">
            <button className={mode === 'review' ? 'active' : ''} onClick={() => setMode('review')}>Review</button>
            <button className={mode === 'records' ? 'active' : ''} onClick={() => setMode('records')}>Records</button>
          </div>
          <button className="btn btn-ghost" onClick={refresh}>⟳ Refresh</button>
        </div>
      </div>

      {mode === 'review' ? (
        <p className="sub" style={{ marginBottom: 14 }}>
          Behaviour-affecting memories (<strong>facts</strong> and <strong>corrections</strong>) distilled from
          sessions are gated here as <code>pending_review</code> — they stay out of recall until accepted.
          {currentApp
            ? <> Reviewing for <strong>{currentApp}</strong>.</>
            : <> Select an app in the Apps tab to review its pending memories.</>}
        </p>
      ) : (
        <p className="sub" style={{ marginBottom: 14 }}>
          Browse the long-term memories distilled for <strong>{currentApp || 'this app'}</strong>, most-recently-observed
          first. <strong>Active</strong> records are the ones recalled into queries.
        </p>
      )}

      {!currentApp && <div className="empty"><div className="ei">🧠</div><p>No app selected. Choose one in the Apps tab.</p></div>}

      {currentApp && mode === 'review' && (
        <>
          <DistillRuns runs={runs} expanded={showRuns} onToggle={() => setShowRuns(v => !v)} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <select className="select" value={kind} onChange={e => setKind(e.target.value)}>
              {KIND_FILTERS.map(k => <option key={k.value} value={k.value}>{k.label}</option>)}
            </select>
            <span className={`settings-msg${msg.cls ? ' ' + msg.cls : ''}`}>{msg.text}</span>
          </div>

          {!memories && !error && <div className="empty"><p><span className="spinning">⟳</span> Loading…</p></div>}
          {error && <div className="empty"><p style={{ color: 'var(--red)' }}>Failed: {error}</p></div>}
          {memories && memories.length === 0 && !error && (
            <div className="empty"><div className="ei">🧠</div><p>Nothing to review. Pending memories appear here after sessions are distilled.</p></div>
          )}
          {memories && memories.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {memories.map(m => (
                <MemoryCard key={m.memory_id} memory={m} busy={busyId === m.memory_id} onReview={review} />
              ))}
            </div>
          )}
        </>
      )}

      {currentApp && mode === 'records' && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <select className="select" value={recStatus} onChange={e => setRecStatus(e.target.value)}>
              {STATUS_FILTERS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
            <select className="select" value={kind} onChange={e => setKind(e.target.value)}>
              {ALL_KIND_FILTERS.map(k => <option key={k.value} value={k.value}>{k.label}</option>)}
            </select>
            <span className="meta" style={{ fontSize: 12, color: 'var(--muted)' }}>
              {records ? `${records.length} record${records.length !== 1 ? 's' : ''}` : ''}
            </span>
          </div>

          {!records && !recError && <div className="empty"><p><span className="spinning">⟳</span> Loading…</p></div>}
          {recError && <div className="empty"><p style={{ color: 'var(--red)' }}>Failed: {recError}</p></div>}
          {records && records.length === 0 && !recError && (
            <div className="empty"><div className="ei">🧠</div><p>No memories yet. They accrue as sessions are distilled.</p></div>
          )}
          {records && records.length > 0 && <RecordsTable records={records} />}
        </>
      )}
    </div>
  )
}

// Collapsible summary of recent distillation runs — the producer of the
// pending-review queue below. Explains an empty queue (running vs. nothing found).
function DistillRuns({ runs, expanded, onToggle }) {
  const active = (runs || []).filter(r => r.status === 'pending' || r.status === 'running').length
  const failed = (runs || []).filter(r => r.status === 'failed').length
  const count = runs ? runs.length : null
  return (
    <div style={{ marginBottom: 16 }}>
      <button className="btn btn-ghost btn-sm" onClick={onToggle} style={{ width: '100%', justifyContent: 'flex-start', display: 'flex', gap: 8 }}>
        <span>{expanded ? '▲' : '▼'}</span>
        <span>Distillation runs{count != null ? ` (${count})` : ''}</span>
        {active > 0 && <span className="badge b-init">{active} in progress</span>}
        {failed > 0 && <span className="badge b-error">{failed} failed</span>}
      </button>

      {expanded && (
        <div style={{ marginTop: 10 }}>
          {!runs && <div className="empty"><p><span className="spinning">⟳</span> Loading…</p></div>}
          {runs && runs.length === 0 && (
            <div className="sub" style={{ padding: '8px 2px' }}>No distillation runs yet. Closing a session enqueues one.</div>
          )}
          {runs && runs.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th style={{ width: 100 }}>Status</th>
                  <th>Session</th>
                  <th style={{ width: 170 }}>Started</th>
                  <th style={{ width: 170 }}>Finished</th>
                </tr>
              </thead>
              <tbody>
                {runs.map(r => (
                  <tr key={r.task_id}>
                    <td><span className={`badge ${STATUS_BADGE[r.status] || 'b-init'}`}>{r.status}</span></td>
                    <td style={{ fontFamily: 'monospace', fontSize: 11 }} title={r.error || undefined}>
                      {r.doc_id || '—'}
                      {r.error && <div style={{ color: 'var(--red)', fontSize: 10 }}>{r.error}</div>}
                    </td>
                    <td style={{ color: 'var(--muted)', fontSize: 11 }}>{fmtTime(r.started_at)}</td>
                    <td style={{ color: 'var(--muted)', fontSize: 11 }}>{fmtTime(r.completed_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}

// Read-only table of stored memory records (the Records mode of the tab).
function RecordsTable({ records }) {
  return (
    <table className="data-tbl">
      <thead>
        <tr>
          <th style={{ width: 90 }}>Kind</th>
          <th>Content</th>
          <th>Entities</th>
          <th style={{ width: 60 }}>Conf.</th>
          <th style={{ width: 110 }}>Status</th>
          <th style={{ width: 110 }}>Observed</th>
        </tr>
      </thead>
      <tbody>
        {records.map(m => (
          <tr key={m.memory_id}>
            <td><span className="badge b-init">{m.kind}</span></td>
            <td>{m.content}</td>
            <td style={{ fontSize: 11, color: 'var(--muted)' }}>{(m.entities || []).join(', ') || '—'}</td>
            <td>{m.confidence != null ? Number(m.confidence).toFixed(2) : '—'}</td>
            <td><span className={`badge ${MEM_STATUS_BADGE[m.status] || 'b-init'}`}>{m.status}</span></td>
            <td style={{ fontSize: 11, color: 'var(--muted)' }} title={m.observed_at || ''}>{fmtDate(m.observed_at)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// Render an ISO-8601 timestamp as a compact local string, or an em dash.
function fmtTime(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return isNaN(d) ? iso : d.toLocaleString()
}

// Render an ISO-8601 timestamp as a date only (no time), or an em dash.
function fmtDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return isNaN(d) ? iso : d.toLocaleDateString()
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
