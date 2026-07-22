import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import DataTable from '../DataTable'

// Behaviour-affecting kinds are the ones gated to pending_review; the rest
// (preferences, retrieval hints) go active without review. Filter accordingly.
// `labelKey` is resolved via the i18n t() at render time.
const KIND_FILTERS = [
  { value: '', labelKey: 'memory.kindAll' },
  { value: 'fact', labelKey: 'memory.kindFact' },
  { value: 'correction', labelKey: 'memory.kindCorrection' },
]

// The full kind set for the (read-only) records browser, which spans every kind.
const ALL_KIND_FILTERS = [
  { value: '', labelKey: 'memory.kindAll' },
  { value: 'fact', labelKey: 'memory.kindFact' },
  { value: 'correction', labelKey: 'memory.kindCorrection' },
  { value: 'preference', labelKey: 'memory.kindPreference' },
  { value: 'retrieval_hint', labelKey: 'memory.kindRetrievalHint' },
]

const STATUS_FILTERS = [
  { value: 'active', labelKey: 'memory.statusActive' },
  { value: 'all', labelKey: 'memory.statusAll' },
  { value: 'pending_review', labelKey: 'memory.statusPending' },
  { value: 'superseded', labelKey: 'memory.statusSuperseded' },
]

// Map a distillation task status to the shared status-badge class.
const STATUS_BADGE = { done: 'b-active', failed: 'b-error', running: 'b-init', pending: 'b-init' }
// Map a memory record's lifecycle status to a status-badge class.
const MEM_STATUS_BADGE = { active: 'b-active', superseded: 'b-error', pending_review: 'b-init' }

export default function MemoryTab({ active }) {
  const { currentAppBase: appBase, authFetch, currentApp } = useApp()
  const { t } = useT()
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
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/memory/pending${qs}`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { memories: list = [] } = await resp.json()
      setMemories(list)
    } catch (e) { setError(e.message) }
  }

  // Load the distillation task runs that feed this review queue (newest first).
  async function loadRuns() {
    if (!currentApp) { setRuns([]); return }
    try {
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/tasks?task_type=distill`)
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
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/memory?${params}`)
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
      const resp = await authFetch(`${appBase}/${encodeURIComponent(currentApp)}/memory/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decisions: [{ memory_id: memory.memory_id, decision }] }),
      })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        setMsg({ text: (Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || t('memory.errStatus', { status: resp.status }), cls: 'err' })
        return
      }
      const { results = [] } = await resp.json()
      const outcome = results[0]?.outcome || 'done'
      setMemories(prev => (prev || []).filter(m => m.memory_id !== memory.memory_id))
      setMsg({ text: outcome === 'accepted' ? t('memory.accepted') : outcome === 'rejected' ? t('memory.rejected') : t('memory.outcome', { outcome }), cls: 'ok' })
    } catch (e) {
      setMsg({ text: t('common.networkError', { msg: e.message }), cls: 'err' })
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div className="page">
      <div className="page-hd">
        <h2>{t('memory.title')}</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="seg">
            <button className={mode === 'review' ? 'active' : ''} onClick={() => setMode('review')}>{t('memory.review')}</button>
            <button className={mode === 'records' ? 'active' : ''} onClick={() => setMode('records')}>{t('memory.records')}</button>
          </div>
          <button className="btn btn-ghost" onClick={refresh}>{t('common.refresh')}</button>
        </div>
      </div>

      {mode === 'review' ? (
        <p className="sub" style={{ marginBottom: 14 }}>
          {t('memory.reviewSubA')}<strong>{t('memory.reviewSubFacts')}</strong>{t('memory.reviewSubAnd')}<strong>{t('memory.reviewSubCorrections')}</strong>{t('memory.reviewSubB')}<code>{t('memory.reviewSubCode')}</code>{t('memory.reviewSubC')}
          {currentApp
            ? <>{t('memory.reviewSubFor')}<strong>{currentApp}</strong>.</>
            : <>{t('memory.reviewSubSelect')}</>}
        </p>
      ) : (
        <p className="sub" style={{ marginBottom: 14 }}>
          {t('memory.recordsSubA')}<strong>{currentApp || t('memory.recordsSubThisApp')}</strong>{t('memory.recordsSubB')}<strong>{t('memory.recordsSubActive')}</strong>{t('memory.recordsSubC')}
        </p>
      )}

      {!currentApp && <div className="empty"><div className="ei">🧠</div><p>{t('memory.noAppMem')}</p></div>}

      {currentApp && mode === 'review' && (
        <>
          <DistillRuns runs={runs} expanded={showRuns} onToggle={() => setShowRuns(v => !v)} t={t} />

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <select className="select" value={kind} onChange={e => setKind(e.target.value)}>
              {KIND_FILTERS.map(k => <option key={k.value} value={k.value}>{t(k.labelKey)}</option>)}
            </select>
            <span className={`settings-msg${msg.cls ? ' ' + msg.cls : ''}`}>{msg.text}</span>
          </div>

          {!memories && !error && <div className="empty"><p><span className="spinning">⟳</span> {t('common.loading')}</p></div>}
          {error && <div className="empty"><p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: error })}</p></div>}
          {memories && memories.length === 0 && !error && (
            <div className="empty"><div className="ei">🧠</div><p>{t('memory.nothingReview')}</p></div>
          )}
          {memories && memories.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {memories.map(m => (
                <MemoryCard key={m.memory_id} memory={m} busy={busyId === m.memory_id} onReview={review} t={t} />
              ))}
            </div>
          )}
        </>
      )}

      {currentApp && mode === 'records' && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
            <select className="select" value={recStatus} onChange={e => setRecStatus(e.target.value)}>
              {STATUS_FILTERS.map(s => <option key={s.value} value={s.value}>{t(s.labelKey)}</option>)}
            </select>
            <select className="select" value={kind} onChange={e => setKind(e.target.value)}>
              {ALL_KIND_FILTERS.map(k => <option key={k.value} value={k.value}>{t(k.labelKey)}</option>)}
            </select>
            <span className="meta" style={{ fontSize: 12, color: 'var(--muted)' }}>
              {records ? (records.length !== 1 ? t('memory.recordsCount', { n: records.length }) : t('memory.recordCount', { n: records.length })) : ''}
            </span>
          </div>

          {!records && !recError && <div className="empty"><p><span className="spinning">⟳</span> {t('common.loading')}</p></div>}
          {recError && <div className="empty"><p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: recError })}</p></div>}
          {records && records.length === 0 && !recError && (
            <div className="empty"><div className="ei">🧠</div><p>{t('memory.noMemories')}</p></div>
          )}
          {records && records.length > 0 && <RecordsTable records={records} t={t} />}
        </>
      )}
    </div>
  )
}

// Collapsible summary of recent distillation runs — the producer of the
// pending-review queue below. Explains an empty queue (running vs. nothing found).
function DistillRuns({ runs, expanded, onToggle, t }) {
  const active = (runs || []).filter(r => r.status === 'pending' || r.status === 'running').length
  const failed = (runs || []).filter(r => r.status === 'failed').length
  const count = runs ? runs.length : null
  return (
    <div style={{ marginBottom: 16 }}>
      <button className="btn btn-ghost btn-sm" onClick={onToggle} style={{ width: '100%', justifyContent: 'flex-start', display: 'flex', gap: 8 }}>
        <span>{expanded ? '▲' : '▼'}</span>
        <span>{count != null ? t('memory.distillRunsN', { n: count }) : t('memory.distillRuns')}</span>
        {active > 0 && <span className="badge b-init">{t('memory.inProgress', { n: active })}</span>}
        {failed > 0 && <span className="badge b-error">{t('memory.nFailed', { n: failed })}</span>}
      </button>

      {expanded && (
        <div style={{ marginTop: 10 }}>
          {!runs && <div className="empty"><p><span className="spinning">⟳</span> {t('common.loading')}</p></div>}
          {runs && runs.length === 0 && (
            <div className="sub" style={{ padding: '8px 2px' }}>{t('memory.noRuns')}</div>
          )}
          {runs && runs.length > 0 && (
            <DataTable
              rows={runs}
              rowKey={r => r.task_id}
              columns={[
                {
                  key: 'status', label: t('memory.colStatus'), width: 100, text: r => r.status,
                  render: r => <span className={`badge ${STATUS_BADGE[r.status] || 'b-init'}`}>{r.status}</span>,
                },
                {
                  key: 'session', label: t('memory.colSession'), text: r => `${r.doc_id || ''} ${r.error || ''}`,
                  cellClassName: 'mono-cell',
                  render: r => (
                    <span title={r.error || undefined}>
                      {r.doc_id || '—'}
                      {r.error && <div style={{ color: 'var(--red)', fontSize: 10 }}>{r.error}</div>}
                    </span>
                  ),
                },
                { key: 'started', label: t('memory.colStarted'), width: 170, cellClassName: 'muted-cell', sortValue: r => r.started_at || '', render: r => fmtTime(r.started_at) },
                { key: 'finished', label: t('memory.colFinished'), width: 170, cellClassName: 'muted-cell', sortValue: r => r.completed_at || '', render: r => fmtTime(r.completed_at) },
              ]}
            />
          )}
        </div>
      )}
    </div>
  )
}

// Read-only table of stored memory records (the Records mode of the tab).
function RecordsTable({ records, t }) {
  return (
    <DataTable
      rows={records}
      rowKey={m => m.memory_id}
      columns={[
        {
          key: 'kind', label: t('memory.colKind'), width: 90, text: m => m.kind,
          render: m => <span className="badge b-init">{m.kind}</span>,
        },
        { key: 'content', label: t('memory.colContent'), value: m => m.content },
        { key: 'entities', label: t('memory.colEntities'), cellClassName: 'muted-cell', value: m => (m.entities || []).join(', ') || '—' },
        {
          key: 'conf', label: t('memory.colConf'), width: 60, align: 'right',
          sortValue: m => (m.confidence != null ? Number(m.confidence) : null),
          render: m => (m.confidence != null ? Number(m.confidence).toFixed(2) : '—'),
        },
        {
          key: 'status', label: t('memory.colStatus'), width: 110, text: m => m.status,
          render: m => <span className={`badge ${MEM_STATUS_BADGE[m.status] || 'b-init'}`}>{m.status}</span>,
        },
        {
          key: 'observed', label: t('memory.colObserved'), width: 110, cellClassName: 'muted-cell',
          sortValue: m => m.observed_at || '', render: m => fmtDate(m.observed_at),
        },
      ]}
    />
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

function MemoryCard({ memory, busy, onReview, t }) {
  const [showEvidence, setShowEvidence] = useState(false)
  const conf = memory.confidence != null ? Number(memory.confidence).toFixed(2) : null
  const hasEvidence = (memory.source_event_ids?.length || 0) > 0 || (memory.evidence_snapshot && Object.keys(memory.evidence_snapshot).length > 0)
  return (
    <div className="ref-card" style={{ cursor: 'default' }}>
      <div className="ref-meta" style={{ marginBottom: 8 }}>
        <span className="badge b-init">{memory.kind}</span>
        {conf !== null && <span className="ref-score" title={t('memory.confidence')}>{t('memory.conf', { n: conf })}</span>}
        {(memory.entities || []).map(e => <span key={e} className="chip">{e}</span>)}
      </div>

      <div style={{ fontSize: 13, lineHeight: 1.5, marginBottom: 10 }}>{memory.content}</div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button className="btn btn-green btn-sm" disabled={busy} onClick={() => onReview(memory, 'accept')}>
          {busy ? '⟳' : '✓'} {t('memory.accept')}
        </button>
        <button className="btn btn-red btn-sm" disabled={busy} onClick={() => onReview(memory, 'reject')}>
          ✕ {t('memory.reject')}
        </button>
        {hasEvidence && (
          <button className="btn btn-ghost btn-sm" onClick={() => setShowEvidence(v => !v)}>
            {showEvidence ? t('memory.hideEvidence') : t('memory.showEvidence')}
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
