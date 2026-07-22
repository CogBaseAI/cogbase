import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { streamSSE } from '../../utils'

export default function WfModal({ state, onClose }) {
  const { appBase, authFetch } = useApp()
  const { t } = useT()
  const [selectedValue, setSelectedValue] = useState('')
  const [running, setRunning] = useState(false)
  const [findings, setFindings] = useState([])
  const [tally, setTally] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (state) {
      setSelectedValue((state.values || [])[0] || '')
      setFindings([])
      setTally(null)
      setError(null)
    }
  }, [state])

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape' && !running) onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose, running])

  if (!state) return null

  async function runWorkflow() {
    if (!selectedValue || running) return
    setRunning(true)
    setFindings([])
    setTally(null)
    setError(null)
    const localFindings = []
    try {
      const resp = await authFetch(
        `${appBase}/${encodeURIComponent(state.appName)}/workflows/${encodeURIComponent(state.workflowName)}/stream`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_id: selectedValue }) }
      )
      if (!resp.ok) { setError(t('wfModal.errStatus', { status: resp.status, msg: await resp.text() })); return }
      for await (const data of streamSSE(resp)) {
        if (data.error) { setError(t('common.error', { msg: data.error })); continue }
        const r = data.record || data
        localFindings.push(r)
        setFindings(prev => [...prev, r])
      }
      if (localFindings.some(f => f.status != null)) {
        const counts = {}
        for (const f of localFindings) {
          const s = f.status != null ? String(f.status).replace(/_/g, '-') : '(none)'
          counts[s] = (counts[s] || 0) + 1
        }
        const parts = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([s, n]) => `${s}: ${n}`)
        setTally(t('wfModal.tallyDetail', { n: localFindings.length, detail: parts.join(' · ') }))
      } else {
        setTally(t('wfModal.tally', { n: localFindings.length }))
      }
    } catch (e) {
      setError(t('common.error', { msg: e.message }))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="wf-modal show">
      <div className="wf-modal-panel" role="dialog" aria-modal="true">
        <div className="wf-modal-hd">
          <h3>{state.label}</h3>
          <button className="wf-modal-close" onClick={onClose} aria-label={t('wfModal.close')}>✕</button>
        </div>
        <div className="wf-modal-body">
          <p className="wf-modal-desc">{state.desc}</p>
          <div className="wf-modal-controls">
            <label>{state.paramLabel || t('wfModal.paramDocument')}:</label>
            <select value={selectedValue} onChange={e => setSelectedValue(e.target.value)} disabled={state.allDone || running}>
              {(state.values || []).map(v => <option key={v} value={v}>{v}</option>)}
            </select>
            <button className="btn btn-primary" disabled={state.allDone || running || !selectedValue} onClick={runWorkflow}>
              {running ? t('wfModal.running') : state.label}
            </button>
          </div>
          {(findings.length > 0 || error || tally) && (
            <div className="wf-output" style={{ display: 'flex' }}>
              {error && <div className="wf-tally" style={{ color: 'var(--red)' }}>{error}</div>}
              {findings.map((r, i) => <WfFinding key={i} record={r} />)}
              {tally && <div className="wf-tally">{tally}</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function WfFinding({ record }) {
  const statusVal = record.status != null ? String(record.status) : null
  const statusClass = statusVal ? statusVal.replace(/_/g, '-') : ''
  const fields = Object.entries(record).filter(([k, v]) => k !== 'status' && v != null)
  return (
    <div className="wf-finding">
      {statusVal && <span className={`wf-status ${statusClass}`}>{statusClass}</span>}
      {fields.map(([k, v]) => {
        const display = typeof v === 'object' ? JSON.stringify(v) : String(v)
        if (display.length > 80) {
          return <LongField key={k} label={k} value={display} />
        }
        return <span key={k} className="wf-field"><span className="wf-key">{k}:</span> {display}</span>
      })}
    </div>
  )
}

function LongField({ label, value }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="wf-long-field">
      <span className="wf-key">{label}:</span>
      <div className={`long-val${expanded ? ' exp' : ''}`} onClick={() => setExpanded(v => !v)}>{value}</div>
    </div>
  )
}
