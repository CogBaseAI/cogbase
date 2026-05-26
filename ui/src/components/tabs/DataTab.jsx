import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'

export default function DataTab({ active, onOpenWfModal, wfCompleteCollection, onWfCompleteHandled }) {
  const { apiUrl, currentApp, demoCatalog } = useApp()
  const [collections, setCollections] = useState([])
  const [activeCollection, setActiveCollectionState] = useState('')
  const [records, setRecords] = useState(null)
  const [cols, setCols] = useState([])
  const [rowCount, setRowCount] = useState('')
  const [loadingColl, setLoadingColl] = useState(false)
  const [collError, setCollError] = useState(null)
  const [pendingBar, setPendingBar] = useState(null) // null | {msg, btnLabel, pendingState}
  const hasApp = !!currentApp

  async function loadCollections() {
    if (!currentApp) { setCollections([]); return }
    setCollError(null)
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/collections`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { structured = [] } = await resp.json()
      setCollections(structured)
      if (structured.length > 0 && (!activeCollection || !structured.includes(activeCollection))) {
        selectCollection(structured[0])
      }
    } catch (e) { setCollError(e.message) }
  }

  useEffect(() => { if (active) loadCollections() }, [active, currentApp])

  // Refresh if WfModal completed for our active collection
  useEffect(() => {
    if (wfCompleteCollection && wfCompleteCollection === activeCollection) {
      selectCollection(activeCollection)
      onWfCompleteHandled()
    }
  }, [wfCompleteCollection])

  async function selectCollection(name) {
    setActiveCollectionState(name)
    setPendingBar(null)
    setLoadingColl(true)
    setRecords(null)
    setRowCount('')
    try {
      const resp = await fetch(
        `${apiUrl}/applications/${encodeURIComponent(currentApp)}/collections/${encodeURIComponent(name)}/query`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filters: [], fields: null }) }
      )
      if (!resp.ok) throw new Error(resp.status + ': ' + await resp.text())
      const { records: recs = [] } = await resp.json()
      setRowCount(recs.length + ' row' + (recs.length !== 1 ? 's' : ''))

      const colsSet = new Set()
      const colsList = []
      for (const row of recs) for (const k of Object.keys(row)) if (!colsSet.has(k)) { colsSet.add(k); colsList.push(k) }
      setCols(colsList)
      setRecords(recs)
      checkWorkflowPending(name)
    } catch (e) {
      setCollError(e.message)
    } finally {
      setLoadingColl(false)
    }
  }

  async function checkWorkflowPending(collName) {
    const demo = demoCatalog.find(d => d.name === currentApp)
    if (!demo || !demo.workflow_save_targets) return
    const target = demo.workflow_save_targets.find(t => t.save_collection === collName)
    if (!target) return
    const wf = (demo.workflow_actions || [])[target.workflow_action_index]
    if (!wf) return
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/tasks?task_type=workflow&task_name=${encodeURIComponent(wf.name)}&status=pending`)
      if (!resp.ok) return
      const { tasks = [] } = await resp.json()
      const pendingIds = [...new Set(tasks.map(t => t.doc_id).filter(Boolean))]
      if (!pendingIds.length) return
      setPendingBar({
        msg: `⚠ ${pendingIds.length} ${pendingIds.length === 1 ? 'doc has' : 'docs have'} pending workflow tasks — ${pendingIds.join(', ')}`,
        btnLabel: (wf.label || 'Run Workflow') + ' →',
        pendingState: { demoKey: demo.key, wfActionIndex: target.workflow_action_index, pendingIds },
      })
    } catch {}
  }

  async function openPendingWfModal() {
    if (!pendingBar?.pendingState) return
    const { demoKey, wfActionIndex, pendingIds } = pendingBar.pendingState
    const demo = demoCatalog.find(d => d.key === demoKey)
    if (!demo) return
    const wf = (demo.workflow_actions || [])[wfActionIndex]
    if (!wf) return
    const saveTarget = (demo.workflow_save_targets || []).find(t => t.workflow_action_index === wfActionIndex)
    onOpenWfModal({
      appName: demo.name,
      workflowName: wf.name,
      paramKey: wf.param_key,
      label: wf.label,
      paramLabel: wf.param_label,
      values: pendingIds,
      desc: `${pendingIds.length} ${wf.param_label?.toLowerCase() || 'doc'}${pendingIds.length === 1 ? ' has' : 's have'} pending tasks.`,
      allDone: false,
      saveCollection: saveTarget?.save_collection || null,
    })
  }

  const demo = demoCatalog.find(d => d.name === currentApp)
  const wfSaveColls = new Set((demo?.workflow_save_targets || []).map(t => t.save_collection))

  function renderCell(val) {
    if (val === null || val === undefined) return <td className="null-val">—</td>
    if (typeof val === 'object') {
      const str = JSON.stringify(val)
      if (str.length > 80) return <td className="long-cell obj-val"><LongVal text={str} /></td>
      return <td className="obj-val">{str}</td>
    }
    const str = String(val)
    if (str.length > 80) return <td className="long-cell"><LongVal text={str} /></td>
    return <td title={str}>{str}</td>
  }

  return (
    <>
      {!hasApp && <div className="warn-bar show">⚠ No app selected — go to Apps and click Use on an app first.</div>}
      <div className="data-layout">
        <div className="data-sidebar">
          <div className="data-sidebar-hd">
            <h3>Collections</h3>
            <button className="btn btn-ghost btn-sm" onClick={loadCollections} title="Refresh">⟳</button>
          </div>
          <div className="coll-list">
            {!hasApp && <div className="empty" style={{ padding: '30px 10px' }}><p>Select an app first.</p></div>}
            {collError && <div style={{ color: 'var(--red)', padding: 10, fontSize: 12 }}>{collError}</div>}
            {collections.map(name => (
              <div key={name} className={`coll-item${name === activeCollection ? ' active' : ''}`} onClick={() => selectCollection(name)}>
                <span className="coll-dot" />
                {name}
                {wfSaveColls.has(name) && <span className="coll-wf-badge">workflow</span>}
              </div>
            ))}
          </div>
        </div>
        <div className="data-main">
          <div className="data-main-hd">
            <h3 style={{ color: activeCollection ? '' : 'var(--muted)' }}>{activeCollection || 'No collection selected'}</h3>
            <span className="meta">{rowCount}</span>
          </div>
          {pendingBar && (
            <div className="wf-pending-bar show">
              <div className="wf-pending-bar-msg">{pendingBar.msg}</div>
              <span className="demo-wf-wrap" data-tip="Select this app under Apps first">
                <button className="btn btn-primary btn-sm demo-wf-btn" onClick={openPendingWfModal}>{pendingBar.btnLabel}</button>
              </span>
            </div>
          )}
          <div className="data-table-wrap">
            {!activeCollection && !loadingColl && <div className="empty"><p>Click a collection on the left to browse its records.</p></div>}
            {loadingColl && <div className="empty"><p><span className="spinning">⟳</span> Loading records…</p></div>}
            {!loadingColl && records && records.length === 0 && <div className="empty"><div className="ei">🗄️</div><p>No records yet — ingest some documents first.</p></div>}
            {!loadingColl && records && records.length > 0 && (
              <table className="data-tbl">
                <thead><tr>{cols.map(c => <th key={c} title={c}>{c}</th>)}</tr></thead>
                <tbody>
                  {records.map((row, i) => (
                    <tr key={i}>{cols.map(col => <React.Fragment key={col}>{renderCell(row[col])}</React.Fragment>)}</tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function LongVal({ text }) {
  const [expanded, setExpanded] = useState(false)
  return <div className={`long-val${expanded ? ' exp' : ''}`} onClick={() => setExpanded(v => !v)}>{text}</div>
}
