import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { copyText } from '../../utils'

export default function DataTab({ active, onOpenWfModal, wfCompleteCollection, onWfCompleteHandled }) {
  const { apiUrl, currentApp, demoCatalog } = useApp()
  const { t } = useT()
  const [collections, setCollections] = useState([])
  const [activeCollection, setActiveCollectionState] = useState('')
  const [records, setRecords] = useState(null)
  const [cols, setCols] = useState([])
  const [rowCount, setRowCount] = useState('')
  const [loadingColl, setLoadingColl] = useState(false)
  const [collError, setCollError] = useState(null)
  const [pendingBar, setPendingBar] = useState(null) // null | {msg, btnLabel, pendingState}
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState(null) // null | {col, dir: 'asc'|'desc'}
  const [detailRow, setDetailRow] = useState(null) // null | record object
  const [tip, setTip] = useState(null) // null | {text, x, y} — hover preview of a truncated cell
  const [colWidths, setColWidths] = useState({}) // {col -> px} — user-resized widths
  const [hiddenCols, setHiddenCols] = useState(() => new Set()) // columns toggled off
  const [showColMenu, setShowColMenu] = useState(false)
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

  // Drop the previous app's view the moment the selection changes (including
  // being deleted/cleared). The load effect below only fetches when the tab is
  // active, so without this the deleted app's records would linger in the table.
  useEffect(() => {
    setCollections([])
    setActiveCollectionState('')
    setRecords(null)
    setCols([])
    setRowCount('')
    setPendingBar(null)
    setCollError(null)
    setSearch('')
    setSort(null)
    setDetailRow(null)
    setColWidths({})
    setHiddenCols(new Set())
    setShowColMenu(false)
  }, [currentApp])

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
    setSearch('')
    setSort(null)
    setDetailRow(null)
    setColWidths({})
    setHiddenCols(new Set())
    setShowColMenu(false)
    try {
      const resp = await fetch(
        `${apiUrl}/applications/${encodeURIComponent(currentApp)}/collections/${encodeURIComponent(name)}/query`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filters: [], fields: null }) }
      )
      if (!resp.ok) throw new Error(resp.status + ': ' + await resp.text())
      const { records: recs = [] } = await resp.json()
      setRowCount(recs.length !== 1 ? t('data.rows', { n: recs.length }) : t('data.row', { n: recs.length }))

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
        msg: t('data.pendingMsg', { count: pendingIds.length, ids: pendingIds.join(', ') }),
        btnLabel: (wf.label || t('data.runWorkflow')) + ' →',
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
      desc: t('data.pendingDesc', { count: pendingIds.length, label: wf.param_label?.toLowerCase() || t('ingest.paramDocument') }),
      allDone: false,
      saveCollection: saveTarget?.save_collection || null,
    })
  }

  const demo = demoCatalog.find(d => d.name === currentApp)
  const wfSaveColls = new Set((demo?.workflow_save_targets || []).map(t => t.save_collection))

  // Flatten a cell value to a plain string for search/sort.
  function cellText(val) {
    if (val === null || val === undefined) return ''
    if (typeof val === 'object') return JSON.stringify(val)
    return String(val)
  }

  function toggleSort(col) {
    setSort(prev => {
      if (!prev || prev.col !== col) return { col, dir: 'asc' }
      if (prev.dir === 'asc') return { col, dir: 'desc' }
      return null // third click clears the sort
    })
  }

  // Columns the user has chosen to keep visible, in original order.
  const visibleCols = React.useMemo(() => cols.filter(c => !hiddenCols.has(c)), [cols, hiddenCols])

  function toggleCol(col) {
    setHiddenCols(prev => {
      const next = new Set(prev)
      if (next.has(col)) next.delete(col)
      else next.add(col)
      if (next.size >= cols.length) return prev // never hide the last column
      return next
    })
  }

  const allVisible = hiddenCols.size === 0
  function showAllCols() { setHiddenCols(new Set()) }
  // Deselecting all keeps the first column so the table is never empty.
  function hideAllCols() { setHiddenCols(new Set(cols.slice(1))) }

  // Drag the handle on a header's right edge to set an explicit width for that
  // column; widening a column lets its cells show more text before truncating.
  function startResize(col, e) {
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const startW = e.currentTarget.parentElement.getBoundingClientRect().width
    const onMove = ev => setColWidths(prev => ({ ...prev, [col]: Math.max(60, Math.round(startW + ev.clientX - startX)) }))
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      document.body.classList.remove('col-resizing')
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    document.body.classList.add('col-resizing')
  }

  // Close the column menu on an outside click.
  useEffect(() => {
    if (!showColMenu) return
    const onDoc = e => { if (!e.target.closest('.col-menu-wrap')) setShowColMenu(false) }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [showColMenu])

  // Filter (across visible columns) then sort — both client-side over loaded records.
  const viewRows = React.useMemo(() => {
    if (!records) return []
    const q = search.trim().toLowerCase()
    let rows = q
      ? records.filter(row => visibleCols.some(c => cellText(row[c]).toLowerCase().includes(q)))
      : records.slice()
    if (sort) {
      const { col, dir } = sort
      rows.sort((a, b) => {
        const av = a[col], bv = b[col]
        // Nulls always sort last, regardless of direction.
        const aNull = av === null || av === undefined
        const bNull = bv === null || bv === undefined
        if (aNull || bNull) return aNull === bNull ? 0 : aNull ? 1 : -1
        let cmp
        if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv
        else cmp = cellText(av).localeCompare(cellText(bv), undefined, { numeric: true, sensitivity: 'base' })
        return dir === 'asc' ? cmp : -cmp
      })
    }
    return rows
  }, [records, visibleCols, search, sort])

  const countLabel = records && search.trim()
    ? t('data.rowsFiltered', { shown: viewRows.length, total: records.length })
    : rowCount

  // Cells stay single-line (truncated with ellipsis). Long values reveal their
  // full content in a floating preview on hover; a row click opens the drawer.
  function renderCell(val, width) {
    const style = width ? { maxWidth: width, width } : undefined
    if (val === null || val === undefined) return <td className="null-val" style={style}>—</td>
    const isObj = typeof val === 'object'
    const str = isObj ? JSON.stringify(val) : String(val)
    if (str.length > 80) {
      const full = isObj ? JSON.stringify(val, null, 2) : str
      return (
        <td
          className={isObj ? 'obj-val' : ''}
          style={style}
          onMouseEnter={e => setTip({ text: full, x: e.clientX, y: e.clientY })}
          onMouseMove={e => setTip(prev => (prev ? { ...prev, x: e.clientX, y: e.clientY } : prev))}
          onMouseLeave={() => setTip(null)}
        >{str}</td>
      )
    }
    return <td className={isObj ? 'obj-val' : ''} style={style} title={str}>{str}</td>
  }

  return (
    <>
      {!hasApp && <div className="warn-bar show">{t('common.noAppWarn')}</div>}
      <div className="data-layout">
        <div className="data-sidebar">
          <div className="data-sidebar-hd">
            <h3>{t('data.collections')}</h3>
            <button className="btn btn-ghost btn-sm" onClick={loadCollections} title={t('data.refresh')}>⟳</button>
          </div>
          <div className="coll-list">
            {!hasApp && <div className="empty" style={{ padding: '30px 10px' }}><p>{t('data.selectApp')}</p></div>}
            {collError && <div style={{ color: 'var(--red)', padding: 10, fontSize: 12 }}>{collError}</div>}
            {collections.map(name => (
              <div key={name} className={`coll-item${name === activeCollection ? ' active' : ''}`} onClick={() => selectCollection(name)}>
                <span className="coll-dot" />
                {name}
                {wfSaveColls.has(name) && <span className="coll-wf-badge">{t('data.workflowBadge')}</span>}
              </div>
            ))}
          </div>
        </div>
        <div className="data-main">
          <div className="data-main-hd">
            <h3 style={{ color: activeCollection ? '' : 'var(--muted)' }}>{activeCollection || t('data.noCollection')}</h3>
            <div className="data-main-hd-right">
              {records && records.length > 0 && (
                <input
                  className="data-search"
                  type="search"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder={t('data.searchPlaceholder')}
                />
              )}
              {records && records.length > 0 && cols.length > 0 && (
                <div className="col-menu-wrap">
                  <button className="btn btn-ghost btn-sm" onClick={() => setShowColMenu(v => !v)}>
                    {t('data.columns')} ({visibleCols.length}/{cols.length}) ▾
                  </button>
                  {showColMenu && (
                    <div className="col-menu">
                      <div className="col-menu-actions">
                        <button className="col-menu-link" disabled={allVisible} onClick={showAllCols}>{t('data.selectAll')}</button>
                        <button className="col-menu-link" onClick={hideAllCols}>{t('data.deselectAll')}</button>
                      </div>
                      {cols.map(c => (
                        <label key={c} className="col-menu-item">
                          <input type="checkbox" checked={!hiddenCols.has(c)} onChange={() => toggleCol(c)} />
                          <span title={c}>{c}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <span className="meta">{countLabel}</span>
            </div>
          </div>
          {pendingBar && (
            <div className="wf-pending-bar show">
              <div className="wf-pending-bar-msg">{pendingBar.msg}</div>
              <span className="demo-wf-wrap" data-tip={t('data.tip')}>
                <button className="btn btn-primary btn-sm demo-wf-btn" onClick={openPendingWfModal}>{pendingBar.btnLabel}</button>
              </span>
            </div>
          )}
          <div className="data-table-wrap">
            {!activeCollection && !loadingColl && <div className="empty"><p>{t('data.browseHint')}</p></div>}
            {loadingColl && <div className="empty"><p><span className="spinning">⟳</span> {t('data.loadingRecords')}</p></div>}
            {!loadingColl && records && records.length === 0 && <div className="empty"><div className="ei">🗄️</div><p>{t('data.noRecords')}</p></div>}
            {!loadingColl && records && records.length > 0 && viewRows.length === 0 && (
              <div className="empty"><div className="ei">🔍</div><p>{t('data.noMatch')}</p></div>
            )}
            {!loadingColl && records && records.length > 0 && viewRows.length > 0 && (
              <table className="data-tbl">
                <thead>
                  <tr>
                    <th className="row-num-th">#</th>
                    {visibleCols.map(c => {
                      const dir = sort?.col === c ? sort.dir : null
                      const w = colWidths[c]
                      return (
                        <th
                          key={c}
                          className="sortable"
                          style={w ? { width: w, minWidth: w, maxWidth: w } : undefined}
                          onClick={() => toggleSort(c)}
                        >
                          <span className="th-label" title={c}>{c}</span>
                          <span className="sort-caret">{dir === 'asc' ? ' ▲' : dir === 'desc' ? ' ▼' : ''}</span>
                          <span className="col-resize-handle" onClick={e => e.stopPropagation()} onMouseDown={e => startResize(c, e)} />
                        </th>
                      )
                    })}
                  </tr>
                </thead>
                <tbody>
                  {viewRows.map((row, i) => (
                    <tr key={i} className="data-row" onClick={() => { setTip(null); setDetailRow(row) }}>
                      <td className="row-num">{i + 1}</td>
                      {visibleCols.map(col => <React.Fragment key={col}>{renderCell(row[col], colWidths[col])}</React.Fragment>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
      {detailRow && <RecordDetail row={detailRow} cols={cols} onClose={() => setDetailRow(null)} />}
      {tip && <CellTip text={tip.text} x={tip.x} y={tip.y} />}
    </>
  )
}

// Floating preview of a truncated cell, following the cursor and flipping away
// from the viewport edges. pointer-events:none (in CSS) keeps it from stealing
// the hover, so the cell's own mouseleave still fires.
function CellTip({ text, x, y }) {
  const W = 380, GAP = 14
  const left = x + GAP + W > window.innerWidth ? Math.max(8, x - GAP - W) : x + GAP
  const top = Math.min(y + GAP, Math.max(8, window.innerHeight - 8 - 240))
  return <div className="cell-tip" style={{ left, top, maxWidth: W }}>{text}</div>
}

// Slide-over drawer showing one record's fields in full, with pretty-printed
// JSON for object values and click-to-copy on each field.
function RecordDetail({ row, cols, onClose }) {
  const { t } = useT()
  const [copied, setCopied] = useState(null)

  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  function fmt(val) {
    if (val === null || val === undefined) return '—'
    if (typeof val === 'object') return JSON.stringify(val, null, 2)
    return String(val)
  }

  async function copy(key, val) {
    if (await copyText(fmt(val))) { setCopied(key); setTimeout(() => setCopied(null), 1500) }
  }

  return (
    <div className="record-detail-overlay" onClick={onClose}>
      <div className="record-detail" onClick={e => e.stopPropagation()}>
        <div className="record-detail-hd">
          <h3>{t('data.recordDetail')}</h3>
          <button className="btn btn-ghost btn-sm" onClick={onClose} title={t('data.close')}>✕</button>
        </div>
        <div className="record-detail-body">
          {cols.map(col => {
            const val = row[col]
            const isNull = val === null || val === undefined
            return (
              <div key={col} className="record-field">
                <div className="record-field-key">
                  <span>{col}</span>
                  <button className="record-copy-btn" onClick={() => copy(col, val)}>
                    {copied === col ? t('data.copied') : '⧉'}
                  </button>
                </div>
                <pre className={`record-field-val${isNull ? ' null-val' : ''}`}>{fmt(val)}</pre>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
