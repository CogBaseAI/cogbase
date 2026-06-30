import React, { useState, useEffect, useRef } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { fmtBytes, waitForTasks } from '../../utils'

export default function IngestTab({ active, refreshKey, onOpenTaskProgress, onOpenWfModal }) {
  const { apiUrl, currentApp } = useApp()
  const { t } = useT()
  const [pickedFiles, setPickedFiles] = useState([])
  const [metaInput, setMetaInput] = useState('{}')
  const [uploading, setUploading] = useState(false)
  const [uploadLog, setUploadLog] = useState(null) // null | [{status, docId, error}]
  const [uploadErr, setUploadErr] = useState(null)
  const [docs, setDocs] = useState(null)
  const [wfMaps, setWfMaps] = useState({}) // {wfName -> {docId -> wfStatus}}
  const [wfNames, setWfNames] = useState([])
  const [anyPendingWf, setAnyPendingWf] = useState(false)
  const fileInputRef = useRef(null)
  const hasApp = !!currentApp

  async function loadIngestDocs() {
    if (!currentApp) { setDocs(null); return }
    try {
      const [docsResp, wfResp] = await Promise.all([
        fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/docs`),
        fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/workflows`),
      ])
      if (!docsResp.ok) { setDocs([]); return }
      const { docs: docList = [] } = await docsResp.json()
      let names = []
      if (wfResp.ok) { const d = await wfResp.json(); names = d.workflows || [] }
      setWfNames(names)

      if (names.length > 0) {
        const wfResults = await Promise.all(
          names.map(wf => fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/workflows/${encodeURIComponent(wf)}/docs`).then(r => r.ok ? r.json() : { docs: [] }))
        )
        const maps = {}
        names.forEach((wf, i) => {
          maps[wf] = {}
          ;(wfResults[i].docs || []).forEach(d => { maps[wf][d.doc_id] = d.workflow_status })
        })
        setWfMaps(maps)
        const ACTIVE = new Set(['pending', 'running'])
        setAnyPendingWf(docList.some(doc => names.some(wf => ACTIVE.has(maps[wf]?.[doc.doc_id]))))
      } else {
        setWfMaps({})
        setAnyPendingWf(false)
      }
      setDocs(docList)
    } catch (e) {
      setDocs([])
    }
  }

  useEffect(() => { if (active || refreshKey > 0) loadIngestDocs() }, [active, currentApp, refreshKey])

  function handleFiles(fileList) {
    setPickedFiles(Array.from(fileList))
    setUploadLog(null)
    setUploadErr(null)
  }

  async function uploadFiles() {
    if (!currentApp || !pickedFiles.length) return
    let meta = {}
    try { meta = JSON.parse(metaInput || '{}') }
    catch { alert(t('ingest.invalidJson')); return }
    setUploading(true)
    setUploadLog(null)
    setUploadErr(null)
    try {
      const form = new FormData()
      pickedFiles.forEach(f => form.append('files', f))
      form.append('metadata', JSON.stringify(meta))
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/upload_documents`, { method: 'POST', body: form })
      if (!resp.ok) { setUploadErr(t('ingest.errStatus', { status: resp.status, msg: await resp.text() })); return }
      const uploadBody = await resp.json()
      setUploadLog([{ status: 'processing' }])
      const tasks = await waitForTasks(apiUrl, currentApp, uploadBody.task_ids, { timeout: 300000 })
      setUploadLog(tasks.map(t => ({ status: t.status, docId: t.doc_id || t.task_id, error: t.error })))
      setPickedFiles([])
      if (fileInputRef.current) fileInputRef.current.value = ''
      loadIngestDocs()
    } catch (e) {
      setUploadErr(t('common.networkError', { msg: e.message }))
    } finally {
      setUploading(false)
    }
  }

  const ACTIVE_WF = new Set(['pending', 'running'])

  function openWfModalDirect(wf, docId) {
    onOpenWfModal({
      appName: currentApp,
      workflowName: wf,
      paramKey: 'doc_id',
      label: wf,
      paramLabel: t('ingest.paramDocument'),
      values: [docId],
      desc: t('ingest.wfDesc', { docId }),
      allDone: false,
      fromIngest: true,
    })
  }

  const statusBadge = (s, fallback = '') => {
    const label = s || fallback
    return label ? <span className={`ingest-status ${label}`}>{label}</span> : <span className="ingest-status none">—</span>
  }

  return (
    <>
      {!hasApp && <div className="warn-bar show">{t('common.noAppWarn')}</div>}
      <div className="ingest-wrap">
        <h2>{t('ingest.title')}</h2>
        <p className="sub">{t('ingest.sub')}</p>

        <div
          className="drop-zone"
          onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add('over') }}
          onDragLeave={e => e.currentTarget.classList.remove('over')}
          onDrop={e => { e.preventDefault(); e.currentTarget.classList.remove('over'); handleFiles(e.dataTransfer.files) }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" multiple style={{ display: 'none' }} onChange={e => handleFiles(e.target.files)} />
          <div className="drop-icon">📄</div>
          <div className="drop-txt">{t('ingest.drop')}</div>
          <div className="drop-hint">{t('ingest.dropHint')}</div>
        </div>

        <div className="chips">
          {pickedFiles.map((f, i) => (
            <span key={i} className="chip">📄 {f.name} <span style={{ color: 'var(--muted)' }}>({fmtBytes(f.size)})</span></span>
          ))}
        </div>

        <div style={{ marginTop: 18 }}>
          <label className="field-label">{t('ingest.metaLabel')}</label>
          <textarea className="field-textarea" rows={2} value={metaInput} onChange={e => setMetaInput(e.target.value)} placeholder='{"doc_type": "contract"}' />
        </div>

        <div style={{ marginTop: 14 }}>
          <button className="btn btn-green" disabled={!hasApp || !pickedFiles.length || uploading} onClick={uploadFiles}>
            {uploading ? t('ingest.uploading') : t('ingest.uploadIngest')}
          </button>
        </div>

        {uploading && <p style={{ color: 'var(--muted)', fontSize: 12, marginTop: 16 }}><span className="spinning">⟳</span> {t('ingest.processing')}</p>}
        {uploadErr && <p style={{ color: 'var(--red)', fontSize: 12, marginTop: 16 }}>{uploadErr}</p>}
        {uploadLog && Array.isArray(uploadLog) && uploadLog[0]?.status !== 'processing' && (
          <div style={{ marginTop: 20 }}>
            {uploadLog.map((task, i) => (
              <div key={i} className="result-row">
                <span style={{ color: task.status === 'done' ? 'var(--green)' : 'var(--red)' }}>{task.status === 'done' ? '✓' : '✗'}</span>
                <span style={{ fontWeight: 500 }}>{task.docId}</span>
                {task.status !== 'done' && <span style={{ color: 'var(--red)' }}>{task.error || t('ingest.unknownError')}</span>}
              </div>
            ))}
          </div>
        )}

        {docs && docs.length > 0 && (
          <div id="ingestDocs">
            <div className="ingest-docs-hd">
              <h3>{t('ingest.ingestedDocs')}</h3>
              <button className="btn btn-ghost btn-sm" onClick={loadIngestDocs}>{t('common.refresh')}</button>
            </div>
            {anyPendingWf && (
              <div className="ingest-wf-alert show">
                {t('ingest.wfAlert')}
              </div>
            )}
            <div className="ingest-docs-wrap">
              <table>
                <thead>
                  <tr>
                    <th>{t('ingest.colDocId')}</th><th>{t('ingest.colMeta')}</th><th>{t('ingest.colIngestedAt')}</th><th>{t('ingest.colIngest')}</th>
                    {wfNames.map(wf => <th key={wf}>{wf}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {docs.map(doc => {
                    const meta = doc.metadata || {}
                    const metaStr = Object.keys(meta).length ? Object.entries(meta).map(([k, v]) => `${k}: ${v}`).join(', ') : ''
                    const pendingWf = wfNames.find(wf => ACTIVE_WF.has(wfMaps[wf]?.[doc.doc_id]))
                    return (
                      <tr
                        key={doc.doc_id}
                        className={pendingWf ? 'wf-pending' : ''}
                        onClick={pendingWf ? () => onOpenTaskProgress({ appName: currentApp, workflowName: pendingWf, docId: doc.doc_id }) : undefined}
                      >
                        <td className="doc-id-cell" title={doc.doc_id}>{doc.doc_id}</td>
                        <td title={metaStr}>{metaStr}</td>
                        <td>{doc.ingested_at ? doc.ingested_at.replace('T', ' ').slice(0, 16) : ''}</td>
                        <td>{statusBadge(doc.status)}</td>
                        {wfNames.map(wf => {
                          const wfStatus = wfMaps[wf]?.[doc.doc_id]
                          const canRun = wfStatus != null && wfStatus !== 'done' && !ACTIVE_WF.has(wfStatus)
                          return (
                            <td key={wf}>
                              {statusBadge(wfStatus)}
                              {canRun && (
                                <button className="ingest-run-btn" onClick={e => { e.stopPropagation(); openWfModalDirect(wf, doc.doc_id) }}>{t('ingest.run')}</button>
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {docs && docs.length === 0 && <p style={{ color: 'var(--muted)', fontSize: 12, marginTop: 20 }}>{t('ingest.noDocs')}</p>}
      </div>
    </>
  )
}
