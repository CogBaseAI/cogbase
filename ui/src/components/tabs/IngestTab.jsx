import React, { useState, useEffect, useRef } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { fmtBytes, waitForTasks } from '../../utils'
import DataTable from '../DataTable'

// Flatten a document's metadata dict to a "k: v, k: v" string for display.
function metaString(doc) {
  const meta = doc.metadata || {}
  return Object.keys(meta).length ? Object.entries(meta).map(([k, v]) => `${k}: ${v}`).join(', ') : ''
}

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
  const [deleting, setDeleting] = useState(null) // docId currently being deleted
  const [downloading, setDownloading] = useState(null) // docId currently being downloaded
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

  // Drop the previous app's docs the moment the selection changes (including
  // being deleted/cleared). The load effect below only fetches when the tab is
  // active or explicitly refreshed, so without this the deleted app's documents
  // would linger in the table.
  useEffect(() => {
    setDocs(null)
    setWfMaps({})
    setWfNames([])
    setAnyPendingWf(false)
    setUploadLog(null)
    setUploadErr(null)
  }, [currentApp])

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

  async function deleteDoc(docId) {
    if (!currentApp || deleting) return
    if (!confirm(t('ingest.confirmDelete', { docId }))) return
    setDeleting(docId)
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/docs/${encodeURIComponent(docId)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        loadIngestDocs()
      } else {
        alert(t('ingest.deleteFailed', { msg: resp.statusText }))
      }
    } catch (e) {
      alert(t('ingest.deleteFailed', { msg: e.message }))
    } finally {
      setDeleting(null)
    }
  }

  async function downloadDoc(doc) {
    if (!currentApp || downloading) return
    const docId = doc.doc_id
    setDownloading(docId)
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/docs/${encodeURIComponent(docId)}/original`)
      if (!resp.ok) { alert(t('ingest.downloadFailed', { msg: resp.status === 404 ? t('ingest.unknownError') : resp.statusText })); return }
      const blob = await resp.blob()
      const filename = (doc.metadata && doc.metadata.source_filename) || docId
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert(t('ingest.downloadFailed', { msg: e.message }))
    } finally {
      setDownloading(null)
    }
  }

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
            <DataTable
              rows={docs}
              rowKey={doc => doc.doc_id}
              rowClassName={doc => (wfNames.find(wf => ACTIVE_WF.has(wfMaps[wf]?.[doc.doc_id])) ? 'wf-pending' : undefined)}
              onRowClick={doc => {
                const pendingWf = wfNames.find(wf => ACTIVE_WF.has(wfMaps[wf]?.[doc.doc_id]))
                if (pendingWf) onOpenTaskProgress({ appName: currentApp, workflowName: pendingWf, docId: doc.doc_id })
              }}
              columns={[
                { key: 'doc_id', label: t('ingest.colDocId'), cellClassName: 'doc-id-cell', value: doc => doc.doc_id },
                { key: 'meta', label: t('ingest.colMeta'), value: doc => metaString(doc) },
                {
                  key: 'ingested_at', label: t('ingest.colIngestedAt'), sortValue: doc => doc.ingested_at || '',
                  value: doc => (doc.ingested_at ? doc.ingested_at.replace('T', ' ').slice(0, 16) : ''),
                },
                { key: 'ingest_status', label: t('ingest.colIngest'), text: doc => doc.status || '', render: doc => statusBadge(doc.status) },
                ...wfNames.map(wf => ({
                  key: `wf:${wf}`, label: wf, text: doc => wfMaps[wf]?.[doc.doc_id] || '',
                  render: doc => {
                    const wfStatus = wfMaps[wf]?.[doc.doc_id]
                    const canRun = wfStatus != null && wfStatus !== 'done' && !ACTIVE_WF.has(wfStatus)
                    return (
                      <>
                        {statusBadge(wfStatus)}
                        {canRun && (
                          <button className="ingest-run-btn" onClick={e => { e.stopPropagation(); openWfModalDirect(wf, doc.doc_id) }}>{t('ingest.run')}</button>
                        )}
                      </>
                    )
                  },
                })),
                {
                  key: 'actions', label: t('ingest.colActions'), sortable: false, cellClassName: 'actions-cell',
                  render: doc => (
                    <>
                      <button
                        className="btn btn-ghost btn-sm"
                        disabled={downloading === doc.doc_id}
                        onClick={e => { e.stopPropagation(); downloadDoc(doc) }}
                      >
                        {downloading === doc.doc_id ? <span className="spinning">⟳</span> : t('ingest.download')}
                      </button>
                      <button
                        className="btn btn-red btn-sm"
                        disabled={deleting === doc.doc_id}
                        onClick={e => { e.stopPropagation(); deleteDoc(doc.doc_id) }}
                      >
                        {deleting === doc.doc_id ? <span className="spinning">⟳</span> : t('common.delete')}
                      </button>
                    </>
                  ),
                },
              ]}
            />
          </div>
        )}
        {docs && docs.length === 0 && <p style={{ color: 'var(--muted)', fontSize: 12, marginTop: 20 }}>{t('ingest.noDocs')}</p>}
      </div>
    </>
  )
}
