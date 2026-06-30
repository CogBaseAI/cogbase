import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { previewText, metaText, waitForTasks } from '../../utils'

export default function DemosTab({ active, onOpenDocModal, onOpenConfigModal, onOpenWfModal, onSwitchTab }) {
  const { apiUrl, currentApp, setCurrentApp, demoCatalog, setDemoCatalog } = useApp()
  const { t } = useT()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [demoStatus, setDemoStatus] = useState(t('demos.status'))
  // steps: { [demoKey]: [{id, text, state}] }
  const [steps, setSteps] = useState({})
  // deploying: Set of demo keys currently being deployed
  const [deploying, setDeploying] = useState(new Set())

  async function loadDemos() {
    setLoading(true); setError(null)
    try {
      const resp = await fetch(`${apiUrl}/examples/demos`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const data = await resp.json()
      setDemoCatalog(data.demos || [])
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  useEffect(() => { if (active && !demoCatalog.length) loadDemos() }, [active])

  function addStep(demoKey, text, state = 'running') {
    const id = `${demoKey}-${Date.now()}-${Math.random()}`
    setSteps(prev => ({ ...prev, [demoKey]: [...(prev[demoKey] || []), { id, text, state }] }))
    return id
  }

  function updateStep(demoKey, id, text, state) {
    setSteps(prev => ({
      ...prev,
      [demoKey]: (prev[demoKey] || []).map(s => s.id === id ? { ...s, text, state } : s)
    }))
  }

  function clearSteps(demoKey) {
    setSteps(prev => ({ ...prev, [demoKey]: [] }))
  }

  async function deployDemo(key) {
    const demo = demoCatalog.find(d => d.key === key)
    if (!demo) return
    setDeploying(prev => new Set([...prev, key]))
    clearSteps(key)

    try {
      // Step 1: check/create app
      const appStepId = addStep(key, t('demos.stepCheckApp', { name: demo.name }))
      const appResp = await fetch(`${apiUrl}/applications/${encodeURIComponent(demo.name)}`)
      if (appResp.ok) {
        const app = await appResp.json()
        if (app.status !== 'active') {
          updateStep(key, appStepId, t('demos.stepNotActive', { name: demo.name }), 'error')
          throw new Error(t('demos.stepExistsErr', { name: demo.name }))
        }
        updateStep(key, appStepId, t('demos.stepExists', { name: demo.name }), 'done')
      } else if (appResp.status !== 404) {
        updateStep(key, appStepId, t('demos.stepCheckFail', { status: appResp.status }), 'error')
        throw new Error(t('demos.stepCheckFailErr', { status: appResp.status }))
      } else {
        updateStep(key, appStepId, t('demos.stepCreating', { name: demo.name }), 'running')
        const deployResp = await fetch(`${apiUrl}/generate/deploy`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config_yaml: demo.config_yaml }),
        })
        const deployData = await deployResp.json()
        if (!deployResp.ok) {
          updateStep(key, appStepId, t('demos.stepCreateFail', { msg: deployData.detail || deployResp.statusText }), 'error')
          throw new Error(deployData.detail || deployResp.statusText)
        }
        if (deployData.status !== 'active') {
          const msg = t('demos.stepDeployStatus', { status: `${deployData.status}${deployData.error ? ' — ' + deployData.error : ''}` })
          updateStep(key, appStepId, msg, 'error')
          throw new Error(msg)
        }
        updateStep(key, appStepId, t('demos.stepCreated', { name: demo.name }), 'done')
      }

      // Step 2: upload docs grouped by metadata
      const docs = demo.docs || []
      const uploadStepId = addStep(key, t('demos.stepUploading', { n: docs.length }))
      const metaGroups = {}
      for (const doc of docs) {
        const mk = JSON.stringify(doc.metadata || {})
        if (!metaGroups[mk]) metaGroups[mk] = []
        metaGroups[mk].push(doc)
      }
      const allIngestTaskIds = []
      for (const [metaJson, batchDocs] of Object.entries(metaGroups)) {
        const formData = new FormData()
        for (const doc of batchDocs) {
          formData.append('files', new Blob([doc.text || ''], { type: 'text/plain' }), doc.doc_id + '.txt')
        }
        formData.append('metadata', metaJson)
        const ingestResp = await fetch(`${apiUrl}/applications/${encodeURIComponent(demo.name)}/upload_documents`, { method: 'POST', body: formData })
        const ingestData = await ingestResp.json()
        if (!ingestResp.ok) {
          updateStep(key, uploadStepId, t('demos.stepUploadFail', { msg: ingestData.detail || ingestResp.statusText }), 'error')
          throw new Error(ingestData.detail || ingestResp.statusText)
        }
        allIngestTaskIds.push(...ingestData.task_ids)
      }
      updateStep(key, uploadStepId, t('demos.stepUploaded', { n: docs.length }), 'done')

      // Step 3: wait for ingest
      const total = allIngestTaskIds.length
      const ingestStepId = addStep(key, t('demos.stepIngesting', { done: 0, total }))
      const ingestTasks = await waitForTasks(apiUrl, demo.name, allIngestTaskIds, {
        timeout: 600000,
        onProgress: (done, n) => updateStep(key, ingestStepId, t('demos.stepIngesting', { done, total: n }), 'running'),
      })
      const okCount = ingestTasks.filter(t => t && t.status === 'done').length
      const failCount = total - okCount
      updateStep(key, ingestStepId, failCount > 0 ? t('demos.stepIngestedSome', { ok: okCount, total, fail: failCount }) : t('demos.stepIngested', { ok: okCount }), failCount === total ? 'error' : 'done')

      addStep(key, t('demos.stepReady', { title: demo.title }), 'done')
      setCurrentApp(demo.name)
      onSwitchTab('ingest')
    } catch (e) {
      addStep(key, t('demos.stepFailed', { msg: e.message }), 'error')
    } finally {
      setDeploying(prev => { const s = new Set(prev); s.delete(key); return s })
    }
  }

  async function openWfModal(demo, wfIndex, overrideValues) {
    const wf = (demo.workflow_actions || [])[wfIndex]
    if (!wf) return

    let values = overrideValues || wf.param_values || []
    let desc = null
    let allDone = false

    if (!overrideValues && currentApp === demo.name) {
      try {
        const [tasksResp, doneResp] = await Promise.all([
          fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/tasks?task_type=workflow&task_name=${encodeURIComponent(wf.name)}&status=pending`),
          fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/workflows/${encodeURIComponent(wf.name)}/docs?status=done`),
        ])
        const pendingIds = tasksResp.ok ? [...new Set(((await tasksResp.json()).tasks || []).map(t => t.doc_id).filter(Boolean))] : []
        const doneIds = doneResp.ok ? new Set(((await doneResp.json()).docs || []).map(d => d.doc_id).filter(Boolean)) : new Set()
        if (pendingIds.length) {
          values = pendingIds
          desc = t('demos.wfPending', { count: pendingIds.length, label: wf.param_label.toLowerCase() })
        } else if (doneIds.size > 0) {
          values = (wf.param_values || []).filter(v => !doneIds.has(v))
          if (values.length === 0) { allDone = true; desc = t('demos.wfAllDone') }
        }
      } catch {}
    }

    if (!desc) {
      desc = overrideValues
        ? t('demos.wfNotChecked', { count: overrideValues.length, label: wf.param_label.toLowerCase() })
        : (wf.description || t('demos.wfRun', { label: wf.label, paramLabel: wf.param_label.toLowerCase() }))
    }

    // Find saveCollection for this workflow
    const saveTarget = (demo.workflow_save_targets || []).find(t =>
      (demo.workflow_actions || [])[t.workflow_action_index]?.name === wf.name
    )

    onOpenWfModal({
      appName: demo.name,
      workflowName: wf.name,
      paramKey: wf.param_key,
      label: wf.label,
      paramLabel: wf.param_label,
      values,
      desc,
      allDone,
      saveCollection: saveTarget?.save_collection || null,
    })
  }

  const stepIcon = (state) => {
    if (state === 'running') return <span className="spinning demo-step-icon">⟳</span>
    if (state === 'done') return <span className="demo-step-icon">✓</span>
    if (state === 'error') return <span className="demo-step-icon">✗</span>
    return <span className="demo-step-icon">·</span>
  }

  return (
    <div className="page">
      <div className="page-hd">
        <div>
          <h2>{t('demos.title')}</h2>
          <p className="sub" style={{ margin: '6px 0 0' }}>{t('demos.sub')}</p>
        </div>
        <button className="btn btn-ghost" onClick={loadDemos}>{t('common.refresh')}</button>
      </div>
      <div className="demo-status">{demoStatus}</div>

      {loading && <div className="empty"><p><span className="spinning">⟳</span> {t('demos.loadingCatalog')}</p></div>}
      {error && <div className="empty"><p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: error })}</p></div>}
      {!loading && !error && demoCatalog.length === 0 && (
        <div className="empty"><p>{t('demos.emptyApi')}</p></div>
      )}
      {!loading && demoCatalog.length > 0 && (
        <div className="demo-grid">
          {demoCatalog.map(demo => (
            <div className="demo-card" key={demo.key}>
              <div>
                <div className="demo-kicker">{demo.name}</div>
                <h3>{demo.title}</h3>
              </div>
              <div className="demo-desc">{demo.description}</div>
              {demo.notes && <div className="demo-desc" style={{ color: 'var(--muted)' }}>{demo.notes}</div>}
              <div className="demo-badges">
                <span className="demo-badge">{t('demos.appLabel', { name: demo.name })}</span>
                <span className="demo-badge">{t('demos.docsCount', { n: (demo.docs || []).length })}</span>
              </div>
              <div className="demo-actions" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  <button className="btn btn-green" disabled={deploying.has(demo.key)} onClick={() => deployDemo(demo.key)}>
                    {deploying.has(demo.key) ? t('demos.working') : t('demos.deployIngest')}
                  </button>
                  <button className="btn btn-ghost" onClick={() => onOpenConfigModal(demo)}>{t('demos.viewConfig')}</button>
                </div>
                {(steps[demo.key] || []).length > 0 && (
                  <div className="demo-progress">
                    {(steps[demo.key] || []).map(s => (
                      <div key={s.id} className={`demo-step ${s.state}`}>
                        {stepIcon(s.state)}
                        <span className="demo-step-text">{s.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {(demo.query_examples || []).length > 0 && (
                <div className="demo-block">
                  <h4>{t('demos.queryIdeas')}</h4>
                  <div className="demo-badges">
                    {demo.query_examples.map((q, i) => <span key={i} className="demo-badge">{q}</span>)}
                  </div>
                </div>
              )}
              {(demo.docs || []).length > 0 && (
                <div className="demo-block">
                  <h4>{t('demos.documents')}</h4>
                  <div className="demo-docs">
                    {demo.docs.map(doc => (
                      <div className="demo-doc" key={doc.doc_id}>
                        <div className="demo-doc-hd">
                          <div className="demo-doc-id">{doc.doc_id}</div>
                          <div className="demo-doc-meta">{metaText(doc.metadata)}</div>
                        </div>
                        <div className="demo-doc-preview">{previewText(doc.text)}</div>
                        <div className="demo-doc-actions">
                          <button className="btn btn-ghost btn-sm" onClick={() => onOpenDocModal({ demoKey: demo.key, demoName: demo.name, docId: doc.doc_id, meta: doc.metadata || {}, text: doc.text || '' })}>{t('demos.viewText')}</button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="demo-block">
                <h4>{t('demos.configPreview')}</h4>
                <pre className="demo-pre">{previewText(demo.config_yaml || '', 18, 1600)}</pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
