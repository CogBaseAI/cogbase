import React, { useState, useEffect, useCallback } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'

// Slide-over drawer that renders an application's resolved config —
// vector/structured collections (with record schemas), pipelines and their
// extraction steps, and workflows — so a deployed app can be inspected
// without re-downloading its bundle. The config is already fully resolved
// (file refs inlined) on the list response, so this is a pure client render.
export default function AppDetailModal({ app, onClose }) {
  const { t } = useT()

  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!app) return null
  const cfg = app.config || {}
  const vecs = cfg.vector_collections || []
  const structs = cfg.structured_collections || []
  const pipelines = cfg.pipelines || []
  const workflows = cfg.workflows || []
  const sc = app.status === 'active' ? 'b-active' : app.status === 'error' ? 'b-error' : 'b-init'

  return (
    <div className="record-detail-overlay" onClick={onClose}>
      <div className="record-detail app-detail" onClick={e => e.stopPropagation()}>
        <div className="record-detail-hd">
          <h3>{t('appDetail.title', { name: app.name })}</h3>
          <button className="btn btn-ghost btn-sm" onClick={onClose} title={t('data.close')}>✕</button>
        </div>
        <div className="record-detail-body">
          <Section title={t('appDetail.overview')} defaultOpen>
            <div className="ad-kv"><span>{t('apps.colStatus')}</span><span className={`badge ${sc}`}>{app.status}</span></div>
            {app.error && <div className="ad-kv"><span>{t('appDetail.error')}</span><span style={{ color: 'var(--red)' }}>{app.error}</span></div>}
            <div className="ad-kv"><span>{t('apps.colCreated')}</span><span>{app.created_at ? new Date(app.created_at).toLocaleString() : '—'}</span></div>
            {app.updated_at && <div className="ad-kv"><span>{t('appDetail.updated')}</span><span>{new Date(app.updated_at).toLocaleString()}</span></div>}
            <div className="ad-counts">
              <span>{t('appDetail.countVectors', { n: vecs.length })}</span>
              <span>{t('appDetail.countStructured', { n: structs.length })}</span>
              <span>{t('appDetail.countPipelines', { n: pipelines.length })}</span>
              <span>{t('appDetail.countWorkflows', { n: workflows.length })}</span>
            </div>
          </Section>

          {vecs.length > 0 && (
            <Section title={t('appDetail.vectorCollections')}>
              {vecs.map(v => (
                <div key={v.name} className="ad-card">
                  <div className="ad-card-hd">{v.name}</div>
                  {v.description && <div className="ad-desc">{v.description}</div>}
                </div>
              ))}
            </Section>
          )}

          {structs.length > 0 && (
            <Section title={t('appDetail.structuredCollections')}>
              {structs.map(s => (
                <div key={s.name} className="ad-card">
                  <div className="ad-card-hd">{s.name}</div>
                  {s.description && <div className="ad-desc">{s.description}</div>}
                  {(s.primary_fields || []).length > 0 && (
                    <div className="ad-tags">
                      {s.primary_fields.map(f => <span key={f} className="ad-tag">{f}</span>)}
                    </div>
                  )}
                  {s.schema && <Collapsible label={t('appDetail.recordSchema')}><JsonBlock value={s.schema} /></Collapsible>}
                </div>
              ))}
            </Section>
          )}

          {pipelines.length > 0 && (
            <Section title={t('appDetail.pipelines')}>
              {pipelines.map(p => (
                <div key={p.name} className="ad-card">
                  <div className="ad-card-hd">{p.name}</div>
                  {p.routing_description && <div className="ad-desc">{p.routing_description}</div>}
                  {(p.steps || []).map((step, i) => <PipelineStep key={i} step={step} t={t} />)}
                </div>
              ))}
            </Section>
          )}

          {workflows.length > 0 && (
            <Section title={t('appDetail.workflows')}>
              {workflows.map(w => (
                <div key={w.name} className="ad-card">
                  <div className="ad-card-hd">{w.name}</div>
                  <div className="ad-tags">
                    <span className="ad-tag">{t('appDetail.trigger')}: {w.trigger?.type || 'manual'}</span>
                    {w.params_from_collection?.collection && <span className="ad-tag">{t('appDetail.paramsFrom')}: {w.params_from_collection.collection}</span>}
                  </div>
                  {(w.steps || []).map((step, i) => <WorkflowStep key={i} step={step} t={t} />)}
                </div>
              ))}
            </Section>
          )}

          <AppSkillsSection appName={app.name} />
        </div>
      </div>
    </div>
  )
}

function Section({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="ad-section">
      <button className="ad-section-hd" onClick={() => setOpen(o => !o)}>
        <span className="ad-caret">{open ? '▾' : '▸'}</span>{title}
      </button>
      {open && <div className="ad-section-body">{children}</div>}
    </div>
  )
}

// Manageable skills section: lists the skills assigned to this application and
// lets the user assign more (from the system skill registry) or unassign them,
// hitting the /applications/{name}/skills endpoints. Assigned skills are shown
// by display name (resolved server-side) rather than raw skill ids.
function AppSkillsSection({ appName }) {
  const { apiUrl } = useApp()
  const { t } = useT()
  const [all, setAll] = useState([])           // all system skills: [{ id, name, ... }]
  const [assigned, setAssigned] = useState(null) // assigned refs [{ id, name, missing }], null=loading
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(null)        // skill id in-flight, or '__add__'
  const [toAdd, setToAdd] = useState('')        // name selected in the add picker

  const load = useCallback(async () => {
    setError(null)
    try {
      const [skillsResp, assignedResp] = await Promise.all([
        fetch(`${apiUrl}/skills`),
        fetch(`${apiUrl}/applications/${encodeURIComponent(appName)}/skills`),
      ])
      if (skillsResp.ok) { const { skills = [] } = await skillsResp.json(); setAll(skills) }
      if (!assignedResp.ok) throw new Error(assignedResp.status + ' ' + assignedResp.statusText)
      const { skills: refs = [] } = await assignedResp.json()
      setAssigned(refs)
    } catch (e) { setError(e.message); setAssigned([]) }
  }, [apiUrl, appName])

  useEffect(() => { load() }, [load])

  async function assign(name) {
    if (!name) return
    setBusy('__add__')
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(appName)}/skills`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skill_name: name }),
      })
      if (!resp.ok && resp.status !== 201) {
        const d = await resp.json().catch(() => ({}))
        alert(t('skills.assignFailed', { msg: d.detail || resp.statusText })); return
      }
      const { skills: refs = [] } = await resp.json()
      setAssigned(refs)
      setToAdd('')
    } catch (e) { alert(t('common.error', { msg: e.message })) }
    finally { setBusy(null) }
  }

  // Unassign by skill id — works for live skills and for dangling (missing) refs
  // whose display name can no longer be resolved server-side.
  async function unassign(ref) {
    setBusy(ref.id)
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(appName)}/skills/${encodeURIComponent(ref.id)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        setAssigned(prev => (prev || []).filter(r => r.id !== ref.id))
      } else {
        const d = await resp.json().catch(() => ({}))
        alert(t('skills.unassignFailed', { msg: d.detail || resp.statusText }))
      }
    } catch (e) { alert(t('common.error', { msg: e.message })) }
    finally { setBusy(null) }
  }

  const assignedSet = new Set((assigned || []).map(r => r.name))
  const addable = all.filter(s => !assignedSet.has(s.name))
  const descByName = Object.fromEntries(all.map(s => [s.name, s.description || '']))
  const count = assigned?.length || 0

  const heading = (
    <>
      {t('appDetail.skills')}
      {assigned !== null && count > 0 && <span className="ad-count-badge">{count}</span>}
    </>
  )

  return (
    <Section title={heading} defaultOpen>
      {error && <div className="ad-desc" style={{ color: 'var(--red)' }}>{t('common.failed', { msg: error })}</div>}
      {assigned === null && !error && <div className="ad-desc"><span className="spinning">⟳</span> {t('common.loading')}</div>}
      {assigned !== null && (
        <>
          {assigned.length === 0
            ? <div className="ad-desc">{t('appDetail.noSkillsAssigned')}</div>
            : (
              <div className="ad-tags">
                {assigned.map(ref => {
                  const removing = busy === ref.id
                  return (
                    <span
                      key={ref.id}
                      className={`ad-tag ad-tag-removable${ref.missing ? ' ad-tag-broken' : ''}`}
                      title={ref.missing ? t('skills.brokenRef', { id: ref.id }) : (descByName[ref.name] || undefined)}
                    >
                      {ref.missing ? t('skills.missingLabel', { name: ref.name }) : ref.name}
                      <button
                        className="ad-tag-x" disabled={removing}
                        title={ref.missing ? t('skills.removeBroken') : t('skills.removeFromApp')}
                        onClick={() => unassign(ref)}
                      >{removing ? <span className="spinning">⟳</span> : '✕'}</button>
                    </span>
                  )
                })}
              </div>
            )}
          {all.length === 0
            ? <div className="ad-desc" style={{ marginTop: 10 }}>{t('appDetail.noSkillsInRegistry')}</div>
            : (
              <>
                <div className="ad-skill-add">
                  <select
                    value={toAdd} onChange={e => setToAdd(e.target.value)}
                    disabled={busy === '__add__' || addable.length === 0}
                  >
                    <option value="">{addable.length === 0 ? t('appDetail.noSkillsToAdd') : t('appDetail.assignSkillPrompt')}</option>
                    {addable.map(s => <option key={s.id} value={s.name} title={s.description || undefined}>{s.name}</option>)}
                  </select>
                  <button
                    className="btn btn-sm btn-primary"
                    disabled={!toAdd || busy === '__add__'} onClick={() => assign(toAdd)}
                  >{busy === '__add__' ? <span className="spinning">⟳</span> : t('appDetail.addSkill')}</button>
                </div>
                {toAdd && descByName[toAdd] && <div className="ad-desc" style={{ marginTop: 6 }}>{descByName[toAdd]}</div>}
              </>
            )}
        </>
      )}
    </Section>
  )
}

function Collapsible({ label, children }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="ad-collapsible">
      <button className="ad-collapsible-hd" onClick={() => setOpen(o => !o)}>
        <span className="ad-caret">{open ? '▾' : '▸'}</span>{label}
      </button>
      {open && <div className="ad-collapsible-body">{children}</div>}
    </div>
  )
}

// Render a resolved JSON schema as an interactive collapsible tree (like a
// browser's JSON viewer), toggleable to raw pretty-printed text. Falls back to
// raw display if the value isn't valid JSON.
function JsonBlock({ value }) {
  const { t } = useT()
  const [rawView, setRawView] = useState(false)
  const [copied, setCopied] = useState(false)

  let parsed
  let parsable = true
  try {
    parsed = typeof value === 'string' ? JSON.parse(value) : value
  } catch { parsable = false }

  const prettyText = parsable ? JSON.stringify(parsed, null, 2) : String(value)

  async function copy() {
    try { await navigator.clipboard.writeText(prettyText); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch {}
  }

  return (
    <div className="ad-json">
      <div className="ad-json-bar">
        {parsable && (
          <button className="ad-json-btn" onClick={() => setRawView(v => !v)}>
            {rawView ? t('appDetail.jsonTree') : t('appDetail.jsonRaw')}
          </button>
        )}
        <button className="ad-json-btn" onClick={copy}>{copied ? t('data.copied') : t('appDetail.jsonCopy')}</button>
      </div>
      {parsable && !rawView
        ? <div className="ad-code json-tree"><JsonNode value={parsed} depth={0} isLast /></div>
        : <pre className="ad-code">{prettyText}</pre>}
    </div>
  )
}

function JsonLeaf({ value }) {
  if (value === null) return <span className="json-null">null</span>
  const ty = typeof value
  if (ty === 'string') return <span className="json-string">{JSON.stringify(value)}</span>
  if (ty === 'number') return <span className="json-number">{String(value)}</span>
  if (ty === 'boolean') return <span className="json-boolean">{String(value)}</span>
  return <span>{String(value)}</span>
}

// One node in the JSON tree. Objects/arrays render a clickable row that
// expands to show their children; primitives render inline. Nodes default to
// expanded for the first couple of levels.
function JsonNode({ name, value, depth, isLast }) {
  const [open, setOpen] = useState(depth < 2)
  const isObj = value !== null && typeof value === 'object'
  const comma = isLast ? null : <span className="json-punct">,</span>
  const indent = { paddingLeft: depth * 14 }
  const keyPart = name != null && (
    <><span className="json-key">{JSON.stringify(name)}</span><span className="json-punct">: </span></>
  )

  if (!isObj) {
    return <div className="json-row" style={indent}><span className="json-toggle" />{keyPart}<JsonLeaf value={value} />{comma}</div>
  }

  const isArray = Array.isArray(value)
  const entries = isArray ? value.map((v, i) => [i, v]) : Object.entries(value)
  const open_br = isArray ? '[' : '{'
  const close_br = isArray ? ']' : '}'

  if (entries.length === 0) {
    return <div className="json-row" style={indent}><span className="json-toggle" />{keyPart}<span className="json-punct">{open_br}{close_br}</span>{comma}</div>
  }

  return (
    <div className="json-node">
      <div className="json-row json-clickable" style={indent} onClick={() => setOpen(o => !o)}>
        <span className="json-toggle">{open ? '▾' : '▸'}</span>
        {keyPart}
        <span className="json-punct">{open_br}</span>
        {!open && <span className="json-punct">…{close_br}</span>}
        {!open && <span className="json-count">{isArray ? `${entries.length}` : `${entries.length} keys`}</span>}
        {!open && comma}
      </div>
      {open && entries.map(([k, v], i) => (
        <JsonNode key={k} name={isArray ? null : k} value={v} depth={depth + 1} isLast={i === entries.length - 1} />
      ))}
      {open && <div className="json-row" style={indent}><span className="json-toggle" /><span className="json-punct">{close_br}</span>{comma}</div>}
    </div>
  )
}

function PipelineStep({ step, t }) {
  const ex = step.extractor
  return (
    <div className="ad-step">
      <div className="ad-step-hd">
        <span className="ad-step-tool">{step.tool}</span>
        {step.collection && <span className="ad-step-coll">→ {step.collection}</span>}
      </div>
      {ex && (
        <>
          <div className="ad-tags">
            {ex.record_mode && <span className="ad-tag">{t('appDetail.mode')}: {ex.record_mode}</span>}
            {ex.response_field && <span className="ad-tag">{t('appDetail.field')}: {ex.response_field}</span>}
            {ex.id_field && <span className="ad-tag">id: {ex.id_field}</span>}
          </div>
          {ex.extraction_schema && <Collapsible label={t('appDetail.extractionSchema')}><JsonBlock value={ex.extraction_schema} /></Collapsible>}
          {ex.prompt && <Collapsible label={t('appDetail.prompt')}><pre className="ad-code">{ex.prompt}</pre></Collapsible>}
        </>
      )}
      {step.doc_prompt && <Collapsible label={t('appDetail.docPrompt')}><pre className="ad-code">{step.doc_prompt}</pre></Collapsible>}
    </div>
  )
}

function WorkflowStep({ step, t }) {
  if (step.foreach !== undefined) {
    return (
      <div className="ad-step">
        <div className="ad-step-hd">
          <span className="ad-step-tool">foreach</span>
          <span className="ad-step-coll">{step.foreach}</span>
        </div>
        {(step.steps || []).map((s, i) => <WorkflowStep key={i} step={s} t={t} />)}
      </div>
    )
  }
  return (
    <div className="ad-step">
      <div className="ad-step-hd">
        <span className="ad-step-tool">{step.tool}</span>
        {step.id && <span className="ad-step-coll">{step.id}</span>}
        {step.collection && <span className="ad-step-coll">→ {step.collection}</span>}
      </div>
      {step.prompt && <Collapsible label={t('appDetail.prompt')}><pre className="ad-code">{step.prompt}</pre></Collapsible>}
      {step.output_schema && <Collapsible label={t('appDetail.outputSchema')}><JsonBlock value={step.output_schema} /></Collapsible>}
    </div>
  )
}
