import React, { useState, useEffect, useRef } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import DataTable from '../DataTable'

export default function SkillsTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const { t } = useT()
  const [skills, setSkills] = useState(null)   // null=loading, []|[...]=loaded
  const [error, setError] = useState(null)
  const [assigned, setAssigned] = useState(new Set()) // skill names assigned to currentApp
  const [picked, setPicked] = useState(null)   // File | null (new upload)
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState({ text: '', cls: '' })
  const [busyId, setBusyId] = useState(null)   // skill id with an in-flight action
  const uploadInputRef = useRef(null)
  const replaceInputRef = useRef(null)
  const replaceTargetRef = useRef(null)        // skill id awaiting a replacement file

  async function loadSkills() {
    setSkills(null); setError(null)
    try {
      const resp = await fetch(`${apiUrl}/skills`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { skills: list = [] } = await resp.json()
      setSkills(list)
    } catch (e) { setError(e.message) }
  }

  async function loadAssigned() {
    if (!currentApp) { setAssigned(new Set()); return }
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(currentApp)}/skills`)
      if (!resp.ok) { setAssigned(new Set()); return }
      const { skills: refs = [] } = await resp.json()
      setAssigned(new Set(refs.map(r => r.name)))
    } catch { setAssigned(new Set()) }
  }

  // Drop the previous app's assignments when the selection changes/clears, so a
  // deleted app's skills don't stay marked "on" until the tab is reopened.
  useEffect(() => { setAssigned(new Set()) }, [currentApp])

  useEffect(() => { if (active) { loadSkills(); loadAssigned() } }, [active, currentApp])

  async function uploadSkill() {
    if (!picked) return
    setUploading(true)
    setUploadMsg({ text: t('skills.uploadingMsg'), cls: '' })
    try {
      const form = new FormData()
      form.append('bundle', picked)
      const resp = await fetch(`${apiUrl}/skills`, { method: 'POST', body: form })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        setUploadMsg({ text: (Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || t('skills.errStatus', { status: resp.status }), cls: 'err' })
        return
      }
      const skill = await resp.json()
      setUploadMsg({ text: t('skills.uploaded', { name: skill.name }), cls: 'ok' })
      setPicked(null)
      if (uploadInputRef.current) uploadInputRef.current.value = ''
      loadSkills()
    } catch (e) {
      setUploadMsg({ text: t('common.networkError', { msg: e.message }), cls: 'err' })
    } finally {
      setUploading(false)
    }
  }

  function startReplace(skillName) {
    replaceTargetRef.current = skillName
    replaceInputRef.current?.click()
  }

  async function doReplace(file) {
    const skillName = replaceTargetRef.current
    replaceTargetRef.current = null
    if (replaceInputRef.current) replaceInputRef.current.value = ''
    if (!skillName || !file) return
    setBusyId(skillName)
    try {
      const form = new FormData()
      form.append('bundle', file)
      const resp = await fetch(`${apiUrl}/skills/${encodeURIComponent(skillName)}`, { method: 'PUT', body: form })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        alert(t('skills.replaceFailed', { msg: (Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || resp.status }))
        return
      }
      loadSkills()
    } catch (e) { alert(t('common.error', { msg: e.message })) }
    finally { setBusyId(null) }
  }

  async function deleteSkill(skill) {
    if (!confirm(t('skills.confirmDelete', { name: skill.name }))) return
    setBusyId(skill.name)
    try {
      const resp = await fetch(`${apiUrl}/skills/${encodeURIComponent(skill.name)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        loadSkills()
        loadAssigned()
      } else {
        alert(t('skills.deleteFailed', { msg: resp.statusText }))
      }
    } catch (e) { alert(t('common.error', { msg: e.message })) }
    finally { setBusyId(null) }
  }

  async function toggleAssign(skill) {
    if (!currentApp) return
    const isOn = assigned.has(skill.name)
    setBusyId(skill.name)
    try {
      const base = `${apiUrl}/applications/${encodeURIComponent(currentApp)}/skills`
      const resp = isOn
        ? await fetch(`${base}/${encodeURIComponent(skill.name)}`, { method: 'DELETE' })
        : await fetch(base, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skill_name: skill.name }) })
      if (resp.ok || resp.status === 204 || resp.status === 201) {
        setAssigned(prev => {
          const next = new Set(prev)
          if (isOn) next.delete(skill.name); else next.add(skill.name)
          return next
        })
      } else {
        const d = await resp.json().catch(() => ({}))
        alert(isOn ? t('skills.unassignFailed', { msg: d.detail || resp.statusText }) : t('skills.assignFailed', { msg: d.detail || resp.statusText }))
      }
    } catch (e) { alert(t('common.error', { msg: e.message })) }
    finally { setBusyId(null) }
  }

  return (
    <div className="page">
      <div className="page-hd">
        <h2>{t('skills.title')}</h2>
        <button className="btn btn-ghost" onClick={() => { loadSkills(); loadAssigned() }}>{t('common.refresh')}</button>
      </div>

      <p className="sub" style={{ marginBottom: 14 }}>
        {t('skills.subA')}
        {currentApp
          ? <>{t('skills.subToggle')}<strong>{t('skills.subAssigned')}</strong>{t('skills.subToAdd')}<strong>{currentApp}</strong>.</>
          : <>{t('skills.subSelect')}</>}
      </p>

      {/* Upload new skill */}
      <div
        className="drop-zone"
        onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add('over') }}
        onDragLeave={e => e.currentTarget.classList.remove('over')}
        onDrop={e => { e.preventDefault(); e.currentTarget.classList.remove('over'); const f = e.dataTransfer.files[0]; if (f) { setPicked(f); setUploadMsg({ text: '', cls: '' }) } }}
        onClick={() => uploadInputRef.current?.click()}
      >
        <input ref={uploadInputRef} type="file" accept=".zip" style={{ display: 'none' }} onChange={e => { const f = e.target.files[0]; if (f) { setPicked(f); setUploadMsg({ text: '', cls: '' }) } }} />
        <div className="drop-icon">🧩</div>
        <div className="drop-txt">{t('skills.drop')}</div>
        <div className="drop-hint">{t('skills.dropHint')}</div>
      </div>

      {picked && (
        <div className="chips">
          <span className="chip">🧩 {picked.name}</span>
        </div>
      )}

      <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 12 }}>
        <button className="btn btn-green" disabled={!picked || uploading} onClick={uploadSkill}>
          {uploading ? t('skills.uploading') : t('skills.upload')}
        </button>
        <span className={`settings-msg${uploadMsg.cls ? ' ' + uploadMsg.cls : ''}`}>{uploadMsg.text}</span>
      </div>

      {/* Hidden input shared by all rows for Replace */}
      <input ref={replaceInputRef} type="file" accept=".zip" style={{ display: 'none' }} onChange={e => doReplace(e.target.files[0])} />

      {/* Skills list */}
      <div style={{ marginTop: 26 }}>
        {!skills && !error && <div className="empty"><p><span className="spinning">⟳</span> {t('common.loading')}</p></div>}
        {error && <div className="empty"><p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: error })}</p></div>}
        {skills && skills.length === 0 && <div className="empty"><div className="ei">🧩</div><p>{t('skills.noSkills')}</p></div>}
        {skills && skills.length > 0 && (
          <DataTable
            rows={skills}
            rowKey={s => s.id}
            columns={[
              {
                key: 'assigned', label: t('skills.colAssigned'), width: 90,
                sortable: false, searchable: false,
                render: s => {
                  const on = assigned.has(s.name)
                  const busy = busyId === s.name
                  return (
                    <button
                      className={`btn btn-sm ${on ? 'btn-primary' : 'btn-ghost'}`}
                      disabled={!currentApp || busy}
                      title={currentApp ? (on ? t('skills.removeFromApp') : t('skills.addToApp')) : t('skills.selectAppFirst')}
                      onClick={() => toggleAssign(s)}
                    >
                      {on ? t('skills.on') : t('skills.add')}
                    </button>
                  )
                },
              },
              {
                key: 'name', label: t('skills.colName'), text: s => `${s.name} ${s.id}`,
                render: s => (
                  <span style={{ fontWeight: 500 }}>
                    {s.name}
                    <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>{s.id}</div>
                  </span>
                ),
              },
              { key: 'description', label: t('skills.colDescription'), value: s => s.description, cellClassName: 'muted-cell' },
              {
                key: 'actions', label: t('skills.colActions'), sortable: false, cellClassName: 'actions-cell',
                render: s => {
                  const busy = busyId === s.name
                  return (
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => startReplace(s.name)}>{t('common.replace')}</button>
                      <button className="btn btn-red btn-sm" disabled={busy} onClick={() => deleteSkill(s)}>{t('common.delete')}</button>
                    </div>
                  )
                },
              },
            ]}
          />
        )}
      </div>
    </div>
  )
}
