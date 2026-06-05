import React, { useState, useEffect, useRef } from 'react'
import { useApp } from '../../context'

export default function SkillsTab({ active }) {
  const { apiUrl, currentApp } = useApp()
  const [skills, setSkills] = useState(null)   // null=loading, []|[...]=loaded
  const [error, setError] = useState(null)
  const [assigned, setAssigned] = useState(new Set()) // skill ids assigned to currentApp
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
      setAssigned(new Set(refs.map(r => r.skill_id)))
    } catch { setAssigned(new Set()) }
  }

  useEffect(() => { if (active) { loadSkills(); loadAssigned() } }, [active, currentApp])

  async function uploadSkill() {
    if (!picked) return
    setUploading(true)
    setUploadMsg({ text: 'Uploading…', cls: '' })
    try {
      const form = new FormData()
      form.append('bundle', picked)
      const resp = await fetch(`${apiUrl}/skills`, { method: 'POST', body: form })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        setUploadMsg({ text: (Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || `Error ${resp.status}`, cls: 'err' })
        return
      }
      const skill = await resp.json()
      setUploadMsg({ text: `Uploaded "${skill.name}"`, cls: 'ok' })
      setPicked(null)
      if (uploadInputRef.current) uploadInputRef.current.value = ''
      loadSkills()
    } catch (e) {
      setUploadMsg({ text: 'Network error: ' + e.message, cls: 'err' })
    } finally {
      setUploading(false)
    }
  }

  function startReplace(skillId) {
    replaceTargetRef.current = skillId
    replaceInputRef.current?.click()
  }

  async function doReplace(file) {
    const skillId = replaceTargetRef.current
    replaceTargetRef.current = null
    if (replaceInputRef.current) replaceInputRef.current.value = ''
    if (!skillId || !file) return
    setBusyId(skillId)
    try {
      const form = new FormData()
      form.append('bundle', file)
      const resp = await fetch(`${apiUrl}/skills/${encodeURIComponent(skillId)}`, { method: 'PUT', body: form })
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}))
        alert('Replace failed: ' + ((Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || resp.status))
        return
      }
      loadSkills()
    } catch (e) { alert('Error: ' + e.message) }
    finally { setBusyId(null) }
  }

  async function deleteSkill(skill) {
    if (!confirm(`Delete skill "${skill.name}"? It will be removed from every application that uses it.`)) return
    setBusyId(skill.id)
    try {
      const resp = await fetch(`${apiUrl}/skills/${encodeURIComponent(skill.id)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        loadSkills()
        loadAssigned()
      } else {
        alert('Delete failed: ' + resp.statusText)
      }
    } catch (e) { alert('Error: ' + e.message) }
    finally { setBusyId(null) }
  }

  async function toggleAssign(skill) {
    if (!currentApp) return
    const isOn = assigned.has(skill.id)
    setBusyId(skill.id)
    try {
      const base = `${apiUrl}/applications/${encodeURIComponent(currentApp)}/skills`
      const resp = isOn
        ? await fetch(`${base}/${encodeURIComponent(skill.id)}`, { method: 'DELETE' })
        : await fetch(base, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skill_id: skill.id }) })
      if (resp.ok || resp.status === 204 || resp.status === 201) {
        setAssigned(prev => {
          const next = new Set(prev)
          if (isOn) next.delete(skill.id); else next.add(skill.id)
          return next
        })
      } else {
        const d = await resp.json().catch(() => ({}))
        alert((isOn ? 'Unassign' : 'Assign') + ' failed: ' + (d.detail || resp.statusText))
      }
    } catch (e) { alert('Error: ' + e.message) }
    finally { setBusyId(null) }
  }

  return (
    <div className="page">
      <div className="page-hd">
        <h2>Skills</h2>
        <button className="btn btn-ghost" onClick={() => { loadSkills(); loadAssigned() }}>⟳ Refresh</button>
      </div>

      <p className="sub" style={{ marginBottom: 14 }}>
        Skills are system-wide capabilities uploaded as ZIP bundles (SKILL.md + scripts).
        {currentApp
          ? <> Toggle <strong>Assigned</strong> to add or remove a skill from <strong>{currentApp}</strong>.</>
          : <> Select an app in the Apps tab to assign skills to it.</>}
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
        <div className="drop-txt">Drop a skill ZIP here or click to browse</div>
        <div className="drop-hint">A bundle containing SKILL.md and any scripts/assets</div>
      </div>

      {picked && (
        <div className="chips">
          <span className="chip">🧩 {picked.name}</span>
        </div>
      )}

      <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 12 }}>
        <button className="btn btn-green" disabled={!picked || uploading} onClick={uploadSkill}>
          {uploading ? '⟳ Uploading…' : '▲ Upload Skill'}
        </button>
        <span className={`settings-msg${uploadMsg.cls ? ' ' + uploadMsg.cls : ''}`}>{uploadMsg.text}</span>
      </div>

      {/* Hidden input shared by all rows for Replace */}
      <input ref={replaceInputRef} type="file" accept=".zip" style={{ display: 'none' }} onChange={e => doReplace(e.target.files[0])} />

      {/* Skills list */}
      <div style={{ marginTop: 26 }}>
        {!skills && !error && <div className="empty"><p><span className="spinning">⟳</span> Loading…</p></div>}
        {error && <div className="empty"><p style={{ color: 'var(--red)' }}>Failed: {error}</p></div>}
        {skills && skills.length === 0 && <div className="empty"><div className="ei">🧩</div><p>No skills yet. Upload a skill bundle above.</p></div>}
        {skills && skills.length > 0 && (
          <table>
            <thead>
              <tr>
                <th style={{ width: 90 }}>Assigned</th>
                <th>Name</th>
                <th>Description</th>
                <th style={{ width: 160 }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {skills.map(s => {
                const on = assigned.has(s.id)
                const busy = busyId === s.id
                return (
                  <tr key={s.id}>
                    <td>
                      <button
                        className={`btn btn-sm ${on ? 'btn-primary' : 'btn-ghost'}`}
                        disabled={!currentApp || busy}
                        title={currentApp ? (on ? 'Remove from app' : 'Add to app') : 'Select an app first'}
                        onClick={() => toggleAssign(s)}
                      >
                        {on ? '✓ On' : 'Add'}
                      </button>
                    </td>
                    <td style={{ fontWeight: 500 }}>
                      {s.name}
                      <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'monospace' }}>{s.id}</div>
                    </td>
                    <td style={{ color: 'var(--muted)', fontSize: 12 }}>{s.description}</td>
                    <td>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => startReplace(s.id)}>Replace</button>
                        <button className="btn btn-red btn-sm" disabled={busy} onClick={() => deleteSkill(s)}>Delete</button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
