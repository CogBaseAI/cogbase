import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'

export default function AppsTab({ active, onSwitchTab }) {
  const { apiUrl, currentApp, setCurrentApp } = useApp()
  const [apps, setApps] = useState(null) // null=loading, []|[...]=loaded
  const [error, setError] = useState(null)

  async function loadApps() {
    setApps(null); setError(null)
    try {
      const resp = await fetch(`${apiUrl}/applications`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { applications = [] } = await resp.json()
      setApps(applications)
    } catch (e) { setError(e.message) }
  }

  useEffect(() => { if (active) loadApps() }, [active])

  async function deleteApp(name) {
    if (!confirm(`Delete "${name}" and all its data?`)) return
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        if (currentApp === name) setCurrentApp('')
        loadApps()
      } else {
        alert('Delete failed: ' + resp.statusText)
      }
    } catch (e) { alert('Error: ' + e.message) }
  }

  return (
    <div className="page">
      <div className="page-hd">
        <h2>Applications</h2>
        <button className="btn btn-ghost" onClick={loadApps}>⟳ Refresh</button>
      </div>
      {!apps && !error && <div className="empty"><p><span className="spinning">⟳</span> Loading…</p></div>}
      {error && <div className="empty"><p style={{ color: 'var(--red)' }}>Failed: {error}</p></div>}
      {apps && apps.length === 0 && <div className="empty"><div className="ei">📭</div><p>No apps yet. Use the Build tab to create one.</p></div>}
      {apps && apps.length > 0 && (
        <table>
          <thead><tr><th>Name</th><th>Status</th><th>Created</th><th style={{ width: 140 }}>Actions</th></tr></thead>
          <tbody>
            {apps.map(a => {
              const sc = a.status === 'active' ? 'b-active' : a.status === 'error' ? 'b-error' : 'b-init'
              const ts = a.created_at ? new Date(a.created_at).toLocaleString() : '—'
              const cur = a.name === currentApp
              return (
                <tr key={a.name}>
                  <td style={{ fontWeight: cur ? 600 : 400 }}>
                    {a.name}
                    {cur && <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 4 }}>● selected</span>}
                  </td>
                  <td><span className={`badge ${sc}`}>{a.status}</span></td>
                  <td style={{ color: 'var(--muted)', fontSize: 11 }}>{ts}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn btn-ghost btn-sm" onClick={() => { setCurrentApp(a.name); loadApps() }}>Use</button>
                      <button className="btn btn-red btn-sm" onClick={() => deleteApp(a.name)}>Delete</button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
