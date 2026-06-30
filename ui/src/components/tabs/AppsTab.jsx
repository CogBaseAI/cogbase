import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'

export default function AppsTab({ active, onSwitchTab }) {
  const { apiUrl, currentApp, setCurrentApp } = useApp()
  const { t } = useT()
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
    if (!confirm(t('apps.confirmDelete', { name }))) return
    try {
      const resp = await fetch(`${apiUrl}/applications/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        if (currentApp === name) setCurrentApp('')
        loadApps()
      } else {
        alert(t('apps.deleteFailed', { msg: resp.statusText }))
      }
    } catch (e) { alert(t('common.error', { msg: e.message })) }
  }

  return (
    <div className="page">
      <div className="page-hd">
        <h2>{t('apps.title')}</h2>
        <button className="btn btn-ghost" onClick={loadApps}>{t('common.refresh')}</button>
      </div>
      {!apps && !error && <div className="empty"><p><span className="spinning">⟳</span> {t('common.loading')}</p></div>}
      {error && <div className="empty"><p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: error })}</p></div>}
      {apps && apps.length === 0 && <div className="empty"><div className="ei">📭</div><p>{t('apps.empty')}</p></div>}
      {apps && apps.length > 0 && (
        <table>
          <thead><tr><th>{t('apps.colName')}</th><th>{t('apps.colStatus')}</th><th>{t('apps.colCreated')}</th><th style={{ width: 140 }}>{t('apps.colActions')}</th></tr></thead>
          <tbody>
            {apps.map(a => {
              const sc = a.status === 'active' ? 'b-active' : a.status === 'error' ? 'b-error' : 'b-init'
              const ts = a.created_at ? new Date(a.created_at).toLocaleString() : '—'
              const cur = a.name === currentApp
              return (
                <tr key={a.name}>
                  <td style={{ fontWeight: cur ? 600 : 400 }}>
                    {a.name}
                    {cur && <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 4 }}>{t('apps.selected')}</span>}
                  </td>
                  <td><span className={`badge ${sc}`}>{a.status}</span></td>
                  <td style={{ color: 'var(--muted)', fontSize: 11 }}>{ts}</td>
                  <td>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn btn-ghost btn-sm" onClick={() => { setCurrentApp(a.name); loadApps() }}>{t('common.use')}</button>
                      <button className="btn btn-red btn-sm" onClick={() => deleteApp(a.name)}>{t('common.delete')}</button>
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
