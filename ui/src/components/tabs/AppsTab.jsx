import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import AppDetailModal from '../modals/AppDetailModal'
import DataTable from '../DataTable'

export default function AppsTab({ active, onSwitchTab }) {
  const { apiUrl, namespaceId, authFetch, currentApp, currentAppNs, setCurrentApp } = useApp()
  const { t } = useT()
  const [apps, setApps] = useState(null) // null=loading, []|[...]=loaded
  const [error, setError] = useState(null)
  const [detailApp, setDetailApp] = useState(null)

  // The list is account-wide (every namespace), so address a single app by its
  // own namespace rather than the header-selected one.
  const appUrl = (a, suffix = '') =>
    `${apiUrl}/namespaces/${encodeURIComponent(a.namespace || namespaceId)}/applications/${encodeURIComponent(a.name)}${suffix}`

  // A name is only unique within a namespace, so the selected app matches on both.
  const isCurrent = (a) => a.name === currentApp && (a.namespace || namespaceId) === currentAppNs

  async function loadApps() {
    setApps(null); setError(null)
    try {
      const resp = await authFetch(`${apiUrl}/applications`)
      if (!resp.ok) throw new Error(resp.status + ' ' + resp.statusText)
      const { applications = [] } = await resp.json()
      setApps(applications)
    } catch (e) { setError(e.message) }
  }

  useEffect(() => { if (active) loadApps() }, [active])

  async function viewApp(a) {
    // The list response already carries the full resolved config, but fetch
    // the single-app endpoint so the drawer always shows the freshest config.
    try {
      const resp = await authFetch(appUrl(a))
      if (resp.ok) { setDetailApp(await resp.json()); return }
    } catch {}
    setDetailApp(a)
  }

  async function deleteApp(a) {
    const name = a.name
    if (!confirm(t('apps.confirmDelete', { name }))) return
    try {
      const resp = await authFetch(appUrl(a), { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        if (isCurrent(a)) setCurrentApp('')
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
        <DataTable
          rows={apps}
          rowKey={a => a.name}
          columns={[
            {
              key: 'name', label: t('apps.colName'), text: a => a.name,
              render: a => {
                const cur = isCurrent(a)
                return (
                  <span style={{ fontWeight: cur ? 600 : 400 }}>
                    <button className="link-btn" onClick={() => viewApp(a)} title={t('appDetail.view')}>{a.name}</button>
                    {cur && <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 4 }}>{t('apps.selected')}</span>}
                  </span>
                )
              },
            },
            {
              key: 'status', label: t('apps.colStatus'), text: a => a.status,
              render: a => {
                const sc = a.status === 'active' ? 'b-active' : a.status === 'error' ? 'b-error' : 'b-init'
                return <span className={`badge ${sc}`}>{a.status}</span>
              },
            },
            {
              key: 'created', label: t('apps.colCreated'), sortValue: a => a.created_at || '',
              cellClassName: 'muted-cell',
              render: a => a.created_at ? new Date(a.created_at).toLocaleString() : '—',
            },
            {
              key: 'actions', label: t('apps.colActions'), sortable: false, cellClassName: 'actions-cell',
              render: a => (
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-ghost btn-sm" onClick={() => viewApp(a)}>{t('appDetail.details')}</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => { setCurrentApp(a.name, a.namespace); loadApps() }}>{t('common.use')}</button>
                  <button className="btn btn-red btn-sm" onClick={() => deleteApp(a)}>{t('common.delete')}</button>
                </div>
              ),
            },
          ]}
        />
      )}
      {detailApp && <AppDetailModal app={detailApp} onClose={() => setDetailApp(null)} />}
    </div>
  )
}
