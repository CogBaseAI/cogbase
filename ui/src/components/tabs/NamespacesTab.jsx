import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import DataTable from '../DataTable'

// Account-scoped namespace management: create, edit (description), delete, and
// switch the header's working namespace. Reads/writes the shared
// `namespaces` list in context and calls `refreshNamespaces` after every mutation
// so the header switcher stays in sync. Backed by the /namespaces CRUD routes.
export default function NamespacesTab({ active }) {
  const { apiUrl, authFetch, namespaces, refreshNamespaces, namespaceName, setNamespaceName } = useApp()
  const { t } = useT()
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  // null = create mode; a name string = editing that namespace's description.
  const [editingId, setEditingId] = useState(null)
  const [form, setForm] = useState({ name: '', description: '' })

  // Pull a fresh list whenever the tab is opened.
  useEffect(() => { if (active) refreshNamespaces() }, [active, refreshNamespaces])

  function resetForm() {
    setEditingId(null)
    setForm({ name: '', description: '' })
    setError(null)
  }

  function startEdit(ns) {
    setEditingId(ns.name)
    setForm({
      name: ns.name,
      description: ns.description || '',
    })
    setError(null)
  }

  async function submit() {
    if (busy) return
    const isEdit = editingId !== null
    if (!isEdit && !form.name.trim()) return
    setBusy(true)
    setError(null)
    try {
      // PATCH sends only the description (the id is immutable); POST creates.
      const resp = isEdit
        ? await authFetch(`${apiUrl}/namespaces/${encodeURIComponent(editingId)}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description: form.description }),
          })
        : await authFetch(`${apiUrl}/namespaces`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              name: form.name.trim(),
              description: form.description || null,
            }),
          })
      if (!resp.ok && resp.status !== 201) {
        const d = await resp.json().catch(() => ({}))
        setError(t(isEdit ? 'nsAdmin.saveFailed' : 'nsAdmin.createFailed', { msg: d.detail || resp.statusText }))
        return
      }
      resetForm()
      await refreshNamespaces()
    } catch (e) {
      setError(t(editingId !== null ? 'nsAdmin.saveFailed' : 'nsAdmin.createFailed', { msg: e.message }))
    } finally {
      setBusy(false)
    }
  }

  async function remove(ns) {
    if (busy) return
    if (!confirm(t('nsAdmin.confirmDelete', { id: ns.name }))) return
    setBusy(true)
    setError(null)
    try {
      const resp = await authFetch(`${apiUrl}/namespaces/${encodeURIComponent(ns.name)}`, { method: 'DELETE' })
      if (resp.ok || resp.status === 204 || resp.status === 404) {
        if (editingId === ns.name) resetForm()
        await refreshNamespaces()
      } else {
        const d = await resp.json().catch(() => ({}))
        setError(t('nsAdmin.deleteFailed', { msg: d.detail || resp.statusText }))
      }
    } catch (e) {
      setError(t('nsAdmin.deleteFailed', { msg: e.message }))
    } finally {
      setBusy(false)
    }
  }

  const isEdit = editingId !== null
  const rows = namespaces || []

  return (
    <div className="page">
      <div className="page-hd">
        <h2>{t('nsAdmin.title')}</h2>
        <button className="btn btn-ghost" onClick={refreshNamespaces}>{t('common.refresh')}</button>
      </div>
      <p className="sub" style={{ marginTop: -8, marginBottom: 16 }}>{t('nsAdmin.sub')}</p>

      {/* Create / edit panel */}
      <div className="settings-section" style={{ marginBottom: 20 }}>
        <h3 style={{ fontSize: 14, marginBottom: 12 }}>
          {isEdit ? t('nsAdmin.editTitle', { id: editingId }) : t('nsAdmin.createTitle')}
        </h3>
        <div className="settings-grid">
          <div className="settings-field">
            <label>{t('nsAdmin.idLabel')}</label>
            <input
              type="text"
              value={form.name}
              disabled={isEdit}
              placeholder="legal-team"
              onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
            />
            {!isEdit && <p className="sub" style={{ marginTop: 6 }}>{t('nsAdmin.idHint')}</p>}
          </div>
          <div className="settings-field full">
            <label>{t('nsAdmin.descLabel')} · {t('nsAdmin.optional')}</label>
            <input
              type="text"
              value={form.description}
              onChange={e => setForm(f => ({ ...f, description: e.target.value }))}
            />
          </div>
        </div>
        {error &&<p style={{ color: 'var(--red)', fontSize: 12, marginTop: 8 }}>{error}</p>}
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button
            className="btn btn-primary btn-sm"
            disabled={busy || (!isEdit && !form.name.trim())}
            onClick={submit}
          >
            {isEdit ? t('nsAdmin.save') : t('nsAdmin.create')}
          </button>
          {isEdit && <button className="btn btn-ghost btn-sm" disabled={busy} onClick={resetForm}>{t('nsAdmin.cancel')}</button>}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="empty"><div className="ei">🗂️</div><p>{t('nsAdmin.empty')}</p></div>
      ) : (
        <DataTable
          rows={rows}
          rowKey={ns => ns.name}
          columns={[
            {
              key: 'name', label: t('nsAdmin.colId'), text: ns => ns.name,
              render: ns => {
                const activeNs = ns.name === namespaceName
                return (
                  <span style={{ fontWeight: activeNs ? 600 : 400 }}>
                    <code>{ns.name}</code>
                    {activeNs && <span style={{ fontSize: 10, color: 'var(--accent)', marginLeft: 6 }}>{t('nsAdmin.active')}</span>}
                  </span>
                )
              },
            },
            { key: 'description', label: t('nsAdmin.colDesc'), value: ns => ns.description || '—', cellClassName: 'muted-cell' },
            {
              key: 'created_at', label: t('nsAdmin.colCreated'), sortValue: ns => ns.created_at || '',
              cellClassName: 'muted-cell',
              render: ns => ns.created_at ? new Date(ns.created_at).toLocaleString() : '—',
            },
            {
              key: 'actions', label: t('nsAdmin.colActions'), sortable: false, cellClassName: 'actions-cell',
              render: ns => (
                <div style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-ghost btn-sm" disabled={ns.name === namespaceName} onClick={() => setNamespaceName(ns.name)}>{t('nsAdmin.switchTo')}</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => startEdit(ns)}>{t('nsAdmin.edit')}</button>
                  <button className="btn btn-red btn-sm" onClick={() => remove(ns)}>{t('common.delete')}</button>
                </div>
              ),
            },
          ]}
        />
      )}
    </div>
  )
}
