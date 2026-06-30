import React, { useState, useEffect } from 'react'
import { simplifyExtractionSchemas } from '../../utils'
import { useT } from '../../i18n'

export default function ConfigModal({ data, onClose }) {
  const { t } = useT()
  const [simplified, setSimplified] = useState(true)

  useEffect(() => { setSimplified(true) }, [data])
  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!data) return null
  const { demo } = data
  const yaml = demo.config_yaml || ''
  const displayed = simplified ? simplifyExtractionSchemas(yaml) : yaml

  return (
    <div className="config-modal show" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="config-modal-panel" role="dialog" aria-modal="true">
        <div className="config-modal-hd">
          <div>
            <h3>{t('configModal.title', { title: demo.title })}</h3>
            <div className="config-modal-meta">{t('configModal.meta', { name: demo.name, n: (demo.docs || []).length })}</div>
          </div>
          <div className="config-modal-actions">
            <button className="btn btn-ghost btn-sm" style={{ fontSize: 10, padding: '2px 7px' }} onClick={() => setSimplified(v => !v)}>
              {simplified ? t('configModal.simplified') : t('configModal.raw')}
            </button>
            <button className="config-modal-close" onClick={onClose} aria-label={t('configModal.close')}>✕</button>
          </div>
        </div>
        <div className="config-modal-body">
          <div className="config-modal-side">
            <h4>{t('configModal.overview')}</h4>
            <div className="config-modal-list">
              {[
                { label: t('configModal.labelApp'), value: demo.name },
                { label: t('configModal.labelDescription'), value: demo.description || '—' },
                { label: t('configModal.labelDocuments'), value: String((demo.docs || []).length) },
                { label: t('configModal.labelQueryIdeas'), value: (demo.query_examples || []).join(' | ') || '—' },
              ].map(({ label, value }) => (
                <div key={label} className="config-modal-item">
                  <div className="config-modal-item-label">{label}</div>
                  <div className="config-modal-item-value">{value}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="config-modal-main">
            <h4>{t('configModal.fullConfig')}</h4>
            <div className="config-modal-text">
              <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 11, lineHeight: 1.5 }}>{displayed}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
