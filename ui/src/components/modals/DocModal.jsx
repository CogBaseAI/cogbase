import React, { useEffect } from 'react'
import { metaText } from '../../utils'
import { useT } from '../../i18n'

export default function DocModal({ doc, onClose }) {
  const { t } = useT()
  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!doc) return null
  const meta = doc.meta || {}
  return (
    <div className="doc-modal show" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="doc-modal-panel" role="dialog" aria-modal="true">
        <div className="doc-modal-hd">
          <div>
            <h3>{doc.docId || t('docModal.document')}</h3>
            <div className="doc-modal-meta">{[doc.demoName, doc.demoKey].filter(Boolean).join(' • ')}</div>
          </div>
          <button className="doc-modal-close" onClick={onClose} aria-label={t('docModal.close')}>✕</button>
        </div>
        <div className="doc-modal-body">
          <div className="doc-modal-side">
            <h4>{t('docModal.metadata')}</h4>
            <div className="doc-modal-tags">
              {doc.demoName && <span className="doc-modal-tag">{t('docModal.appTag', { name: doc.demoName })}</span>}
              {doc.docId && <span className="doc-modal-tag">{t('docModal.docTag', { id: doc.docId })}</span>}
              {Object.entries(meta).map(([k, v]) => <span key={k} className="doc-modal-tag">{k}: {String(v)}</span>)}
            </div>
            <div className="doc-modal-kv">
              <div>{t('docModal.kvDocument')}</div><div>{doc.docId || '—'}</div>
              <div>{t('docModal.kvDemo')}</div><div>{doc.demoName || '—'}</div>
              <div>{t('docModal.kvMetadata')}</div><div>{metaText(meta)}</div>
            </div>
          </div>
          <div className="doc-modal-main">
            <h4>{t('docModal.fullContent')}</h4>
            <pre className="doc-modal-text">{doc.text || ''}</pre>
          </div>
        </div>
      </div>
    </div>
  )
}
