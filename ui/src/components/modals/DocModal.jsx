import React, { useEffect } from 'react'
import { metaText } from '../../utils'

export default function DocModal({ doc, onClose }) {
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
            <h3>{doc.docId || 'Document'}</h3>
            <div className="doc-modal-meta">{[doc.demoName, doc.demoKey].filter(Boolean).join(' • ')}</div>
          </div>
          <button className="doc-modal-close" onClick={onClose} aria-label="Close document viewer">✕</button>
        </div>
        <div className="doc-modal-body">
          <div className="doc-modal-side">
            <h4>Metadata</h4>
            <div className="doc-modal-tags">
              {doc.demoName && <span className="doc-modal-tag">App: {doc.demoName}</span>}
              {doc.docId && <span className="doc-modal-tag">Doc: {doc.docId}</span>}
              {Object.entries(meta).map(([k, v]) => <span key={k} className="doc-modal-tag">{k}: {String(v)}</span>)}
            </div>
            <div className="doc-modal-kv">
              <div>Document</div><div>{doc.docId || '—'}</div>
              <div>Demo</div><div>{doc.demoName || '—'}</div>
              <div>Metadata</div><div>{metaText(meta)}</div>
            </div>
          </div>
          <div className="doc-modal-main">
            <h4>Full Content</h4>
            <pre className="doc-modal-text">{doc.text || ''}</pre>
          </div>
        </div>
      </div>
    </div>
  )
}
