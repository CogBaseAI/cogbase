import React, { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useApp } from '../../context'
import { useT } from '../../i18n'

// Human-readable byte size for the file list.
function fmtSize(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

// SKILL.md carries YAML front-matter (name/description/metadata) that renders as a
// mangled heading in markdown — strip it so only the human-written body is shown.
// The front-matter fields are surfaced separately in the side panel.
function stripFrontMatter(md) {
  return (md || '').replace(/^---\s*\n[\s\S]*?\n---\s*\n?/, '')
}

// Detail view for a skill: renders SKILL.md and lets the user browse the
// bundle's scripts/assets so they can fully audit what the skill does.
export default function SkillModal({ skill, onClose }) {
  const { apiUrl } = useApp()
  const { t } = useT()
  const [content, setContent] = useState(null)   // { markdown, files } | null=loading
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)  // file path being viewed, null=SKILL.md
  const [fileBody, setFileBody] = useState(null)   // { content, truncated } | null=loading
  const [fileError, setFileError] = useState(null)

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    if (!skill) return
    setContent(null); setError(null)
    fetch(`${apiUrl}/skills/${encodeURIComponent(skill.name)}/content`)
      .then(async r => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); return r.json() })
      .then(setContent)
      .catch(e => setError(e.message))
  }, [skill, apiUrl])

  async function openFile(path) {
    setSelected(path); setFileBody(null); setFileError(null)
    try {
      const r = await fetch(`${apiUrl}/skills/${encodeURIComponent(skill.name)}/files/${path.split('/').map(encodeURIComponent).join('/')}`)
      if (!r.ok) throw new Error(r.status + ' ' + r.statusText)
      setFileBody(await r.json())
    } catch (e) { setFileError(e.message) }
  }

  if (!skill) return null
  const files = content?.files || []

  return (
    <div className="doc-modal show" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="doc-modal-panel" role="dialog" aria-modal="true">
        <div className="doc-modal-hd">
          <div>
            <h3>{skill.name}{skill.builtin && <span className="badge b-init" style={{ marginLeft: 8 }}>{t('skills.builtin')}</span>}</h3>
            <div className="doc-modal-meta">{skill.id}</div>
          </div>
          <button className="doc-modal-close" onClick={onClose} aria-label={t('docModal.close')}>✕</button>
        </div>
        <div className="doc-modal-body">
          <div className="doc-modal-side">
            <h4>{t('skillModal.details')}</h4>
            <p style={{ fontSize: 12, marginBottom: 10 }}>{skill.description || <span className="doc-modal-meta">{t('skillModal.noDescription')}</span>}</p>
            {skill.metadata && Object.keys(skill.metadata).length > 0 && (
              <div className="doc-modal-kv" style={{ marginBottom: 14 }}>
                {Object.entries(skill.metadata).map(([k, v]) => (
                  <React.Fragment key={k}><div>{k}</div><div>{typeof v === 'string' ? v : JSON.stringify(v)}</div></React.Fragment>
                ))}
              </div>
            )}
            <h4>{t('skillModal.files')}</h4>
            <ul className="skill-file-list">
              <li>
                <button
                  className={`skill-file${selected === null ? ' active' : ''}`}
                  onClick={() => setSelected(null)}
                >SKILL.md</button>
              </li>
              {files.map(f => (
                <li key={f.path}>
                  <button
                    className={`skill-file${selected === f.path ? ' active' : ''}`}
                    disabled={!f.is_text}
                    title={f.is_text ? f.path : t('skillModal.binary')}
                    onClick={() => f.is_text && openFile(f.path)}
                  >
                    <span className="skill-file-name">{f.path}</span>
                    <span className="skill-file-size">{fmtSize(f.size)}</span>
                  </button>
                </li>
              ))}
              {content && files.length === 0 && <li className="doc-modal-meta">{t('skillModal.noFiles')}</li>}
            </ul>
          </div>
          <div className="doc-modal-main">
            {error && <p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: error })}</p>}
            {!content && !error && <p className="doc-modal-meta"><span className="spinning">⟳</span> {t('common.loading')}</p>}
            {content && selected === null && (
              <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{stripFrontMatter(content.markdown)}</ReactMarkdown></div>
            )}
            {content && selected !== null && (
              <>
                <h4>{selected}</h4>
                {fileError && <p style={{ color: 'var(--red)' }}>{t('common.failed', { msg: fileError })}</p>}
                {!fileBody && !fileError && <p className="doc-modal-meta"><span className="spinning">⟳</span> {t('common.loading')}</p>}
                {fileBody && <pre className="doc-modal-text">{fileBody.content}</pre>}
                {fileBody?.truncated && <p className="doc-modal-meta">{t('skillModal.truncated')}</p>}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
