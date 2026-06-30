import React, { useState, useRef, useEffect } from 'react'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { streamSSE, simplifyExtractionSchemas } from '../../utils'

function stripConfigMarkers(text) {
  const S = '---CONFIG---', E = '---END CONFIG---'
  if (!text.includes(S)) return text
  const before = text.split(S)[0].trimEnd()
  const rest = text.split(S)[1] || ''
  const after = rest.includes(E) ? rest.split(E)[1].trimStart() : ''
  return [before, after].filter(s => s.trim()).join('\n\n')
}

export default function BuildTab({ active }) {
  const { apiUrl, setCurrentApp } = useApp()
  const { t } = useT()
  const [msgs, setMsgs] = useState([{ role: 'sys', text: t('build.intro') }])
  const [input, setInput] = useState('')
  const [building, setBuilding] = useState(false)
  const [cfgYaml, setCfgYaml] = useState(null)
  const [cfgSimplified, setCfgSimplified] = useState(true)
  const [cfgStatus, setCfgStatus] = useState('—')
  const [asideWidth, setAsideWidth] = useState(310)
  const msgsRef = useRef(null)
  const historyRef = useRef([])
  const textareaRef = useRef(null)
  const resizerRef = useRef(null)
  const asideRef = useRef(null)
  const stickToBottomRef = useRef(true)

  // Pin to bottom only while the user hasn't scrolled up. During streaming we
  // call scrollMsgs on every token, so without this a user can never scroll up
  // to read the head of a long output.
  const scrollMsgs = () => {
    if (msgsRef.current && stickToBottomRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight
  }
  // Force scroll to bottom and re-pin (used when the user sends a message).
  const scrollMsgsForce = () => {
    stickToBottomRef.current = true
    if (msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight
  }
  function onMsgsScroll() {
    const el = msgsRef.current
    if (!el) return
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }

  function autoResize(el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 110) + 'px'
  }

  async function sendBuild() {
    if (building) return
    const text = input.trim()
    if (!text) return
    setInput('')
    if (textareaRef.current) { textareaRef.current.style.height = '' }
    setBuilding(true)

    const userMsg = { role: 'user', text }
    const botMsg = { role: 'bot', text: '' }
    setMsgs(prev => [...prev, userMsg, botMsg])
    setTimeout(scrollMsgsForce, 0)

    try {
      const resp = await fetch(`${apiUrl}/generate/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, history: historyRef.current }),
      })
      if (!resp.ok) {
        const errText = await resp.text()
        setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('build.httpErr', { status: resp.status, msg: errText }) }])
        return
      }

      let raw = ''
      let cfgFound = null
      for await (const d of streamSSE(resp)) {
        if (d.token) {
          raw += d.token
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: stripConfigMarkers(raw) }])
          setTimeout(scrollMsgs, 0)
        } else if (d.result) {
          if (d.result.config_yaml) cfgFound = d.result.config_yaml
          if (d.result.content) raw = d.result.content
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: stripConfigMarkers(raw) }])
        } else if (d.error) {
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('common.error', { msg: d.error }) }])
        }
      }

      historyRef.current.push({ role: 'user', content: text })
      historyRef.current.push({ role: 'assistant', content: raw })

      if (cfgFound) {
        setCfgYaml(cfgFound)
        setCfgStatus(t('build.statusReady'))
        setMsgs(prev => [...prev, { role: 'sys', text: t('build.configReady') }])
        setTimeout(scrollMsgs, 0)
      }
    } catch (e) {
      setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: t('common.networkError', { msg: e.message }), error: true }])
    } finally {
      setBuilding(false)
    }
  }

  function resetBuild() {
    historyRef.current = []
    setCfgYaml(null)
    setCfgStatus('—')
    setMsgs([{ role: 'sys', text: t('build.intro') }])
  }

  async function deployApp() {
    if (!cfgYaml) return
    try {
      const resp = await fetch(`${apiUrl}/generate/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_yaml: cfgYaml }),
      })
      const data = await resp.json()
      if (!resp.ok) {
        setMsgs(prev => [...prev, { role: 'sys', text: t('build.deployFailed', { msg: data.detail || resp.statusText }) }])
      } else if (data.status === 'active') {
        setMsgs(prev => [...prev, { role: 'sys', text: t('build.deployLive', { name: data.name }) }])
        setCurrentApp(data.name)
      } else {
        setMsgs(prev => [...prev, { role: 'sys', text: t('build.deployStatus', { status: data.status + (data.error ? ' — ' + data.error : '') }) }])
      }
    } catch (e) {
      setMsgs(prev => [...prev, { role: 'sys', text: t('common.error', { msg: e.message }) }])
    }
    setTimeout(scrollMsgs, 0)
  }

  // Drag resize
  useEffect(() => {
    const resizer = resizerRef.current
    const aside = asideRef.current
    if (!resizer || !aside) return
    const MIN = 200, MAX = 800
    let startX, startW
    function onDown(e) {
      startX = e.clientX
      startW = aside.getBoundingClientRect().width
      resizer.classList.add('dragging')
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      document.addEventListener('mousemove', onMove)
      document.addEventListener('mouseup', onUp)
    }
    function onMove(e) {
      const delta = startX - e.clientX
      setAsideWidth(Math.min(MAX, Math.max(MIN, startW + delta)))
    }
    function onUp() {
      resizer.classList.remove('dragging')
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
    resizer.addEventListener('mousedown', onDown)
    return () => resizer.removeEventListener('mousedown', onDown)
  }, [])

  const cfgDisplay = cfgYaml ? (cfgSimplified ? simplifyExtractionSchemas(cfgYaml) : cfgYaml) : t('build.noConfig')

  return (
    <div className="chat-layout">
      <div className="chat-col">
        <div className="msgs" ref={msgsRef} onScroll={onMsgsScroll}>
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === 'user' && <div className="msg-who">{t('build.you')}</div>}
              {m.role === 'bot' && <div className="msg-who">{t('build.ai')}</div>}
              <div className="msg-body" style={m.error ? { color: 'var(--red)' } : {}}>{m.text}</div>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <textarea
            ref={textareaRef}
            value={input}
            placeholder={t('build.placeholder')}
            rows={1}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendBuild() } }}
            onChange={e => { setInput(e.target.value); autoResize(e.target) }}
            disabled={building}
          />
          <button className="btn btn-ghost" title={t('build.restart')} onClick={resetBuild}>↺</button>
          <button className="btn btn-primary" disabled={building} onClick={sendBuild}>{t('common.send')}</button>
        </div>
      </div>

      <div className="build-resizer" ref={resizerRef} />

      <div className="chat-aside" id="buildAside" ref={asideRef} style={{ width: asideWidth }}>
        <div className="aside-hd">
          <h3>{t('build.generatedConfig')}</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--muted)' }}>{cfgStatus}</span>
            {cfgYaml && (
              <button className="btn btn-ghost btn-sm" style={{ fontSize: 10, padding: '2px 7px' }}
                onClick={() => setCfgSimplified(v => !v)}>
                {cfgSimplified ? t('build.simplified') : t('build.raw')}
              </button>
            )}
          </div>
        </div>
        <div className="aside-body">
          <pre className="cfg-pre">{cfgDisplay}</pre>
        </div>
        <div className="aside-ft">
          <button className="btn btn-green" disabled={!cfgYaml} onClick={deployApp}>{t('build.deployApp')}</button>
        </div>
      </div>
    </div>
  )
}
