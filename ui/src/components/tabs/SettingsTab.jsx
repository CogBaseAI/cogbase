import React, { useState, useEffect } from 'react'
import { useApp } from '../../context'

const LLM_DEFAULTS = {
  openai: { model: 'gpt-5.4', mini_model: 'gpt-5.4-mini', base_url: 'https://api.openai.com/v1' },
  'openai-compatible': { model: '', mini_model: '', base_url: '' },
}
const EMB_DEFAULTS = {
  openai: { model: 'text-embedding-3-small', dimensions: '1536', base_url: 'https://api.openai.com/v1' },
  'openai-compatible': { model: '', dimensions: '', base_url: '' },
}

export default function SettingsTab({ active, onAutoSwitch }) {
  const { apiUrl, llmConfigured, embConfigured, setLlmConfigured, setEmbConfigured } = useApp()

  const [llm, setLlm] = useState({ provider: 'openai', model: 'gpt-5.4', mini_model: 'gpt-5.4-mini', base_url: 'https://api.openai.com/v1', api_key: '' })
  const [emb, setEmb] = useState({ provider: 'openai', model: 'text-embedding-3-small', dimensions: '1536', base_url: 'https://api.openai.com/v1', api_key: '' })
  const [llmMsg, setLlmMsg] = useState({ text: '', cls: '' })
  const [embMsg, setEmbMsg] = useState({ text: '', cls: '' })

  async function loadConfig() {
    try {
      const r = await fetch(`${apiUrl}/system/config`)
      if (!r.ok) return
      const d = await r.json()
      if (d.llm) setLlm(l => ({ ...l, provider: d.llm.provider || 'openai', model: d.llm.model || '', mini_model: d.llm.mini_model || '', base_url: d.llm.base_url || '', api_key: d.llm.api_key || '' }))
      if (d.embedding) setEmb(e => ({ ...e, provider: d.embedding.provider || 'openai', model: d.embedding.model || '', dimensions: String(d.embedding.dimensions || ''), base_url: d.embedding.base_url || '', api_key: d.embedding.api_key || '' }))
      const lc = !!d.llm, ec = !!d.embedding
      setLlmConfigured(lc)
      setEmbConfigured(ec)
      if ((!lc || !ec) && onAutoSwitch) onAutoSwitch()
    } catch {}
  }

  useEffect(() => { if (active) loadConfig() }, [active])

  // Load on first mount regardless of active for auto-switch
  useEffect(() => { loadConfig() }, [])

  async function saveConfig(which) {
    const setMsg = which === 'llm' ? setLlmMsg : setEmbMsg
    setMsg({ text: 'Saving…', cls: '' })
    let body = {}
    if (which === 'llm') {
      if (!llm.api_key.trim()) { setMsg({ text: 'API key is required', cls: 'err' }); return }
      body.llm = { provider: llm.provider, model: llm.model.trim(), base_url: llm.base_url.trim(), api_key: llm.api_key.trim() }
      if (llm.mini_model.trim()) body.llm.mini_model = llm.mini_model.trim()
    } else {
      if (!emb.api_key.trim()) { setMsg({ text: 'API key is required', cls: 'err' }); return }
      const dims = parseInt(emb.dimensions, 10)
      if (!dims) { setMsg({ text: 'Dimensions is required', cls: 'err' }); return }
      body.embedding = { provider: emb.provider, model: emb.model.trim(), base_url: emb.base_url.trim(), dimensions: dims, api_key: emb.api_key.trim() }
    }
    try {
      const r = await fetch(`${apiUrl}/system/config`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      const d = await r.json()
      if (!r.ok) { setMsg({ text: (Array.isArray(d.detail) ? d.detail[0]?.msg : d.detail) || 'Error ' + r.status, cls: 'err' }); return }
      setMsg({ text: 'Saved', cls: 'ok' })
      setTimeout(() => setMsg({ text: '', cls: '' }), 3000)
      loadConfig()
    } catch { setMsg({ text: 'Network error', cls: 'err' }) }
  }

  const missing = (!llmConfigured ? 'LLM' : '') + (!llmConfigured && !embConfigured ? ' and ' : '') + (!embConfigured ? 'embedding' : '')

  function applyLlmDefaults(provider) {
    const d = LLM_DEFAULTS[provider] || LLM_DEFAULTS['openai-compatible']
    setLlm(l => ({ ...l, provider, model: d.model, mini_model: d.mini_model, base_url: d.base_url }))
  }
  function applyEmbDefaults(provider) {
    const d = EMB_DEFAULTS[provider] || EMB_DEFAULTS['openai-compatible']
    setEmb(e => ({ ...e, provider, model: d.model, dimensions: d.dimensions, base_url: d.base_url }))
  }

  return (
    <>
      {missing && <div className="warn-bar show">⚠ {missing.charAt(0).toUpperCase() + missing.slice(1)} provider{!llmConfigured && !embConfigured ? 's are' : ' is'} not configured — fill in the fields below to enable queries and ingestion.</div>}
      <div className="page" style={{ overflowY: 'auto' }}>
        <div style={{ maxWidth: 680, margin: '0 auto', padding: 24 }}>
          <div className="settings-section">
            <h3>LLM Provider</h3>
            <p className="sub">Controls the language model used for queries, extraction, and app generation.</p>
            <div className="settings-grid">
              <div className="settings-field">
                <label>Provider</label>
                <select value={llm.provider} onChange={e => applyLlmDefaults(e.target.value)}>
                  <option value="openai">openai</option>
                  <option value="openai-compatible">openai-compatible</option>
                </select>
              </div>
              <div className="settings-field"><label>Model</label><input type="text" value={llm.model} onChange={e => setLlm(l => ({ ...l, model: e.target.value }))} placeholder="gpt-5.4" /></div>
              <div className="settings-field"><label>Mini Model (optional)</label><input type="text" value={llm.mini_model} onChange={e => setLlm(l => ({ ...l, mini_model: e.target.value }))} placeholder="gpt-5.4-mini" /></div>
              <div className="settings-field"><label>Base URL</label><input type="text" value={llm.base_url} onChange={e => setLlm(l => ({ ...l, base_url: e.target.value }))} placeholder="https://api.openai.com/v1" /></div>
              <div className="settings-field full"><label>API Key</label><input type="text" value={llm.api_key} onChange={e => setLlm(l => ({ ...l, api_key: e.target.value }))} placeholder="sk-… (required; use EMPTY for local servers)" /></div>
            </div>
            <div className="settings-actions">
              <button className="btn btn-primary btn-sm" onClick={() => saveConfig('llm')}>Save LLM</button>
              <span className={`settings-msg${llmMsg.cls ? ' ' + llmMsg.cls : ''}`}>{llmMsg.text}</span>
            </div>
          </div>
          <div className="settings-section">
            <h3>Embedding Provider</h3>
            <p className="sub">Controls the embedding model used during ingest and vector search.</p>
            <div className="settings-grid">
              <div className="settings-field">
                <label>Provider</label>
                <select value={emb.provider} onChange={e => applyEmbDefaults(e.target.value)}>
                  <option value="openai">openai</option>
                  <option value="openai-compatible">openai-compatible</option>
                </select>
              </div>
              <div className="settings-field"><label>Model</label><input type="text" value={emb.model} onChange={e => setEmb(em => ({ ...em, model: e.target.value }))} placeholder="text-embedding-3-small" /></div>
              <div className="settings-field"><label>Dimensions</label><input type="number" value={emb.dimensions} onChange={e => setEmb(em => ({ ...em, dimensions: e.target.value }))} placeholder="1536" /></div>
              <div className="settings-field"><label>Base URL</label><input type="text" value={emb.base_url} onChange={e => setEmb(em => ({ ...em, base_url: e.target.value }))} placeholder="https://api.openai.com/v1" /></div>
              <div className="settings-field full"><label>API Key</label><input type="text" value={emb.api_key} onChange={e => setEmb(em => ({ ...em, api_key: e.target.value }))} placeholder="sk-… (required; use EMPTY for local servers)" /></div>
            </div>
            <div className="settings-actions">
              <button className="btn btn-primary btn-sm" onClick={() => saveConfig('embedding')}>Save Embedding</button>
              <span className={`settings-msg${embMsg.cls ? ' ' + embMsg.cls : ''}`}>{embMsg.text}</span>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
