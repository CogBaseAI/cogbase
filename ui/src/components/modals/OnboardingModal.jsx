import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useApp } from '../../context'
import { useT } from '../../i18n'
import { streamSSE } from '../../utils'

// The kickoff turn. The interview script opens by offering a 2-minute and a full
// path (skills/account-onboarding-interview-legal/SKILL.md), so the conversation
// has to start with the *model* talking — but POST /profile/interview/chat is a
// turn-taking endpoint that needs a user message. So we send this one on open and
// keep it out of the transcript: it goes into the history the server sees (which
// must stay a coherent user/assistant alternation) but is never shown, because a
// message the user did not write should not appear over their name.
const KICKOFF = "Hi — I'd like to set up my company profile."

// What "finish later" sends on the user's behalf. The script already knows how to
// handle someone who wants to stop mid-interview — "if they decline, get bored, or
// say 'just let me use it' — save what you have and stop asking" — so leaving is
// one more turn down that branch rather than a second save path with its own
// rules. Hidden like KICKOFF, and for the same reason: the user pressed a button,
// they did not write this.
//
// The last clause matters. Without it, closing the modal after reading the first
// question would still fire a save, and the model would dutifully write a profile
// out of nothing — flipping `exists: true`, dismissing the onboarding card, and
// relabelling the next run a re-run over a document containing no facts.
const FINAL_SAVE =
  "I need to stop here — save what you have so far so it isn't lost. If I haven't " +
  "actually told you anything worth keeping yet, don't save anything; I'll start over later."

// What the Save button sends. Unlike KICKOFF and FINAL_SAVE this one is *shown* as
// the user's own turn, because it is: they pressed a button that means precisely
// this sentence. The rule KICKOFF follows is that words the user did not choose
// shouldn't appear over their name — an intent they expressed by clicking is not
// that, and hiding it would leave the model's "saved!" reply answering nothing.
const SAVE_NOW = 'Save what you have so far.'

// The onboarding interview, on its own surface (docs/preference-profiles-plan.md
// Phase 5). Stateless like the Build chat: this component owns the message
// history and posts it back each turn.
//
// It is a modal rather than a tab because it is an interruption of work already in
// progress — the user is mid-flow in Query or Settings and returns there when it
// closes. Closing mid-interview loses the transcript but not the answers: every
// exit runs through `finishLater`, which spends one last turn asking the model to
// save what it has. A partial profile is the expected outcome, not a degraded one.
export default function OnboardingModal({ open, onClose }) {
  const { apiUrl, authFetch, profile, adoptSavedProfile } = useApp()
  const { t } = useT()
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  // Whether *this* conversation has written the profile. Drives the saved banner;
  // the card that opened the modal is dismissed by the context update instead.
  const [saved, setSaved] = useState(false)
  const msgsRef = useRef(null)
  const historyRef = useRef([])
  const textareaRef = useRef(null)
  const startedRef = useRef(false)
  // Answers the user has given that no save has captured yet — the one condition
  // under which leaving costs them something, and so the gate on both the Save
  // button and the closing save. Mirrored into state because the button's enabled
  // state has to re-render on it, and into a ref because the closing save reads it
  // after the modal is gone, where a captured state value would be stale.
  const dirtyRef = useRef(false)
  const [dirty, setDirtyState] = useState(false)
  const markDirty = (v) => { dirtyRef.current = v; setDirtyState(v) }
  // The turn currently streaming, if any. `finishLater` waits for it before
  // saving, because a turn in flight is holding the answer the user typed
  // immediately before deciding to leave — the most expensive one to drop.
  const inflightRef = useRef(null)
  // Whether the profile already existed when this modal *opened* — which is what
  // makes the turn a re-run server-side, and here decides the title. Frozen at
  // open rather than read live, so saving mid-conversation doesn't relabel the
  // interview the user is still in the middle of.
  const [rerun, setRerun] = useState(false)

  const scrollMsgs = () => {
    if (msgsRef.current) msgsRef.current.scrollTop = msgsRef.current.scrollHeight
  }

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape' && !busy) finishLater() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose, busy])

  // Open → clear the last interview, then run the kickoff turn once. Reopening
  // starts fresh rather than resuming a half-finished interview (the server holds
  // no state; a stale client transcript would be the only thing carrying it).
  //
  // The clearing happens at *open*, not at close, because `finishLater` lets a
  // save turn outlive the modal — and that turn reads `historyRef` and `dirtyRef`
  // on its way out. Wiping them on close would pull the rug out from under work
  // already in flight. Nothing renders while closed, so stale state costs nothing.
  useEffect(() => {
    if (!open) {
      startedRef.current = false
      return
    }
    if (startedRef.current) return
    startedRef.current = true
    historyRef.current = []
    markDirty(false)
    inflightRef.current = null
    setMsgs([])
    setInput('')
    setSaved(false)
    setRerun(!!profile?.exists)
    startTurn(KICKOFF, { hidden: true })
  }, [open])

  // Every turn goes through here so `inflightRef` always names the one streaming
  // now. Only `finishLater` reads it; everything else just wants `send`.
  function startTurn(text, opts) {
    const p = send(text, opts)
    inflightRef.current = p
    p.finally(() => { if (inflightRef.current === p) inflightRef.current = null })
    return p
  }

  async function send(text, { hidden = false } = {}) {
    if (busy) return
    const body = (text || '').trim()
    if (!body) return
    // Marked dirty before the round trip, not after: if the user leaves while
    // this very turn is streaming, it is still an answer that needs saving.
    if (!hidden) markDirty(true)
    setBusy(true)
    setMsgs(prev => [
      ...prev,
      ...(hidden ? [] : [{ role: 'user', text: body }]),
      { role: 'bot', text: '' },
    ])
    setTimeout(scrollMsgs, 0)

    let answer = ''
    let savedMarkdown = null
    try {
      const resp = await authFetch(`${apiUrl}/profile/interview/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: body, history: historyRef.current }),
      })
      if (!resp.ok) {
        // 503 is the deployment saying it has no interview to run (no script
        // registered, or no LLM) — a configuration fact, not a transient error,
        // so it gets its own sentence instead of a bare status code.
        const detail = await resp.text().catch(() => '')
        setMsgs(prev => [...prev.slice(0, -1), {
          role: 'sys',
          text: resp.status === 503 ? t('onboard.unavailable') : t('onboard.httpErr', { status: resp.status, msg: detail }),
          error: true,
        }])
        return
      }

      for await (const d of streamSSE(resp)) {
        if (d.token) {
          answer += d.token
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer }])
          setTimeout(scrollMsgs, 0)
        } else if (d.done) {
          if (d.content) answer = d.content
          if (d.profile_saved && d.markdown) savedMarkdown = d.markdown
          setMsgs(prev => [...prev.slice(0, -1), { role: 'bot', text: answer }])
        } else if (d.error) {
          setMsgs(prev => [...prev.slice(0, -1), { role: 'sys', text: t('common.error', { msg: d.error }), error: true }])
          return
        }
      }

      historyRef.current.push({ role: 'user', content: body })
      historyRef.current.push({ role: 'assistant', content: answer })

      if (savedMarkdown) {
        // Every app in the account already reads this — the server hot-patched the
        // live instances before answering (apply_profile_to_live_apps).
        adoptSavedProfile(savedMarkdown)
        setSaved(true)
        markDirty(false)
      }
    } catch (e) {
      setMsgs(prev => [...prev.slice(0, -1), { role: 'sys', text: t('common.networkError', { msg: e.message }), error: true }])
    } finally {
      setBusy(false)
      setTimeout(scrollMsgs, 0)
    }
  }

  // Leaving is a first-class action, so "finish later" is not a bare close: it
  // spends one more turn asking the model to save what it has, then closes
  // immediately without waiting for the answer. Waiting would be the wrong trade —
  // the user pressed a button that means "let me go".
  function finishLater() {
    if (dirtyRef.current) saveUnfinished(historyRef.current, inflightRef.current)
    onClose()
  }

  // The detached closing turn. It deliberately writes no component state: the
  // modal is already gone, so there is no transcript to append to and no error to
  // show. If it fails, the profile is simply unsaved — exactly where we started.
  //
  // `history` is the live array `send` pushes into, and the reset above replaces
  // the ref rather than mutating it. So waiting on `inflight` here picks up the
  // turn that was streaming when the user left, and saves the answer inside it.
  async function saveUnfinished(history, inflight) {
    if (inflight) {
      try { await inflight } catch { /* send() already surfaced it */ }
    }
    // That turn may have been a save itself, in which case there is nothing left
    // to write and the model should not be asked to write it twice.
    if (!dirtyRef.current) return
    try {
      const resp = await authFetch(`${apiUrl}/profile/interview/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: FINAL_SAVE, history }),
      })
      if (!resp.ok) return
      for await (const d of streamSSE(resp)) {
        if (d.done && d.profile_saved && d.markdown) {
          markDirty(false)
          adoptSavedProfile(d.markdown)
        }
      }
    } catch { /* the user has left; an unsaved profile is the status quo, not a regression */ }
  }

  function autoResize(el) {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 110) + 'px'
  }

  if (!open) return null

  return (
    <div className="onboard-modal show">
      <div className="onboard-modal-panel" role="dialog" aria-modal="true" aria-label={t('onboard.title')}>
        <div className="onboard-modal-hd">
          <div>
            <h3>{rerun ? t('onboard.titleRerun') : t('onboard.title')}</h3>
            <p className="onboard-modal-sub">{t('onboard.sub')}</p>
          </div>
          <button className="wf-modal-close" onClick={finishLater} aria-label={t('onboard.close')}>✕</button>
        </div>
        {saved && <div className="onboard-saved">{t('onboard.saved')}</div>}
        <div className="msgs" ref={msgsRef}>
          {msgs.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === 'user' && <div className="msg-who">{t('build.you')}</div>}
              {m.role === 'bot' && <div className="msg-who">{t('onboard.ai')}</div>}
              <div className="msg-body" style={m.error ? { color: 'var(--red)' } : {}}>
                {/* The interviewer writes prose with emphasis and the occasional
                    list, so its turns render as markdown (as the Query tab's do)
                    rather than showing their own asterisks. */}
                {m.role === 'bot' && m.text
                  ? <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown></div>
                  : m.text || (busy && m.role === 'bot' ? t('onboard.thinking') : '')}
              </div>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <textarea
            ref={textareaRef}
            value={input}
            placeholder={t('onboard.placeholder')}
            rows={1}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                const text = input.trim()
                if (!text) return
                setInput('')
                if (textareaRef.current) textareaRef.current.style.height = ''
                startTurn(text)
              }
            }}
            onChange={e => { setInput(e.target.value); autoResize(e.target) }}
            disabled={busy}
          />
          <button
            className="btn btn-primary"
            disabled={busy || !input.trim()}
            onClick={() => {
              const text = input.trim()
              if (!text) return
              setInput('')
              if (textareaRef.current) textareaRef.current.style.height = ''
              startTurn(text)
            }}
          >{t('common.send')}</button>
        </div>
        <div className="onboard-modal-ft">
          {/* Leaving is a first-class action, not an escape hatch: the interview
              competes with real work and is written to lose gracefully. */}
          <button className="btn btn-ghost btn-sm" onClick={finishLater}>
            {saved ? t('onboard.done') : t('onboard.later')}
          </button>
          {/* Saving on demand, so the user can bank what they've said and keep
              going — or stop, having watched it land. The closing save covers the
              same ground silently, but only for someone who didn't think to ask;
              a durable answer shouldn't depend on the user trusting an invisible
              mechanism. Disabled with nothing to bank, so it never writes an empty
              profile. */}
          <button
            className="btn btn-sm"
            disabled={busy || !dirty}
            onClick={() => startTurn(SAVE_NOW)}
          >{t('onboard.saveNow')}</button>
        </div>
      </div>
    </div>
  )
}
