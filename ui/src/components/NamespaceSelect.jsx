import React, { useState, useRef, useEffect } from 'react'
import { useT } from '../i18n'

// Filtering combobox for the sidebar namespace switcher. Focusing it opens a
// dropdown of all the account's namespaces; typing substring-filters that list
// (case-insensitive, matching names that *contain* the typed chars). Selecting an
// option — by click or arrow-keys + Enter — commits it. Arbitrary free text is
// still allowed (Enter / blur commits the draft) so a not-yet-created namespace
// can be typed to deploy into, preserving the old free-text input's behaviour.
export default function NamespaceSelect({ id, value, options, onChange }) {
  const { t } = useT()
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState(value)
  const [dirty, setDirty] = useState(false)   // has the user typed since opening?
  const [active, setActive] = useState(-1)     // keyboard-highlighted option index
  const wrapRef = useRef(null)

  // Follow the committed value when it changes elsewhere (hash routing, or picking
  // an app snapping the namespace) — but only while closed, so an in-progress draft
  // isn't clobbered mid-edit.
  useEffect(() => { if (!open) setDraft(value) }, [value, open])

  // Close (and revert any abandoned draft) on an outside click.
  useEffect(() => {
    if (!open) return
    const onDown = e => { if (wrapRef.current && !wrapRef.current.contains(e.target)) close() }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const q = draft.trim().toLowerCase()
  // Show the full list until the user starts typing; then filter by substring.
  const filtered = dirty ? options.filter(o => o.toLowerCase().includes(q)) : options

  function openList() { setOpen(true); setDirty(false); setActive(-1) }
  function close() { setDraft(value); setOpen(false); setActive(-1) }
  function commit(next) {
    const v = (next ?? draft).trim()
    if (v) onChange(v)
    setDraft(v || value)
    setOpen(false)
    setActive(-1)
  }

  function onKeyDown(e) {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open) { openList(); return }
      setActive(a => Math.min(a + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive(a => Math.max(a - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      commit(active >= 0 && filtered[active] ? filtered[active] : draft)
    } else if (e.key === 'Escape') {
      close()
    }
  }

  return (
    <div className="ns-combo" ref={wrapRef}>
      <input
        id={id}
        type="text"
        role="combobox"
        aria-expanded={open}
        autoComplete="off"
        value={draft}
        placeholder={t('header.namespacePlaceholder')}
        onFocus={openList}
        // Reopen on click even when the input already has focus — after a select,
        // focus stays put (the option's mousedown is prevented), so onFocus won't
        // re-fire on the next click.
        onClick={() => { if (!open) openList() }}
        onChange={e => { setDraft(e.target.value); setDirty(true); setOpen(true); setActive(-1) }}
        onKeyDown={onKeyDown}
        onBlur={() => { if (open) commit(draft) }}
      />
      {open && (
        <ul className="ns-combo-menu" role="listbox">
          {filtered.length === 0 && (
            <li className="ns-combo-empty">{t('header.namespaceNoMatch')}</li>
          )}
          {filtered.map((o, i) => (
            <li
              key={o}
              role="option"
              aria-selected={o === value}
              className={`ns-combo-option${i === active ? ' active' : ''}${o === value ? ' current' : ''}`}
              onMouseEnter={() => setActive(i)}
              // mousedown (not click) so it fires before the input's blur.
              onMouseDown={e => { e.preventDefault(); commit(o) }}
            >
              {o}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
