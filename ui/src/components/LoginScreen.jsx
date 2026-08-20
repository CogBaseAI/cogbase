import React, { useState } from 'react'
import { useApp } from '../context'

// First-party auth screen shown by the gate in saas mode when there's no valid
// session. Email one-time-password: enter an address, get a 6-digit code, enter
// the code. There is no separate sign-in / sign-up step — a known address logs
// in and a first-seen one signs up, on the same /auth/otp/verify call, and this
// form never has to know which happened (the invite token, read from a
// ?invite=... query param, rides along the same way). Kept intentionally small —
// the account/tenant plumbing lives in the context; this is just the form.
export default function LoginScreen() {
  const { requestOtp, verifyOtp } = useApp()

  const inviteToken = new URLSearchParams(window.location.search).get('invite') || ''
  const [step, setStep] = useState('email') // 'email' | 'code'
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const sendCode = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await requestOtp(email, inviteToken)
      setStep('code')
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  const verify = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      await verifyOtp(email, code)
      // On success the context adopts the session; the gate re-renders into the app.
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  const changeEmail = () => {
    setStep('email')
    setCode('')
    setError('')
  }

  return (
    <div style={styles.wrap}>
      <form style={styles.card} onSubmit={step === 'email' ? sendCode : verify}>
        <h1 style={styles.brand}>⚡ Cog<span style={{ color: 'var(--accent, #6c5ce7)' }}>Base</span></h1>
        <p style={styles.subtitle}>Review and manage your contracts</p>

        {inviteToken && (
          <div style={styles.info}>You've been invited to join a team. Enter your email to continue.</div>
        )}

        {step === 'email' ? (
          <label style={styles.label}>
            Email
            <input type="email" value={email} required autoComplete="email" autoFocus
              onChange={e => setEmail(e.target.value)} style={styles.input} />
          </label>
        ) : (
          <>
            <div style={styles.hint}>
              We sent a 6-digit code to <strong>{email}</strong>.{' '}
              <button type="button" onClick={changeEmail} style={styles.linkBtn}>Use a different email</button>
            </div>
            <label style={styles.label}>
              Code
              <input type="text" inputMode="numeric" pattern="[0-9]*" maxLength={6}
                value={code} required autoComplete="one-time-code" autoFocus
                onChange={e => setCode(e.target.value.replace(/\D/g, ''))} style={styles.input} />
            </label>
          </>
        )}

        {error && <div style={styles.error}>{error}</div>}

        <button className="btn btn-primary" type="submit" disabled={busy} style={{ marginTop: 12 }}>
          {busy ? 'Please wait…' : step === 'email' ? 'Send code' : 'Verify code'}
        </button>
      </form>
    </div>
  )
}

const styles = {
  wrap: { minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 },
  card: { width: 360, maxWidth: '100%', display: 'flex', flexDirection: 'column', gap: 8,
    padding: 28, borderRadius: 12, border: '1px solid #e5e7eb', boxShadow: '0 8px 30px rgba(0,0,0,0.08)', background: '#fff' },
  brand: { margin: '0 0 2px', fontSize: 26 },
  subtitle: { margin: '0 0 16px', color: '#6b7280', fontSize: 14 },
  label: { display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13, color: '#374151' },
  input: { padding: '9px 11px', border: '1px solid #d1d5db', borderRadius: 8, fontSize: 14, letterSpacing: 1 },
  hint: { fontSize: 12, color: '#6b7280', marginBottom: 4 },
  linkBtn: { border: 'none', background: 'none', padding: 0, color: 'var(--accent, #6c5ce7)', cursor: 'pointer', fontSize: 12, textDecoration: 'underline' },
  info: { fontSize: 13, color: '#374151', background: '#eef2ff', padding: '10px 12px', borderRadius: 8, marginBottom: 12 },
  error: { fontSize: 13, color: '#b91c1c', background: '#fef2f2', padding: '9px 11px', borderRadius: 8, marginTop: 6 },
}
