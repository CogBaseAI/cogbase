import React, { useState } from 'react'
import { useApp } from '../context'

// First-party auth screen shown by the gate in saas mode when there's no valid
// session. Handles sign-in, new-account sign-up, and invite acceptance (the
// invite token is read from a ?invite=... query param on the link the owner
// shares). Kept intentionally small — the account/tenant plumbing lives in the
// context; this is just the form.
export default function LoginScreen() {
  const { login, signup } = useApp()

  const inviteToken = new URLSearchParams(window.location.search).get('invite') || ''
  // With an invite link, default to sign-up (the invitee is creating their user);
  // otherwise default to sign-in.
  const [tab, setTab] = useState(inviteToken ? 'signup' : 'login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (tab === 'signup') await signup(email, password, inviteToken)
      else await login(email, password)
      // On success the context adopts the session; the gate re-renders into the app.
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  const isSignup = tab === 'signup'

  return (
    <div style={styles.wrap}>
      <form style={styles.card} onSubmit={submit}>
        <h1 style={styles.brand}>⚡ Cog<span style={{ color: 'var(--accent, #6c5ce7)' }}>Base</span></h1>
        <p style={styles.subtitle}>Review and manage your contracts</p>

        {inviteToken ? (
          <div style={styles.info}>You've been invited to join a team. Create your account below.</div>
        ) : (
          <div style={styles.tabs}>
            <button type="button" onClick={() => setTab('login')}
              style={{ ...styles.tab, ...(isSignup ? {} : styles.tabActive) }}>Sign in</button>
            <button type="button" onClick={() => setTab('signup')}
              style={{ ...styles.tab, ...(isSignup ? styles.tabActive : {}) }}>Create account</button>
          </div>
        )}

        <label style={styles.label}>
          Email
          <input type="email" value={email} required autoComplete="email"
            onChange={e => setEmail(e.target.value)} style={styles.input} />
        </label>
        <label style={styles.label}>
          Password
          <input type="password" value={password} required
            autoComplete={isSignup ? 'new-password' : 'current-password'}
            minLength={isSignup ? 8 : undefined}
            onChange={e => setPassword(e.target.value)} style={styles.input} />
        </label>
        {isSignup && <div style={styles.hint}>At least 8 characters.</div>}

        {error && <div style={styles.error}>{error}</div>}

        <button className="btn btn-primary" type="submit" disabled={busy} style={{ marginTop: 12 }}>
          {busy ? 'Please wait…' : isSignup ? 'Create account' : 'Sign in'}
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
  tabs: { display: 'flex', gap: 6, marginBottom: 12 },
  tab: { flex: 1, padding: '8px 10px', border: '1px solid #e5e7eb', background: '#f9fafb', borderRadius: 8, cursor: 'pointer', fontSize: 14 },
  tabActive: { background: '#111827', color: '#fff', borderColor: '#111827' },
  label: { display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13, color: '#374151' },
  input: { padding: '9px 11px', border: '1px solid #d1d5db', borderRadius: 8, fontSize: 14 },
  hint: { fontSize: 12, color: '#9ca3af' },
  info: { fontSize: 13, color: '#374151', background: '#eef2ff', padding: '10px 12px', borderRadius: 8, marginBottom: 12 },
  error: { fontSize: 13, color: '#b91c1c', background: '#fef2f2', padding: '9px 11px', borderRadius: 8, marginTop: 6 },
}
