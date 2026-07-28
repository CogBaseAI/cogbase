import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'
import LoginScreen from '../components/LoginScreen'
import { renderWithCtx } from './renderWithCtx'

// The auth gate (saas mode) and the LoginScreen it shows. These exercise the new
// context auth actions (login/signup) end-to-end: the form calls the context, the
// context hits /auth/*, adopts the returned session, and the gate swaps in the app.
beforeEach(() => {
  // Route-clean start (the hash router persists into the shared jsdom window), and a
  // clean localStorage so a seeded token/mode from one test never leaks into another.
  window.location.hash = ''
  window.history.pushState({}, '', '/')
  window.localStorage.clear()
})
afterEach(() => vi.restoreAllMocks())

const sidebar = () => document.querySelector('.sidebar')
const onLoginScreen = () => screen.queryByText('Review and manage your contracts')

// A fetch router: answers /whoami and /auth/* explicitly, everything else with the
// permissive empty-list default that the app's mount-time list fetches expect.
function mockFetch({ whoami = {}, authRoutes = {} } = {}) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
    const u = String(url)
    const ok = (body, status = 200) =>
      Promise.resolve({ ok: status < 400, status, json: async () => body, text: async () => '' })
    if (u.endsWith('/whoami')) return ok(whoami)
    for (const [suffix, handler] of Object.entries(authRoutes)) {
      if (u.endsWith(suffix)) return handler(opts)
    }
    return ok({ applications: [], namespaces: [], skills: [] })
  })
}

describe('AuthGate — saas gating', () => {
  it('shows the login screen in saas mode with no session', async () => {
    // Seed the last-known mode so the gate resolves to saas on first render; /whoami
    // with no account_id confirms "signed out" and clears any stale local session.
    window.localStorage.setItem('cogbase.mode', 'saas')
    mockFetch({ whoami: { mode: 'saas' } })
    render(<App />)
    await waitFor(() => expect(onLoginScreen()).toBeInTheDocument())
    // The app shell is gated away entirely — no sidebar behind the login form.
    expect(sidebar()).toBeNull()
  })

  it('renders the app (not the login screen) in saas mode with a valid token', async () => {
    window.localStorage.setItem('cogbase.mode', 'saas')
    window.localStorage.setItem('cogbase.accessToken', 'test-token')
    mockFetch({ whoami: { mode: 'saas', account_id: 'acct-1', email: 'a@b.co' } })
    render(<App />)
    await waitFor(() => expect(sidebar()).not.toBeNull())
    expect(onLoginScreen()).toBeNull()
  })

  it('renders the app straight through in dev mode (no gate, no token)', async () => {
    mockFetch({ whoami: { mode: 'dev', account_id: 'default' } })
    render(<App />)
    await waitFor(() => expect(sidebar()).not.toBeNull())
    expect(onLoginScreen()).toBeNull()
  })

  it('signing in adopts the session and swaps the login screen for the app', async () => {
    window.localStorage.setItem('cogbase.mode', 'saas')
    const login = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200,
        json: async () => ({ access_token: 'new-token', account_id: 'acct-9', email: 'user@corp.com' }),
        text: async () => '' }))
    // Before login /whoami reports signed-out (no account_id); the gate stays on the
    // login screen until the token exists. After login the same endpoint would report
    // the resolved account, so seed that so a re-bootstrap doesn't clear the session.
    mockFetch({ whoami: { mode: 'saas', account_id: 'acct-9', email: 'user@corp.com' },
                authRoutes: { '/auth/login': login } })
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(onLoginScreen()).toBeInTheDocument())

    await user.type(screen.getByLabelText('Email'), 'user@corp.com')
    await user.type(screen.getByLabelText('Password'), 'hunter2!!')
    await user.click(document.querySelector('button.btn-primary'))

    // The login POST fired, and once the context adopts the token the gate re-renders
    // into the app shell (accessToken is now set, so the gate no longer matches).
    expect(login).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(sidebar()).not.toBeNull())
    expect(onLoginScreen()).toBeNull()
  })

  it('auto-selects the seeded namespace + app after a brand-new-account signup', async () => {
    window.localStorage.setItem('cogbase.mode', 'saas')
    // A fresh signup mints a new account with no remembered selection. The server
    // seeds a starter workspace (api/provisioning.py); the UI should land the owner
    // in it without a manual pick.
    const signup = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200,
        json: async () => ({ access_token: 'new-token', account_id: 'acct-new', email: 'owner@corp.com' }),
        text: async () => '' }))
    vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
      const u = String(url)
      const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body, text: async () => '' })
      if (u.endsWith('/whoami')) return ok({ mode: 'saas', account_id: 'acct-new', email: 'owner@corp.com' })
      if (u.endsWith('/auth/signup')) return signup(opts)
      if (u.endsWith('/namespaces')) return ok({ namespaces: [{ name: 'legal-team' }] })
      if (u.endsWith('/namespaces/legal-team/applications')) return ok({ applications: [{ name: 'contract-analyst' }] })
      // saas mode configures LLM/embedding providers at the service level, so
      // /system/config reports them configured — otherwise SettingsTab yanks the
      // user to Settings, which would (correctly) preempt the Ingest landing.
      if (u.endsWith('/system/config')) return ok({ llm: { provider: 'openai' }, embedding: { provider: 'openai' } })
      return ok({ applications: [], namespaces: [], skills: [] })
    })
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(onLoginScreen()).toBeInTheDocument())

    // Switch to the sign-up tab, then create the account.
    await user.click(screen.getByRole('button', { name: 'Create account' }))
    await user.type(screen.getByLabelText('Email'), 'owner@corp.com')
    await user.type(screen.getByLabelText('Password'), 'hunter2!!')
    await user.click(document.querySelector('button.btn-primary'))

    expect(signup).toHaveBeenCalledTimes(1)
    // The header's app pill reflects the seeded (namespace, app) pair — proof the
    // provisioned workspace was auto-selected, no manual pick required.
    await waitFor(() => expect(sidebar()).not.toBeNull())
    // Re-query inside waitFor: the pill only gains `.on` once the async cascade
    // (namespaces → reconcile → apps → auto-select) resolves, after the sidebar paints.
    await waitFor(() => expect(document.querySelector('.app-pill.on')?.textContent).toContain('contract-analyst'))
    expect(document.querySelector('.app-pill.on .app-pill-ns')?.textContent).toBe('legal-team')
    // The seeded app has no documents, so the owner is dropped on Ingest to upload:
    // the application tier is focused and its Ingest tab is the active side-nav item.
    await waitFor(() => {
      const ingestNav = within(sidebar()).queryByRole('button', { name: 'Ingest' })
      expect(ingestNav?.className).toContain('active')
    })
  })

  it('auto-selects the seeded app even when a stale application-tier hash survives', async () => {
    window.localStorage.setItem('cogbase.mode', 'saas')
    // A prior session left an application-tier route in the URL (deep-link / a
    // previous account testing loop). The login gate doesn't clear it, so it's live
    // when the seeded workspace mounts. The stale app ('old-app') isn't in the new
    // account's namespace, so the reconcile drops it — the auto-select must still win
    // and land the owner on the seeded contract-analyst, not the pick-an-app state.
    window.location.hash = '#/ns/legal-team/app/old-app/query'

    const signup = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200,
        json: async () => ({ access_token: 'new-token', account_id: 'acct-new', email: 'owner@corp.com' }),
        text: async () => '' }))
    vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
      const u = String(url)
      const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body, text: async () => '' })
      if (u.endsWith('/whoami')) return ok({ mode: 'saas', account_id: 'acct-new', email: 'owner@corp.com' })
      if (u.endsWith('/auth/signup')) return signup(opts)
      if (u.endsWith('/namespaces')) return ok({ namespaces: [{ name: 'legal-team' }] })
      if (u.endsWith('/namespaces/legal-team/applications')) return ok({ applications: [{ name: 'contract-analyst' }] })
      if (u.endsWith('/system/config')) return ok({ llm: { provider: 'openai' }, embedding: { provider: 'openai' } })
      return ok({ applications: [], namespaces: [], skills: [] })
    })
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(onLoginScreen()).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: 'Create account' }))
    await user.type(screen.getByLabelText('Email'), 'owner@corp.com')
    await user.type(screen.getByLabelText('Password'), 'hunter2!!')
    await user.click(document.querySelector('button.btn-primary'))

    expect(signup).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(sidebar()).not.toBeNull())
    await waitFor(() => expect(document.querySelector('.app-pill.on')?.textContent).toContain('contract-analyst'))
    await waitFor(() => {
      const ingestNav = within(sidebar()).queryByRole('button', { name: 'Ingest' })
      expect(ingestNav?.className).toContain('active')
    })
  })

  it('restores the account\'s last-used namespace + app on sign-in', async () => {
    window.localStorage.setItem('cogbase.mode', 'saas')
    // The account's remembered selection from a prior session, keyed by the account
    // it belongs to. Signing in should land the user back on this namespace + app.
    window.localStorage.setItem('cogbase.ns.acct-9', 'alpha')
    window.localStorage.setItem('cogbase.app.acct-9', 'proj-x')

    const login = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200,
        json: async () => ({ access_token: 'new-token', account_id: 'acct-9', email: 'user@corp.com' }),
        text: async () => '' }))
    // A fetch router that reports the remembered namespace + app as real, so the
    // App.jsx reconciliation keeps the restored selection rather than dropping it.
    vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
      const u = String(url)
      const ok = (body) => Promise.resolve({ ok: true, status: 200, json: async () => body, text: async () => '' })
      if (u.endsWith('/whoami')) return ok({ mode: 'saas', account_id: 'acct-9', email: 'user@corp.com' })
      if (u.endsWith('/auth/login')) return login(opts)
      if (u.endsWith('/namespaces')) return ok({ namespaces: [{ name: 'alpha' }] })
      if (u.endsWith('/namespaces/alpha/applications')) return ok({ applications: [{ name: 'proj-x' }] })
      return ok({ applications: [], namespaces: [], skills: [] })
    })
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(onLoginScreen()).toBeInTheDocument())

    await user.type(screen.getByLabelText('Email'), 'user@corp.com')
    await user.type(screen.getByLabelText('Password'), 'hunter2!!')
    await user.click(document.querySelector('button.btn-primary'))

    // Once signed in, the header's app pill reflects the restored (namespace, app)
    // pair — proof the account's last-used selection was rehydrated on login.
    await waitFor(() => expect(sidebar()).not.toBeNull())
    const pill = document.querySelector('.app-pill.on')
    await waitFor(() => expect(pill?.textContent).toContain('proj-x'))
    expect(pill.querySelector('.app-pill-ns')?.textContent).toBe('alpha')
  })
})

describe('LoginScreen — form behavior', () => {
  it('defaults to sign-in with both tabs offered', () => {
    renderWithCtx(<LoginScreen />)
    // Two tab buttons plus a submit button; submit reads "Sign in" by default.
    expect(screen.getByRole('button', { name: 'Create account' })).toBeInTheDocument()
    expect(document.querySelector('button.btn-primary').textContent).toBe('Sign in')
  })

  it('an ?invite= link defaults to sign-up and hides the tab switcher', () => {
    window.history.pushState({}, '', '/?invite=tok123')
    renderWithCtx(<LoginScreen />)
    // The invitee is creating a user: no tab switcher, an invite notice, and the
    // submit button is the create-account action.
    expect(screen.getByText(/You've been invited/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Sign in' })).toBeNull()
    expect(document.querySelector('button.btn-primary').textContent).toBe('Create account')
  })

  it('sign-up submits to /auth/signup and forwards the invite token', async () => {
    window.history.pushState({}, '', '/?invite=tok123')
    const signup = vi.fn(() =>
      Promise.resolve({ ok: true, status: 200,
        json: async () => ({ access_token: 't', account_id: 'a', email: 'x@y.co' }),
        text: async () => '' }))
    mockFetch({ authRoutes: { '/auth/signup': signup } })
    const user = userEvent.setup()
    renderWithCtx(<LoginScreen />)

    await user.type(screen.getByLabelText('Email'), 'invitee@corp.com')
    await user.type(screen.getByLabelText('Password'), 'longenough1')
    await user.click(document.querySelector('button.btn-primary'))

    await waitFor(() => expect(signup).toHaveBeenCalledTimes(1))
    const body = JSON.parse(signup.mock.calls[0][0].body)
    expect(body).toMatchObject({ email: 'invitee@corp.com', invite_token: 'tok123' })
  })

  it('surfaces the server error message on a failed sign-in', async () => {
    const login = vi.fn(() =>
      Promise.resolve({ ok: false, status: 401,
        json: async () => ({ detail: 'Invalid email or password' }),
        text: async () => '' }))
    mockFetch({ authRoutes: { '/auth/login': login } })
    const user = userEvent.setup()
    renderWithCtx(<LoginScreen />)

    await user.type(screen.getByLabelText('Email'), 'user@corp.com')
    await user.type(screen.getByLabelText('Password'), 'wrongpass')
    await user.click(document.querySelector('button.btn-primary'))

    expect(await screen.findByText('Invalid email or password')).toBeInTheDocument()
  })
})
