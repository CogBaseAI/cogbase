import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, within, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../App'

// The Layout (sidebar + main) is only reachable through <App/>, which brings its
// own providers — so render App directly rather than via renderWithCtx. A single
// permissive fetch mock covers the mount-time list fetches the tabs fire.
beforeEach(() => {
  // The hash router mirrors state into window.location.hash, which persists across
  // tests in the shared jsdom window — reset it so each test starts route-clean.
  window.location.hash = ''
  window.localStorage.clear()
  vi.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    json: async () => ({ applications: [], namespaces: [], skills: [] }),
    text: async () => '',
  })
})
afterEach(() => vi.restoreAllMocks())

const sidebar = () => document.querySelector('.sidebar')

describe('Layout — focus-driven sidebar', () => {
  it('opens the workspace tier by default and hides other tiers\' items', () => {
    render(<App />)
    const nav = within(sidebar())
    // Workspace items visible…
    expect(nav.getByRole('button', { name: 'Build' })).toBeInTheDocument()
    expect(nav.getByRole('button', { name: 'Apps' })).toBeInTheDocument()
    // …application/account items are not (only their headers are)
    expect(nav.queryByRole('button', { name: 'Query' })).not.toBeInTheDocument()
    expect(nav.queryByRole('button', { name: 'Namespaces' })).not.toBeInTheDocument()
  })

  it('focusing a tier via its header swaps the visible items', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(within(sidebar()).getByRole('button', { name: 'Account' }))
    const nav = within(sidebar())
    expect(nav.getByRole('button', { name: 'Namespaces' })).toBeInTheDocument()
    expect(nav.getByRole('button', { name: 'Skills' })).toBeInTheDocument()
    // workspace items are now hidden
    expect(nav.queryByRole('button', { name: 'Build' })).not.toBeInTheDocument()
  })

  it('focusing the application tier with no app selected shows the empty state', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(within(sidebar()).getByRole('button', { name: 'Application' }))
    // sidebar shows the App switcher (empty namespace → "no apps" hint); main shows
    // the CTA to pick one
    expect(within(sidebar()).getByLabelText('App')).toBeInTheDocument()
    expect(within(sidebar()).getByText('No apps in this namespace')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Go to Apps' })).toBeInTheDocument()
  })

  it('selecting an app in the App switcher clears the empty state', async () => {
    // Namespace-scoped app list returns one app for this test.
    global.fetch.mockResolvedValue({ ok: true, json: async () => ({ applications: [{ name: 'contracts' }] }), text: async () => '' })
    const user = userEvent.setup()
    render(<App />)
    await user.click(within(sidebar()).getByRole('button', { name: 'Application' }))
    const select = await within(sidebar()).findByRole('option', { name: 'contracts' })
    expect(select).toBeInTheDocument()
    await user.selectOptions(within(sidebar()).getByLabelText('App'), 'contracts')
    // app selected → empty-state CTA gone, app tabs now available
    expect(screen.queryByRole('button', { name: 'Go to Apps' })).not.toBeInTheDocument()
    expect(within(sidebar()).getByRole('button', { name: 'Query' })).toBeInTheDocument()
  })

  it('using an app from another namespace snaps the working namespace to it (unified model)', async () => {
    // Persisted selection would otherwise leak between tests; start from default.
    window.localStorage.clear()
    // Account-wide list (Apps tab) has an app living in the 'legal' namespace; the
    // namespace-scoped switcher list stays empty. Report configured providers so
    // the Settings tab doesn't auto-switch away from Apps.
    global.fetch.mockImplementation((url) => {
      const u = String(url)
      const body = u.endsWith('/system/config')
        ? { llm: { provider: 'openai' }, embedding: { provider: 'openai' } }
        : u.includes('/namespaces/') && u.endsWith('/applications')
          ? { applications: [] }                                          // switcher (namespace-scoped)
          : u.endsWith('/applications')
            ? { applications: [{ name: 'contracts', namespace: 'legal', status: 'active' }] }  // Apps tab (account-wide)
            : { namespaces: [] }
      return Promise.resolve({ ok: true, json: async () => body, text: async () => '' })
    })
    const user = userEvent.setup()
    render(<App />)
    // The working namespace starts at 'default'.
    expect(within(sidebar()).getByLabelText('Namespace').value).toBe('default')
    // Open the Apps tab and "Use" the app from the 'legal' namespace.
    await user.click(within(sidebar()).getByRole('button', { name: 'Apps' }))
    await user.click(await screen.findByRole('button', { name: 'Use' }))
    // Selecting it snapped the working namespace to the app's own.
    expect(within(sidebar()).getByLabelText('Namespace').value).toBe('legal')
  })
})

// A fetch mock that reports configured providers (so Settings doesn't auto-switch)
// and lets each test seed the namespace-scoped / account-wide app lists.
function mockApi({ nsApps = [], acctApps = [] } = {}) {
  global.fetch.mockImplementation((url) => {
    const u = String(url)
    const body = u.endsWith('/system/config')
      ? { llm: { provider: 'openai' }, embedding: { provider: 'openai' } }
      : u.includes('/namespaces/') && u.endsWith('/applications')
        ? { applications: nsApps }
        : u.endsWith('/applications')
          ? { applications: acctApps }
          : { namespaces: [] }
    return Promise.resolve({ ok: true, json: async () => body, text: async () => '' })
  })
}

describe('Layout — hash routing', () => {
  it('mirrors the default view into the hash on mount', async () => {
    mockApi()
    render(<App />)
    // Default state is the workspace tier's Build tab, in the default namespace.
    await waitFor(() => expect(window.location.hash).toBe('#/ns/default/build'))
  })

  it('updates the hash as the user navigates', async () => {
    mockApi()
    const user = userEvent.setup()
    render(<App />)
    await user.click(within(sidebar()).getByRole('button', { name: 'Account' }))
    await waitFor(() => expect(window.location.hash).toBe('#/account/namespaces'))
    await user.click(within(sidebar()).getByRole('button', { name: 'Skills' }))
    await waitFor(() => expect(window.location.hash).toBe('#/account/skills'))
  })

  it('restores state from an initial hash (deep link)', async () => {
    window.location.hash = '#/ns/legal/app/contracts/query'
    mockApi({ nsApps: [{ name: 'contracts' }] })
    render(<App />)
    // The application tier is restored with the app selected — its tabs show and the
    // empty-state CTA is absent — and the working namespace follows the deep link.
    await waitFor(() => expect(within(sidebar()).getByRole('button', { name: 'Query' })).toBeInTheDocument())
    expect(within(sidebar()).getByLabelText('Namespace').value).toBe('legal')
    expect(screen.queryByRole('button', { name: 'Go to Apps' })).not.toBeInTheDocument()
  })

  it('follows hashchange (browser back/forward)', async () => {
    mockApi()
    render(<App />)
    expect(within(sidebar()).getByRole('button', { name: 'Build' })).toBeInTheDocument()
    await act(async () => {
      window.location.hash = '#/account/skills'
      window.dispatchEvent(new Event('hashchange'))
    })
    const nav = within(sidebar())
    expect(nav.getByRole('button', { name: 'Skills' })).toBeInTheDocument()
    expect(nav.queryByRole('button', { name: 'Build' })).not.toBeInTheDocument()
  })
})
