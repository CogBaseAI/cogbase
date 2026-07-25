import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithCtx } from './renderWithCtx'
import DemosTab from '../components/tabs/DemosTab'

// A minimal demo: one doc, no starter subset, so the deploy flow uploads the full
// corpus in a single metadata group.
const DEMO = {
  key: 'contract',
  name: 'contract-analyst',
  title: 'Contract Analyst',
  description: 'Analyze contracts.',
  config_yaml: 'name: contract-analyst',
  docs: [{ doc_id: 'd1', text: 'hello', metadata: {} }],
}

function jsonResp(body, status = 200) {
  return { ok: status < 400, status, statusText: 'OK', json: async () => body }
}

// Route fetches by URL + method so the deploy flow's calls resolve regardless of
// order, and record which routes were hit so the test can assert the refresh.
function installFetchRouter() {
  const hits = []
  global.fetch = vi.fn(async (url, opts = {}) => {
    const u = String(url)
    const method = (opts.method || 'GET').toUpperCase()
    hits.push(`${method} ${u}`)
    if (u.endsWith('/examples/demos')) return jsonResp({ demos: [DEMO] })
    // App existence check → 404 so the flow creates it.
    if (u.endsWith('/applications/contract-analyst') && method === 'GET') return jsonResp({}, 404)
    if (u.endsWith('/namespaces') && method === 'POST') return jsonResp({}, 201)
    if (u.endsWith('/namespaces') && method === 'GET') return jsonResp({ namespaces: [{ name: 'default' }] })
    if (u.endsWith('/generate/deploy')) return jsonResp({ status: 'active' })
    if (u.endsWith('/upload_documents')) return jsonResp({ task_ids: [] })
    // refreshApps — the namespace-scoped App switcher list.
    if (u.endsWith('/namespaces/default/applications') && method === 'GET')
      return jsonResp({ applications: [{ name: 'contract-analyst', status: 'active' }] })
    return jsonResp({})
  })
  return hits
}

beforeEach(() => {
  window.localStorage.setItem('cogbase.ns.default', 'default')
})

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

it('refreshes the namespace app list after deploying a demo', async () => {
  const hits = installFetchRouter()
  const onSwitchTab = vi.fn()
  const user = userEvent.setup()

  renderWithCtx(<DemosTab active={true} onSwitchTab={onSwitchTab} />)

  // Catalog loads on mount, then deploy.
  await waitFor(() => screen.getByText('Contract Analyst'))
  await user.click(screen.getByRole('button', { name: 'Deploy & Ingest' }))

  // The flow ends by switching to the Ingest tab…
  await waitFor(() => expect(onSwitchTab).toHaveBeenCalledWith('ingest'))

  // …and it must have re-fetched the namespace-scoped app list (refreshApps) so the
  // sidebar App switcher can resolve the just-deployed app as selected. Without the
  // fix this GET never fired and the dropdown showed nothing selected.
  expect(hits.some(h => h.startsWith('GET ') && h.endsWith('/namespaces/default/applications'))).toBe(true)
})

it('refuses to deploy with no namespace selected', async () => {
  window.localStorage.setItem('cogbase.ns.default', '')
  installFetchRouter()
  const onSwitchTab = vi.fn()
  const user = userEvent.setup()

  renderWithCtx(<DemosTab active={true} onSwitchTab={onSwitchTab} />)
  await waitFor(() => screen.getByText('Contract Analyst'))
  await user.click(screen.getByRole('button', { name: 'Deploy & Ingest' }))

  // Surfaces the create-a-namespace hint and never switches tabs.
  await waitFor(() => expect(screen.getByText(/Create or select a namespace first/)).toBeInTheDocument())
  expect(onSwitchTab).not.toHaveBeenCalled()
})
