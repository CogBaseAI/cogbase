import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithCtx } from './renderWithCtx'
import NamespacesTab from '../components/tabs/NamespacesTab'

const NAMESPACES = [
  { name: 'default', description: null, created_at: '2024-01-01T00:00:00Z' },
  { name: 'legal-team', description: 'Contracts', created_at: '2024-02-01T00:00:00Z' },
]

function mockList(items = NAMESPACES) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts = {}) => {
    const method = (opts.method || 'GET').toUpperCase()
    if (method === 'GET') return Promise.resolve({ ok: true, json: async () => ({ namespaces: items, total: items.length }) })
    // POST create / PATCH / DELETE
    return Promise.resolve({ ok: true, status: method === 'POST' ? 201 : 204, json: async () => ({}) })
  })
}

beforeEach(() => { mockList() })
afterEach(() => vi.restoreAllMocks())

it('lists namespaces when active', async () => {
  renderWithCtx(<NamespacesTab active={true} />)
  await waitFor(() => expect(screen.getByText('legal-team')).toBeInTheDocument())
  expect(screen.getByText('Contracts')).toBeInTheDocument()
  // 'default' is an ordinary namespace with no special status — just listed by name.
  expect(screen.getAllByText('default').length).toBeGreaterThan(0)
})

it('does not load when inactive', () => {
  renderWithCtx(<NamespacesTab active={false} />)
  expect(global.fetch).not.toHaveBeenCalled()
})

it('shows the empty state with no namespaces', async () => {
  mockList([])
  renderWithCtx(<NamespacesTab active={true} />)
  await waitFor(() => expect(screen.getByText(/No namespaces yet/)).toBeInTheDocument())
})

it('creates a namespace via POST and refreshes', async () => {
  const spy = mockList()
  const user = userEvent.setup()
  renderWithCtx(<NamespacesTab active={true} />)
  await waitFor(() => screen.getByText('legal-team'))

  await user.type(screen.getByPlaceholderText('legal-team'), 'research')
  await user.click(screen.getByRole('button', { name: /^Create$/ }))

  await waitFor(() => expect(spy).toHaveBeenCalledWith(
    expect.stringMatching(/\/namespaces$/),
    expect.objectContaining({ method: 'POST' }),
  ))
})

it('deletes a namespace after confirm', async () => {
  const spy = mockList()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  const user = userEvent.setup()
  renderWithCtx(<NamespacesTab active={true} />)
  await waitFor(() => screen.getByText('legal-team'))

  // Delete the legal-team row specifically (every namespace, including 'default',
  // is now deletable — no protected default).
  const row = screen.getByText('legal-team').closest('tr')
  const delBtn = within(row).getByRole('button', { name: /Delete/ })
  await user.click(delBtn)

  await waitFor(() => expect(spy).toHaveBeenCalledWith(
    expect.stringContaining('/namespaces/legal-team'),
    expect.objectContaining({ method: 'DELETE' }),
  ))
})
