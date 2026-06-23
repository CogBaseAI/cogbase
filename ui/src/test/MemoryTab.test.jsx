import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../context'
import MemoryTab from '../components/tabs/MemoryTab'

// Render MemoryTab inside a provider with `currentApp` pre-selected.
function SetApp({ name }) {
  const { setCurrentApp } = useApp()
  useEffect(() => { setCurrentApp(name) }, [name, setCurrentApp])
  return null
}

function renderMemoryTab(appName = 'contract-analyst') {
  return render(
    <AppProvider>
      {appName ? <SetApp name={appName} /> : null}
      <MemoryTab active={true} />
    </AppProvider>
  )
}

const PENDING = [
  {
    memory_id: 'mem-1', kind: 'fact', content: 'The user prefers EU data residency.',
    entities: ['data-residency'], confidence: 0.91, status: 'pending_review',
    source_event_ids: [{ session_id: 's1', ulid: '01H' }], evidence_snapshot: { turn: 4 },
    created_at: '2026-06-01T00:00:00Z', updated_at: '2026-06-01T00:00:00Z',
  },
  {
    memory_id: 'mem-2', kind: 'correction', content: 'Counterparty is Acme Corp, not Acme LLC.',
    entities: ['acme'], confidence: 0.77, status: 'pending_review',
    source_event_ids: [], evidence_snapshot: {},
    created_at: '2026-06-02T00:00:00Z', updated_at: '2026-06-02T00:00:00Z',
  },
]

function mockPending(list = PENDING) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
    if (String(url).includes('/memory/pending')) {
      return Promise.resolve({ ok: true, json: async () => ({ memories: list }) })
    }
    if (String(url).includes('/memory/review')) {
      const body = JSON.parse(opts.body)
      const id = body.decisions[0].memory_id
      const outcome = body.decisions[0].decision === 'accept' ? 'accepted' : 'rejected'
      return Promise.resolve({ ok: true, json: async () => ({ results: [{ memory_id: id, outcome }] }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
}

afterEach(() => vi.restoreAllMocks())

it('does not load when inactive', () => {
  vi.spyOn(global, 'fetch')
  render(<AppProvider><MemoryTab active={false} /></AppProvider>)
  expect(global.fetch).not.toHaveBeenCalled()
})

it('prompts to select an app when none is chosen', async () => {
  vi.spyOn(global, 'fetch')
  renderMemoryTab('')
  await waitFor(() => expect(screen.getByText(/No app selected/)).toBeInTheDocument())
  expect(global.fetch).not.toHaveBeenCalled()
})

it('renders pending memories after load', async () => {
  mockPending()
  renderMemoryTab()
  await waitFor(() => expect(screen.getByText(/EU data residency/)).toBeInTheDocument())
  expect(screen.getByText(/Acme Corp, not Acme LLC/)).toBeInTheDocument()
  expect(screen.getByText('fact')).toBeInTheDocument()
  expect(screen.getByText('correction')).toBeInTheDocument()
})

it('shows empty state when nothing to review', async () => {
  mockPending([])
  renderMemoryTab()
  await waitFor(() => expect(screen.getByText(/Nothing to review/)).toBeInTheDocument())
})

it('shows error when the fetch fails', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ ok: false, status: 500, statusText: 'Server Error' })
  renderMemoryTab()
  await waitFor(() => expect(screen.getByText(/Failed:/)).toBeInTheDocument())
})

it('filters by kind via the dropdown', async () => {
  const spy = mockPending()
  const user = userEvent.setup()
  renderMemoryTab()
  await waitFor(() => screen.getByText(/EU data residency/))
  await user.selectOptions(screen.getByRole('combobox'), 'correction')
  await waitFor(() =>
    expect(spy).toHaveBeenCalledWith(expect.stringContaining('kind=correction'))
  )
})

describe('review', () => {
  it('accepts a memory and removes its row', async () => {
    const spy = mockPending()
    const user = userEvent.setup()
    renderMemoryTab()
    await waitFor(() => screen.getByText(/EU data residency/))

    await user.click(screen.getAllByRole('button', { name: /Accept/ })[0])

    expect(spy).toHaveBeenCalledWith(
      expect.stringContaining('/memory/review'),
      expect.objectContaining({ method: 'POST' })
    )
    await waitFor(() => expect(screen.queryByText(/EU data residency/)).not.toBeInTheDocument())
    expect(screen.getByText(/now active/)).toBeInTheDocument()
  })

  it('rejects a memory and removes its row', async () => {
    mockPending()
    const user = userEvent.setup()
    renderMemoryTab()
    await waitFor(() => screen.getByText(/Acme Corp/))

    await user.click(screen.getAllByRole('button', { name: /Reject/ })[1])

    await waitFor(() => expect(screen.queryByText(/Acme Corp/)).not.toBeInTheDocument())
    expect(screen.getByText(/superseded/)).toBeInTheDocument()
  })
})

it('toggles the evidence panel when provenance exists', async () => {
  mockPending()
  const user = userEvent.setup()
  renderMemoryTab()
  await waitFor(() => screen.getByText(/EU data residency/))

  // mem-1 has provenance; mem-2 has none, so exactly one Evidence toggle.
  const toggle = screen.getByText(/Evidence/)
  await user.click(toggle)
  await waitFor(() => expect(screen.getByText(/evidence_snapshot/)).toBeInTheDocument())
})
