import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../context'
import { I18nProvider } from '../i18n'
import DataTab from '../components/tabs/DataTab'

// Drives `currentApp` from a prop so a rerender can switch/clear the selection.
function SetApp({ name }) {
  const { setCurrentApp } = useApp()
  useEffect(() => { setCurrentApp(name) }, [name, setCurrentApp])
  return null
}

function Harness({ appName, active = true }) {
  return (
    <I18nProvider>
      <AppProvider>
        <SetApp name={appName} />
        <DataTab active={active} onOpenWfModal={() => {}} wfCompleteCollection={null} onWfCompleteHandled={() => {}} />
      </AppProvider>
    </I18nProvider>
  )
}

// Route fetch by app name so switching apps returns distinct data.
function mockFetch(byApp) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
    const u = String(url)
    const app = decodeURIComponent(u.match(/\/applications\/([^/]+)\//)?.[1] || '')
    const data = byApp[app] || {}
    if (u.endsWith('/query')) {
      return Promise.resolve({ ok: true, json: async () => ({ records: data.records || [] }) })
    }
    if (u.endsWith('/collections')) {
      return Promise.resolve({ ok: true, json: async () => ({ structured: data.collections || [] }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({ tasks: [] }) })
  })
}

afterEach(() => vi.restoreAllMocks())

it('clears the previous app records when the app is deleted (cleared)', async () => {
  mockFetch({ app1: { collections: ['contracts'], records: [{ id: 1, party: 'SecretValueA' }] } })

  const { rerender } = render(<Harness appName="app1" active={true} />)
  await waitFor(() => expect(screen.getByText('SecretValueA')).toBeInTheDocument())

  // App deleted from the Apps tab: the Data tab is inactive when the selection
  // clears, so only the reset effect (not the load effect) can drop the stale
  // records.
  rerender(<Harness appName="" active={false} />)

  await waitFor(() => expect(screen.queryByText('SecretValueA')).not.toBeInTheDocument())
})

it('does not show a previous app records after switching apps', async () => {
  mockFetch({
    app1: { collections: ['contracts'], records: [{ id: 1, party: 'SecretValueA' }] },
    app2: { collections: ['invoices'], records: [{ id: 9, party: 'SecretValueB' }] },
  })

  const { rerender } = render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('SecretValueA')).toBeInTheDocument())

  rerender(<Harness appName="app2" />)

  await waitFor(() => expect(screen.getByText('SecretValueB')).toBeInTheDocument())
  expect(screen.queryByText('SecretValueA')).not.toBeInTheDocument()
})

// Returns the party-cell text of each body row, in display order.
function partyColumn() {
  const rows = screen.getAllByRole('row').slice(1) // drop the header row
  return rows.map(r => within(r).getAllByRole('cell')[2].textContent) // #, id, party
}

const THREE = {
  app1: {
    collections: ['contracts'],
    records: [
      { id: 2, party: 'Beta' },
      { id: 1, party: 'Alpha' },
      { id: 3, party: 'Gamma' },
    ],
  },
}

it('filters rows by the search box across all columns', async () => {
  mockFetch(THREE)
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument())

  await userEvent.type(screen.getByPlaceholderText('Filter records…'), 'gamma')

  await waitFor(() => expect(screen.queryByText('Alpha')).not.toBeInTheDocument())
  expect(screen.getByText('Gamma')).toBeInTheDocument()
  // Count reflects the filtered subset.
  expect(screen.getByText('1 of 3 rows')).toBeInTheDocument()
})

it('shows a no-match state when the filter excludes everything', async () => {
  mockFetch(THREE)
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument())

  await userEvent.type(screen.getByPlaceholderText('Filter records…'), 'zzz')

  await waitFor(() => expect(screen.getByText('No records match your filter.')).toBeInTheDocument())
})

it('sorts by a column, toggling asc → desc → off on repeated clicks', async () => {
  mockFetch(THREE)
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Alpha')).toBeInTheDocument())

  expect(partyColumn()).toEqual(['Beta', 'Alpha', 'Gamma']) // store order

  const header = screen.getByRole('columnheader', { name: /party/i })
  await userEvent.click(header)
  expect(partyColumn()).toEqual(['Alpha', 'Beta', 'Gamma']) // asc

  await userEvent.click(header)
  expect(partyColumn()).toEqual(['Gamma', 'Beta', 'Alpha']) // desc

  await userEvent.click(header)
  expect(partyColumn()).toEqual(['Beta', 'Alpha', 'Gamma']) // cleared → store order
})

it('opens a detail drawer with all fields when a row is clicked', async () => {
  mockFetch({
    app1: { collections: ['contracts'], records: [{ id: 7, party: 'Acme', terms: { years: 3 } }] },
  })
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Acme')).toBeInTheDocument())

  await userEvent.click(screen.getByText('Acme'))

  const drawer = screen.getByText('Record detail').closest('.record-detail')
  expect(drawer).toBeInTheDocument()
  // Object field is pretty-printed in the drawer.
  expect(within(drawer).getByText(/"years": 3/)).toBeInTheDocument()

  await userEvent.click(within(drawer).getByTitle('Close'))
  await waitFor(() => expect(screen.queryByText('Record detail')).not.toBeInTheDocument())
})
