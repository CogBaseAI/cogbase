import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor, within, fireEvent } from '@testing-library/react'
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

function Harness({ appName, active = true, refreshKey = 0 }) {
  return (
    <I18nProvider>
      <AppProvider>
        <SetApp name={appName} />
        <DataTab active={active} refreshKey={refreshKey} onOpenWfModal={() => {}} wfCompleteCollection={null} onWfCompleteHandled={() => {}} />
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

it('re-queries the active collection when refreshKey changes (doc ingested/deleted)', async () => {
  // Same app, mutable records: the store changes out-of-band (a doc was ingested
  // or deleted in the Ingest tab) and a refreshKey bump must pull the new rows.
  const byApp = { app1: { collections: ['contracts'], records: [{ id: 1, party: 'BeforeChange' }] } }
  mockFetch(byApp)

  const { rerender } = render(<Harness appName="app1" refreshKey={0} />)
  await waitFor(() => expect(screen.getByText('BeforeChange')).toBeInTheDocument())

  byApp.app1.records = [{ id: 2, party: 'AfterChange' }]
  rerender(<Harness appName="app1" refreshKey={1} />)

  await waitFor(() => expect(screen.getByText('AfterChange')).toBeInTheDocument())
  expect(screen.queryByText('BeforeChange')).not.toBeInTheDocument()
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

it('shows a floating preview when hovering a long cell, and hides it on leave', async () => {
  const longText = 'X'.repeat(120)
  mockFetch({ app1: { collections: ['contracts'], records: [{ id: 1, note: longText }] } })
  render(<Harness appName="app1" />)
  const cell = await screen.findByText(longText)

  expect(document.querySelector('.cell-tip')).toBeNull()

  await userEvent.hover(cell)
  await waitFor(() => expect(document.querySelector('.cell-tip')).not.toBeNull())
  expect(document.querySelector('.cell-tip').textContent).toBe(longText)

  await userEvent.unhover(cell)
  await waitFor(() => expect(document.querySelector('.cell-tip')).toBeNull())
})

it('hides and restores a column via the Columns menu', async () => {
  mockFetch({ app1: { collections: ['contracts'], records: [{ id: 1, party: 'Acme' }] } })
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Acme')).toBeInTheDocument())
  expect(screen.getByRole('columnheader', { name: /party/i })).toBeInTheDocument()

  await userEvent.click(screen.getByRole('button', { name: /Columns/i }))
  const menu = document.querySelector('.col-menu')
  await userEvent.click(within(menu).getByText('party'))

  expect(screen.queryByRole('columnheader', { name: /party/i })).not.toBeInTheDocument()
  expect(screen.queryByText('Acme')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Columns \(1\/2\)/ })).toBeInTheDocument()

  // Re-check it to bring the column back.
  await userEvent.click(within(document.querySelector('.col-menu')).getByText('party'))
  expect(screen.getByRole('columnheader', { name: /party/i })).toBeInTheDocument()
  expect(screen.getByText('Acme')).toBeInTheDocument()
})

it('deselects all (keeping one column) and re-selects all from the menu', async () => {
  mockFetch({ app1: { collections: ['contracts'], records: [{ id: 1, party: 'Acme', region: 'EU' }] } })
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Acme')).toBeInTheDocument())

  await userEvent.click(screen.getByRole('button', { name: /Columns/i }))
  const menu = document.querySelector('.col-menu')

  await userEvent.click(within(menu).getByText('Deselect all'))
  // Only the first column (id) survives; the rest are hidden.
  expect(screen.getByRole('columnheader', { name: /id/i })).toBeInTheDocument()
  expect(screen.queryByRole('columnheader', { name: /party/i })).not.toBeInTheDocument()
  expect(screen.queryByRole('columnheader', { name: /region/i })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Columns \(1\/3\)/ })).toBeInTheDocument()

  await userEvent.click(within(menu).getByText('Select all'))
  expect(screen.getByRole('columnheader', { name: /party/i })).toBeInTheDocument()
  expect(screen.getByRole('columnheader', { name: /region/i })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Columns \(3\/3\)/ })).toBeInTheDocument()
})

it('never hides the last remaining column', async () => {
  mockFetch({ app1: { collections: ['contracts'], records: [{ id: 1, party: 'Acme' }] } })
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Acme')).toBeInTheDocument())

  await userEvent.click(screen.getByRole('button', { name: /Columns/i }))
  const menu = document.querySelector('.col-menu')
  await userEvent.click(within(menu).getByText('party'))
  // Trying to hide the only column left is a no-op.
  await userEvent.click(within(menu).getByText('id'))

  expect(screen.getByRole('columnheader', { name: /id/i })).toBeInTheDocument()
})

it('resizes a column by dragging its handle', async () => {
  mockFetch({ app1: { collections: ['contracts'], records: [{ id: 1, party: 'Acme' }] } })
  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('Acme')).toBeInTheDocument())

  const header = screen.getByRole('columnheader', { name: /party/i })
  fireEvent.mouseDown(header.querySelector('.col-resize-handle'), { clientX: 100 })
  fireEvent.mouseMove(document, { clientX: 260 })
  fireEvent.mouseUp(document)

  // startWidth is 0 under jsdom, so width = max(60, 260 - 100) = 160px.
  expect(header.style.width).toBe('160px')
  // Dragging must not trigger the column's sort.
  expect(header.querySelector('.sort-caret').textContent).toBe('')
})
