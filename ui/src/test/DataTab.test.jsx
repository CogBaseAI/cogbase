import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
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
