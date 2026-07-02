import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { AppProvider, useApp } from '../context'
import { I18nProvider } from '../i18n'
import IngestTab from '../components/tabs/IngestTab'

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
        <IngestTab active={active} refreshKey={0} onOpenTaskProgress={() => {}} onOpenWfModal={() => {}} />
      </AppProvider>
    </I18nProvider>
  )
}

// Route the /docs fetch by app name so switching apps returns distinct docs.
function mockFetch(docsByApp) {
  return vi.spyOn(global, 'fetch').mockImplementation((url) => {
    const u = String(url)
    const app = decodeURIComponent(u.match(/\/applications\/([^/]+)\//)?.[1] || '')
    if (u.endsWith('/docs')) {
      return Promise.resolve({ ok: true, json: async () => ({ docs: docsByApp[app] || [] }) })
    }
    if (u.endsWith('/workflows')) {
      return Promise.resolve({ ok: true, json: async () => ({ workflows: [] }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
}

afterEach(() => vi.restoreAllMocks())

it('clears the previous app docs when the app is deleted (cleared)', async () => {
  mockFetch({ app1: [{ doc_id: 'doc-alpha', metadata: {}, status: 'done' }] })

  const { rerender } = render(<Harness appName="app1" active={true} />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  // App deleted from the Apps tab: the Ingest tab is inactive when the
  // selection clears, so only the reset effect (not the load effect) can drop
  // the stale docs.
  rerender(<Harness appName="" active={false} />)

  await waitFor(() => expect(screen.queryByText('doc-alpha')).not.toBeInTheDocument())
})

it('does not show a previous app docs after switching apps', async () => {
  mockFetch({
    app1: [{ doc_id: 'doc-alpha', metadata: {}, status: 'done' }],
    app2: [{ doc_id: 'doc-beta', metadata: {}, status: 'done' }],
  })

  const { rerender } = render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  rerender(<Harness appName="app2" />)

  await waitFor(() => expect(screen.getByText('doc-beta')).toBeInTheDocument())
  expect(screen.queryByText('doc-alpha')).not.toBeInTheDocument()
})
