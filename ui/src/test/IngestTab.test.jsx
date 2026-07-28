import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../context'
import { I18nProvider } from '../i18n'
import IngestTab from '../components/tabs/IngestTab'

// Drives `currentApp` from a prop so a rerender can switch/clear the selection.
function SetApp({ name }) {
  const { setCurrentApp } = useApp()
  useEffect(() => { setCurrentApp(name) }, [name, setCurrentApp])
  return null
}

function Harness({ appName, active = true, onDocsChanged = () => {}, onStartQuery = () => {} }) {
  return (
    <I18nProvider>
      <AppProvider>
        <SetApp name={appName} />
        <IngestTab active={active} refreshKey={0} onOpenTaskProgress={() => {}} onOpenWfModal={() => {}} onDocsChanged={onDocsChanged} onStartQuery={onStartQuery} />
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

// Routes /docs off a mutable store so the reload after a delete returns the
// updated list, and captures every DELETE request. `deleteResponse` controls
// what the DELETE call resolves to (or rejects with, when it's an Error).
function mockFetchDeletable(docs, deleteResponse = { ok: true, status: 200 }) {
  const store = { docs: [...docs] }
  const deleteCalls = []
  vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
    const u = String(url)
    if (opts?.method === 'DELETE') {
      deleteCalls.push(u)
      if (deleteResponse instanceof Error) return Promise.reject(deleteResponse)
      return Promise.resolve(deleteResponse)
    }
    if (u.endsWith('/docs')) {
      return Promise.resolve({ ok: true, json: async () => ({ docs: store.docs }) })
    }
    if (u.endsWith('/workflows')) {
      return Promise.resolve({ ok: true, json: async () => ({ workflows: [] }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
  return { store, deleteCalls }
}

it('deletes a document after confirmation and reloads the list', async () => {
  const { store, deleteCalls } = mockFetchDeletable([
    { doc_id: 'doc-alpha', metadata: {}, status: 'done' },
    { doc_id: 'doc-beta', metadata: {}, status: 'done' },
  ])
  const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)

  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  // Simulate the backend dropping the doc, so the reload returns the shorter list.
  store.docs = store.docs.filter(d => d.doc_id !== 'doc-alpha')

  const rows = screen.getAllByText('Delete')
  await userEvent.click(rows[0])

  expect(confirmSpy).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(screen.queryByText('doc-alpha')).not.toBeInTheDocument())
  expect(screen.getByText('doc-beta')).toBeInTheDocument()
  expect(deleteCalls).toHaveLength(1)
  expect(deleteCalls[0]).toMatch(/\/applications\/app1\/docs\/doc-alpha$/)
})

it('notifies onDocsChanged after a delete so the Data tab can refresh', async () => {
  const { store } = mockFetchDeletable([
    { doc_id: 'doc-alpha', metadata: {}, status: 'done' },
  ])
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  const onDocsChanged = vi.fn()

  render(<Harness appName="app1" onDocsChanged={onDocsChanged} />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  store.docs = []
  await userEvent.click(screen.getByText('Delete'))

  await waitFor(() => expect(onDocsChanged).toHaveBeenCalledTimes(1))
})

it('does not notify onDocsChanged when the delete fails', async () => {
  mockFetchDeletable(
    [{ doc_id: 'doc-alpha', metadata: {}, status: 'done' }],
    { ok: false, status: 500, statusText: 'Server Error' },
  )
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  vi.spyOn(window, 'alert').mockImplementation(() => {})
  const onDocsChanged = vi.fn()

  render(<Harness appName="app1" onDocsChanged={onDocsChanged} />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  await userEvent.click(screen.getByText('Delete'))

  await waitFor(() => expect(window.alert).toHaveBeenCalledTimes(1))
  expect(onDocsChanged).not.toHaveBeenCalled()
})

it('does not delete when the confirmation is cancelled', async () => {
  const { deleteCalls } = mockFetchDeletable([
    { doc_id: 'doc-alpha', metadata: {}, status: 'done' },
  ])
  vi.spyOn(window, 'confirm').mockReturnValue(false)

  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  await userEvent.click(screen.getByText('Delete'))

  expect(deleteCalls).toHaveLength(0)
  expect(screen.getByText('doc-alpha')).toBeInTheDocument()
})

it('treats a 404 as a successful delete and reloads', async () => {
  const { store, deleteCalls } = mockFetchDeletable(
    [{ doc_id: 'doc-alpha', metadata: {}, status: 'done' }],
    { ok: false, status: 404 },
  )
  vi.spyOn(window, 'confirm').mockReturnValue(true)

  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  store.docs = []
  await userEvent.click(screen.getByText('Delete'))

  await waitFor(() => expect(screen.queryByText('doc-alpha')).not.toBeInTheDocument())
  expect(deleteCalls).toHaveLength(1)
})

it('alerts and keeps the doc when the delete fails', async () => {
  const { store, deleteCalls } = mockFetchDeletable(
    [{ doc_id: 'doc-alpha', metadata: {}, status: 'done' }],
    { ok: false, status: 500, statusText: 'Server Error' },
  )
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})

  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  // A failed delete should not remove the row even if the store later changes.
  store.docs = []
  await userEvent.click(screen.getByText('Delete'))

  await waitFor(() => expect(alertSpy).toHaveBeenCalledTimes(1))
  expect(alertSpy.mock.calls[0][0]).toContain('Server Error')
  expect(deleteCalls).toHaveLength(1)
  expect(screen.getByText('doc-alpha')).toBeInTheDocument()
})

// Routes /docs off a fixed list and captures every original-download request.
// `downloadResponse` controls what the /original call resolves to (or rejects
// with, when it's an Error).
function mockFetchDownloadable(docs, downloadResponse = { ok: true, blob: async () => new Blob(['pdf-bytes']) }) {
  const downloadCalls = []
  vi.spyOn(global, 'fetch').mockImplementation((url) => {
    const u = String(url)
    if (u.endsWith('/original')) {
      downloadCalls.push(u)
      if (downloadResponse instanceof Error) return Promise.reject(downloadResponse)
      return Promise.resolve(downloadResponse)
    }
    if (u.endsWith('/docs')) {
      return Promise.resolve({ ok: true, json: async () => ({ docs }) })
    }
    if (u.endsWith('/workflows')) {
      return Promise.resolve({ ok: true, json: async () => ({ workflows: [] }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
  return { downloadCalls }
}

it('downloads the original file using the source filename', async () => {
  const { downloadCalls } = mockFetchDownloadable([
    { doc_id: 'doc-alpha', metadata: { source_filename: 'Contract A.pdf' }, status: 'done' },
  ])
  URL.createObjectURL = vi.fn(() => 'blob:fake')
  URL.revokeObjectURL = vi.fn()
  const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  await userEvent.click(screen.getByText('⤓ Download'))

  await waitFor(() => expect(downloadCalls).toHaveLength(1))
  expect(downloadCalls[0]).toMatch(/\/applications\/app1\/docs\/doc-alpha\/original$/)
  expect(clickSpy).toHaveBeenCalledTimes(1)
  expect(URL.createObjectURL).toHaveBeenCalledTimes(1)
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake')
})

// Routes the full upload → poll → docs cycle so the post-upload review CTA can
// appear. `filenames` drives how many files (and thus tasks) the upload yields;
// `taskStatus` lets a test drive a failed ingest (no CTA). Each file gets a
// task tN → doc docN, all resolving to `taskStatus`.
function mockFetchUpload(taskStatus = 'done', count = 1) {
  const taskIds = Array.from({ length: count }, (_, i) => `t${i + 1}`)
  vi.spyOn(global, 'fetch').mockImplementation((url) => {
    const u = String(url)
    if (u.endsWith('/upload_documents')) {
      return Promise.resolve({ ok: true, json: async () => ({ task_ids: taskIds }) })
    }
    const m = u.match(/\/tasks\/t(\d+)$/)
    if (m) {
      const tid = `t${m[1]}`
      return Promise.resolve({ ok: true, json: async () => ({ task_id: tid, doc_id: `doc-${m[1]}`, status: taskStatus, error: taskStatus === 'failed' ? 'boom' : undefined }) })
    }
    if (u.endsWith('/docs')) {
      return Promise.resolve({ ok: true, json: async () => ({ docs: [] }) })
    }
    if (u.endsWith('/workflows')) {
      return Promise.resolve({ ok: true, json: async () => ({ workflows: [] }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
}

async function uploadFiles(container, filenames) {
  const fileInput = container.querySelector('input[type=file]')
  const files = filenames.map(name => new File(['contract bytes'], name, { type: 'application/pdf' }))
  await userEvent.upload(fileInput, files)
  await userEvent.click(screen.getByText('▲ Upload & Ingest'))
}

it('auto-sends a filename-scoped review when a single contract is uploaded', async () => {
  mockFetchUpload('done', 1)
  const onStartQuery = vi.fn()

  const { container } = render(<Harness appName="app1" onStartQuery={onStartQuery} />)
  await uploadFiles(container, ['MSA_Acme.pdf'])

  const reviewBtn = await screen.findByText('⚖ Review "MSA_Acme.pdf"', {}, { timeout: 4000 })
  await userEvent.click(reviewBtn)

  expect(onStartQuery).toHaveBeenCalledTimes(1)
  const [query, send] = onStartQuery.mock.calls[0]
  expect(query).toContain('MSA_Acme.pdf')
  expect(send).toBe(true) // single → auto-send
  // One-shot: the card is dismissed once the review is launched.
  expect(screen.queryByText('⚖ Review "MSA_Acme.pdf"')).not.toBeInTheDocument()
})

it('prefills the first contract (no auto-send) when multiple are uploaded', async () => {
  mockFetchUpload('done', 2)
  const onStartQuery = vi.fn()

  const { container } = render(<Harness appName="app1" onStartQuery={onStartQuery} />)
  await uploadFiles(container, ['Acme_MSA.pdf', 'Globex_NDA.pdf'])

  // A single "Review a contract" button, not one-per-file.
  const reviewBtn = await screen.findByText('⚖ Review a contract', {}, { timeout: 4000 })
  await userEvent.click(reviewBtn)

  expect(onStartQuery).toHaveBeenCalledTimes(1)
  const [query, send] = onStartQuery.mock.calls[0]
  expect(query).toContain('Acme_MSA.pdf') // scoped to the first, one at a time
  expect(query).not.toContain('Globex_NDA.pdf')
  expect(send).toBe(false) // multiple → prefill only, user picks then sends
})

it('the CTA "Ask a question" jumps to Query without seeding a query', async () => {
  mockFetchUpload('done', 1)
  const onStartQuery = vi.fn()

  const { container } = render(<Harness appName="app1" onStartQuery={onStartQuery} />)
  await uploadFiles(container, ['MSA_Acme.pdf'])

  const askBtn = await screen.findByText('💬 Ask a question', {}, { timeout: 4000 })
  await userEvent.click(askBtn)

  expect(onStartQuery).toHaveBeenCalledWith(null)
})

it('does not offer a review CTA when the ingest fails', async () => {
  mockFetchUpload('failed', 1)
  const onStartQuery = vi.fn()

  const { container } = render(<Harness appName="app1" onStartQuery={onStartQuery} />)
  await uploadFiles(container, ['MSA_Acme.pdf'])

  // The failed-task row confirms the upload cycle finished; no CTA should show.
  await screen.findByText('doc-1', {}, { timeout: 4000 })
  expect(screen.queryByText(/Review/)).not.toBeInTheDocument()
  expect(onStartQuery).not.toHaveBeenCalled()
})

it('alerts when the original download fails', async () => {
  const { downloadCalls } = mockFetchDownloadable(
    [{ doc_id: 'doc-alpha', metadata: {}, status: 'done' }],
    { ok: false, status: 404, statusText: 'Not Found' },
  )
  const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})

  render(<Harness appName="app1" />)
  await waitFor(() => expect(screen.getByText('doc-alpha')).toBeInTheDocument())

  await userEvent.click(screen.getByText('⤓ Download'))

  await waitFor(() => expect(alertSpy).toHaveBeenCalledTimes(1))
  expect(alertSpy.mock.calls[0][0]).toContain('Download failed')
  expect(downloadCalls).toHaveLength(1)
})
