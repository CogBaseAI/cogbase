import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render } from '@testing-library/react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../context'
import { I18nProvider } from '../i18n'
import QueryTab from '../components/tabs/QueryTab'

// Render QueryTab inside a provider with `currentApp` pre-selected.
function SetApp({ name }) {
  const { setCurrentApp } = useApp()
  useEffect(() => { setCurrentApp(name) }, [name, setCurrentApp])
  return null
}

function renderQueryTab(appName = 'contract-analyst') {
  return render(
    <I18nProvider>
      <AppProvider>
        <SetApp name={appName} />
        <QueryTab active={true} />
      </AppProvider>
    </I18nProvider>
  )
}

// Build a fake streaming Response body that emits the given SSE data objects.
function sseResponse(events) {
  const lines = events.map(e => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n'
  const bytes = new TextEncoder().encode(lines)
  let sent = false
  return {
    ok: true,
    body: {
      getReader() {
        return {
          read() {
            if (sent) return Promise.resolve({ done: true, value: undefined })
            sent = true
            return Promise.resolve({ done: false, value: bytes })
          },
        }
      },
    },
  }
}

// Route fetch by URL + method: list sessions (GET), create session (POST),
// load a transcript (GET /sessions/{id}), close (POST .../close), and the query
// stream. `sessions` is what the history sidebar loads; `transcript` is returned
// when a past chat is opened.
function mockFetch({
  sessionId = 'sess-1',
  streamEvents = [{ result: { answer: 'Hello there', references: { chunks: [], structured_records: [] } } }],
  sessions = [],
  transcript = { messages: [] },
} = {}) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts = {}) => {
    const u = String(url)
    const method = (opts.method || 'GET').toUpperCase()
    if (u.includes('/sessions/') && u.endsWith('/close')) {
      return Promise.resolve({ ok: true, json: async () => ({ session_id: sessionId, distillation: 'enqueued' }) })
    }
    if (u.includes('/sessions/') && method === 'DELETE') {
      const sid = decodeURIComponent(u.split('/sessions/')[1])
      return Promise.resolve({ ok: true, json: async () => ({ session_id: sid, deleted: true }) })
    }
    if (u.endsWith('/sessions') && method === 'POST') {
      return Promise.resolve({ ok: true, json: async () => ({ session_id: sessionId }) })
    }
    if (u.endsWith('/sessions') && method === 'GET') {
      return Promise.resolve({ ok: true, json: async () => ({ sessions }) })
    }
    if (u.includes('/sessions/') && method === 'GET') {
      const sid = decodeURIComponent(u.split('/sessions/')[1])
      return Promise.resolve({ ok: true, json: async () => ({ session_id: sid, ...transcript }) })
    }
    if (u.endsWith('/query/stream')) {
      return Promise.resolve(sseResponse(streamEvents))
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
}

// Session-start (POST /sessions) calls only — filters out the GET list calls the
// history sidebar makes against the same URL.
function startSessionCalls(fetchSpy) {
  return fetchSpy.mock.calls.filter(
    ([u, o]) => String(u).endsWith('/sessions') && (o?.method || 'GET').toUpperCase() === 'POST'
  )
}

async function ask(user, text) {
  const box = screen.getByPlaceholderText(/Ask a question/)
  await user.type(box, text)
  await user.click(screen.getByText('Send'))
}

afterEach(() => vi.restoreAllMocks())

it('starts a session on the first question and threads session_id into the query', async () => {
  const fetchSpy = mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'What is the term?')

  // start_session was called once.
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/applications\/contract-analyst\/sessions$/),
      expect.objectContaining({ method: 'POST' })
    )
  )

  // The stream carried the new session_id and no history field.
  const streamCall = fetchSpy.mock.calls.find(([u]) => String(u).endsWith('/query/stream'))
  expect(streamCall).toBeTruthy()
  const body = JSON.parse(streamCall[1].body)
  expect(body.session_id).toBe('sess-1')
  expect(body.text).toBe('What is the term?')
  expect(body).not.toHaveProperty('history')

  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())
})

it('renders a markdown table answer as an HTML table', async () => {
  const table = '| Name | Term |\n| --- | --- |\n| Acme | 12 months |\n| Globex | 24 months |'
  mockFetch({ streamEvents: [{ result: { answer: table, references: { chunks: [], structured_records: [] } } }] })
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'summarize the contracts')

  // remark-gfm turns the pipe syntax into a real <table>, not raw text.
  const tableEl = await waitFor(() => document.querySelector('.md table'))
  expect(tableEl).toBeTruthy()
  expect(tableEl.querySelectorAll('th')).toHaveLength(2)
  expect(tableEl.querySelectorAll('tbody tr')).toHaveLength(2)
  expect(screen.getByRole('columnheader', { name: 'Name' })).toBeInTheDocument()
  expect(screen.getByRole('cell', { name: 'Globex' })).toBeInTheDocument()
  // The raw pipe markup must not leak through as text.
  expect(screen.queryByText(/\| --- \|/)).not.toBeInTheDocument()
})

it('renders a skill download path as an absolute, app-resolved clickable link', async () => {
  const answer =
    'Done — merged file produced.\n\nDownload it here:\n' +
    '`/applications/<app_name>/documents/lease-merged__27dd52c2.docx/download`'
  mockFetch({ streamEvents: [{ result: { answer, references: { chunks: [], structured_records: [] } } }] })
  const user = userEvent.setup()
  renderQueryTab('contract-analyst')
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'apply the amendment')

  // The <app_name> placeholder is resolved to the current app, the path is
  // prefixed with the API origin (jsdom: http://localhost:3000), and it becomes
  // a real anchor (not raw text).
  const dlUrl = `${window.location.origin}/applications/contract-analyst/documents/lease-merged__27dd52c2.docx/download`
  const link = await waitFor(() => document.querySelector('.md a[href*="/download"]'))
  expect(link.getAttribute('href')).toBe(dlUrl)
  // The unresolved placeholder must not leak through as visible text.
  expect(screen.queryByText(/<app_name>/)).not.toBeInTheDocument()

  // Clicking fetches the artifact and saves it via a throwaway object-URL
  // anchor, using the server-provided filename — no bare navigation.
  const blob = new Blob(['docx'], { type: 'application/octet-stream' })
  const dlSpy = vi.spyOn(global, 'fetch').mockResolvedValueOnce({
    ok: true,
    blob: async () => blob,
    headers: { get: () => 'attachment; filename="lease-merged.docx"' },
  })
  // jsdom lacks the object-URL APIs; provide them so the download path runs.
  const createObjSpy = vi.fn().mockReturnValue('blob:mock')
  URL.createObjectURL = createObjSpy
  URL.revokeObjectURL = vi.fn()
  const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})

  await user.click(link)

  await waitFor(() => expect(dlSpy).toHaveBeenCalledWith(dlUrl))
  expect(createObjSpy).toHaveBeenCalledWith(blob)
  // A save anchor was clicked carrying the server-provided filename.
  const saved = clickSpy.mock.instances.find(a => a.download)
  expect(saved.download).toBe('lease-merged.docx')
})

it('copies the question and the answer via their copy buttons', async () => {
  const writeText = vi.fn().mockResolvedValue(undefined)
  mockFetch({ streamEvents: [{ result: { answer: 'The term is 12 months.', references: { chunks: [], structured_records: [] } } }] })
  const user = userEvent.setup()
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'What is the term?')
  await waitFor(() => expect(screen.getByText('The term is 12 months.')).toBeInTheDocument())

  // Each user and bot message exposes a Copy button.
  const copyButtons = screen.getAllByRole('button', { name: 'Copy' })
  expect(copyButtons).toHaveLength(2)

  await user.click(copyButtons[0]) // question
  await waitFor(() => expect(writeText).toHaveBeenLastCalledWith('What is the term?'))
  await user.click(copyButtons[1]) // answer
  await waitFor(() => expect(writeText).toHaveBeenLastCalledWith('The term is 12 months.'))
})

it('copies a markdown table as tab-separated rows', async () => {
  const writeText = vi.fn().mockResolvedValue(undefined)
  const table = '| Name | Term |\n| --- | --- |\n| Acme | 12 months |\n| Globex | 24 months |'
  mockFetch({ streamEvents: [{ result: { answer: table, references: { chunks: [], structured_records: [] } } }] })
  const user = userEvent.setup()
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'summarize the contracts')
  await waitFor(() => expect(document.querySelector('.md table')).toBeTruthy())

  await user.click(screen.getByRole('button', { name: 'Copy table' }))
  await waitFor(() => expect(writeText).toHaveBeenLastCalledWith(
    'Name\tTerm\nAcme\t12 months\nGlobex\t24 months'
  ))
})

it('reuses the same session across multiple questions', async () => {
  const fetchSpy = mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'first')
  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())
  await ask(user, 'second')

  expect(startSessionCalls(fetchSpy)).toHaveLength(1)
})

it('closes the session and opens a new one after New chat', async () => {
  const fetchSpy = mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'first')
  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())

  // "New chat" (the ↺ button) closes the current session.
  await user.click(screen.getByTitle('+ New chat'))
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/sessions\/sess-1\/close$/),
      expect.objectContaining({ method: 'POST' })
    )
  )

  // Next question opens a fresh session.
  await ask(user, 'second')
  expect(startSessionCalls(fetchSpy)).toHaveLength(2)
})

const SESSIONS_FIXTURE = [
  { session_id: 's-1', title: 'What is the term?', message_count: 3, status: 'closed', created_at: '2026-07-01T00:00:00Z', updated_at: '2026-07-02T00:00:00Z' },
  { session_id: 's-2', title: 'Renewal dates', message_count: 1, status: 'open', created_at: '2026-07-03T00:00:00Z', updated_at: '2026-07-03T00:00:00Z' },
]

it('loads the chat history sidebar on app select', async () => {
  const fetchSpy = mockFetch({ sessions: SESSIONS_FIXTURE })
  renderQueryTab()

  // Both past chats are listed, titled by their first message.
  await waitFor(() => expect(screen.getByText('What is the term?')).toBeInTheDocument())
  expect(screen.getByText('Renewal dates')).toBeInTheDocument()

  // The list was fetched with a GET against /sessions.
  expect(
    fetchSpy.mock.calls.some(
      ([u, o]) => String(u).endsWith('/sessions') && (o?.method || 'GET').toUpperCase() === 'GET'
    )
  ).toBe(true)
})

it('shows an empty-state message when there are no past chats', async () => {
  mockFetch({ sessions: [] })
  renderQueryTab()
  await waitFor(() => expect(screen.getByText(/No chats yet/)).toBeInTheDocument())
})

it('opens a past chat, loads its transcript, and resumes it on the next question', async () => {
  const fetchSpy = mockFetch({
    sessions: SESSIONS_FIXTURE,
    transcript: { messages: [
      { role: 'user', content: 'old question' },
      { role: 'assistant', content: 'old answer' },
    ] },
  })
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.getByText('What is the term?')).toBeInTheDocument())

  // Click a past chat -> its transcript replaces the chat view.
  await user.click(screen.getByText('What is the term?'))
  await waitFor(() => expect(screen.getByText('old question')).toBeInTheDocument())
  expect(screen.getByText('old answer')).toBeInTheDocument()

  // The transcript was loaded from GET /sessions/{id}.
  expect(
    fetchSpy.mock.calls.some(([u, o]) => String(u).endsWith('/sessions/s-1') && (o?.method || 'GET').toUpperCase() === 'GET')
  ).toBe(true)

  // The next question resumes the opened session — no new session is started.
  await ask(user, 'follow up')
  const streamCall = fetchSpy.mock.calls.find(([u]) => String(u).endsWith('/query/stream'))
  expect(JSON.parse(streamCall[1].body).session_id).toBe('s-1')
  expect(startSessionCalls(fetchSpy)).toHaveLength(0)
})

it('restores the references pane from a past chat transcript', async () => {
  mockFetch({
    sessions: SESSIONS_FIXTURE,
    transcript: { messages: [
      { role: 'user', content: 'old question', references: null },
      {
        role: 'assistant',
        content: 'old answer',
        references: {
          structured_records: [{ contract_type: 'NDA' }],
          chunks: [{ chunk_id: 'doc_0_0', doc_id: 'doc_0', text: 'net 30 payment terms' }],
          document_slices: [],
          memories: [],
        },
      },
    ] },
  })
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.getByText('What is the term?')).toBeInTheDocument())

  await user.click(screen.getByText('What is the term?'))
  await waitFor(() => expect(screen.getByText('old answer')).toBeInTheDocument())

  // The assistant turn's references are surfaced in the pane, not left empty.
  expect(screen.getByText('net 30 payment terms')).toBeInTheDocument()
  expect(screen.getByText(/NDA/)).toBeInTheDocument()
})

it('switches the references pane to the clicked answer in a multi-turn chat', async () => {
  mockFetch({
    sessions: SESSIONS_FIXTURE,
    transcript: { messages: [
      { role: 'user', content: 'first question', references: null },
      {
        role: 'assistant',
        content: 'first answer',
        references: {
          structured_records: [],
          chunks: [{ chunk_id: 'c-1', doc_id: 'd-1', text: 'evidence for the first answer' }],
          document_slices: [], memories: [],
        },
      },
      { role: 'user', content: 'second question', references: null },
      {
        role: 'assistant',
        content: 'second answer',
        references: {
          structured_records: [],
          chunks: [{ chunk_id: 'c-2', doc_id: 'd-2', text: 'evidence for the second answer' }],
          document_slices: [], memories: [],
        },
      },
    ] },
  })
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.getByText('What is the term?')).toBeInTheDocument())
  await user.click(screen.getByText('What is the term?'))

  // Loads defaulting to the latest answer's references.
  await waitFor(() => expect(screen.getByText('evidence for the second answer')).toBeInTheDocument())
  expect(screen.queryByText('evidence for the first answer')).not.toBeInTheDocument()

  // Clicking the earlier answer re-points the pane at that turn's evidence.
  await user.click(screen.getByText('first answer'))
  await waitFor(() => expect(screen.getByText('evidence for the first answer')).toBeInTheDocument())
  expect(screen.queryByText('evidence for the second answer')).not.toBeInTheDocument()
})

// DELETE /sessions/{id} calls only.
function deleteSessionCalls(fetchSpy) {
  return fetchSpy.mock.calls.filter(
    ([u, o]) => String(u).includes('/sessions/') && (o?.method || 'GET').toUpperCase() === 'DELETE'
  )
}

it('deletes a past chat after confirmation and refreshes the sidebar', async () => {
  const fetchSpy = mockFetch({ sessions: SESSIONS_FIXTURE })
  const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.getByText('What is the term?')).toBeInTheDocument())

  // Each history row exposes a delete button; click the first one.
  const delButtons = screen.getAllByRole('button', { name: 'Delete chat' })
  expect(delButtons).toHaveLength(2)
  await user.click(delButtons[0])

  // The user was asked to confirm, and a DELETE hit /sessions/s-1.
  expect(confirmSpy).toHaveBeenCalled()
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/sessions\/s-1$/),
      expect.objectContaining({ method: 'DELETE' })
    )
  )

  // The sidebar was re-fetched after the delete (an extra GET /sessions).
  const listGets = fetchSpy.mock.calls.filter(
    ([u, o]) => String(u).endsWith('/sessions') && (o?.method || 'GET').toUpperCase() === 'GET'
  )
  expect(listGets.length).toBeGreaterThanOrEqual(2)
})

it('does not delete a chat when the user cancels the confirm', async () => {
  const fetchSpy = mockFetch({ sessions: SESSIONS_FIXTURE })
  vi.spyOn(window, 'confirm').mockReturnValue(false)
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.getByText('What is the term?')).toBeInTheDocument())

  await user.click(screen.getAllByRole('button', { name: 'Delete chat' })[0])

  // No DELETE request was issued.
  expect(deleteSessionCalls(fetchSpy)).toHaveLength(0)
})

it('resets the chat view and drops the handle when deleting the active session', async () => {
  // The history sidebar lists the active session (sess-1, titled by its first
  // message) so it can be deleted from the list. A follow-up POST returns a new id.
  const fetchSpy = mockFetch({
    sessions: [
      { session_id: 'sess-1', title: 'hi', message_count: 1, status: 'open', created_at: '2026-07-03T00:00:00Z', updated_at: '2026-07-03T00:00:00Z' },
    ],
  })
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  // Ask a question -> opens the active session (sess-1) and renders the answer.
  await ask(user, 'hi')
  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())
  // The active session's row (with its delete button) shows in the sidebar.
  await waitFor(() => expect(screen.getByRole('button', { name: 'Delete chat' })).toBeInTheDocument())

  // Delete the active session from the sidebar.
  await user.click(screen.getByRole('button', { name: 'Delete chat' }))
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/sessions\/sess-1$/),
      expect.objectContaining({ method: 'DELETE' })
    )
  )

  // The transcript view was cleared: the streamed answer is gone.
  await waitFor(() => expect(screen.queryByText('Hello there')).not.toBeInTheDocument())

  // With the active handle dropped, the next question starts a fresh session.
  await ask(user, 'again')
  expect(startSessionCalls(fetchSpy)).toHaveLength(2)
})

it('hides the references pane on a fresh chat and shows it after the first exchange', async () => {
  mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  // No turns yet: the pane (and its collapse toggle) is not rendered.
  expect(screen.queryByText('References')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Hide references panel' })).not.toBeInTheDocument()

  await ask(user, 'What is the term?')
  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())

  // After the exchange the pane appears with its collapse toggle.
  expect(screen.getByText('References')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Hide references panel' })).toBeInTheDocument()
})

it('collapses the references pane to a rail and restores it', async () => {
  mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())
  await ask(user, 'What is the term?')
  await waitFor(() => expect(screen.getByText('References')).toBeInTheDocument())

  // Collapse: the pane header is gone, but the reopen toggle remains (minimized,
  // not fully hidden).
  await user.click(screen.getByRole('button', { name: 'Hide references panel' }))
  expect(screen.queryByText('References')).not.toBeInTheDocument()
  const reopen = screen.getByRole('button', { name: 'Show references panel' })
  expect(reopen).toBeInTheDocument()

  // Restore from the rail.
  await user.click(reopen)
  expect(screen.getByText('References')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Show references panel' })).not.toBeInTheDocument()
})

it('collapses the chats sidebar to a rail and restores it', async () => {
  mockFetch({ sessions: SESSIONS_FIXTURE })
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.getByText('Chats')).toBeInTheDocument())

  // Collapse: the "Chats" header and history rows are gone, reopen toggle remains.
  await user.click(screen.getByRole('button', { name: 'Hide chats panel' }))
  expect(screen.queryByText('Chats')).not.toBeInTheDocument()
  const reopen = screen.getByRole('button', { name: 'Show chats panel' })
  expect(reopen).toBeInTheDocument()

  // Restore from the rail.
  await user.click(reopen)
  expect(screen.getByText('Chats')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Show chats panel' })).not.toBeInTheDocument()
})
