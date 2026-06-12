import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render } from '@testing-library/react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../context'
import QueryTab from '../components/tabs/QueryTab'

// Render QueryTab inside a provider with `currentApp` pre-selected.
function SetApp({ name }) {
  const { setCurrentApp } = useApp()
  useEffect(() => { setCurrentApp(name) }, [name, setCurrentApp])
  return null
}

function renderQueryTab(appName = 'contract-analyst') {
  return render(
    <AppProvider>
      <SetApp name={appName} />
      <QueryTab active={true} />
    </AppProvider>
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

// Route fetch by URL: sessions create, session close, and query stream.
function mockFetch({ sessionId = 'sess-1', streamEvents = [{ result: { answer: 'Hello there', chunks: [], structured_records: [] } }] } = {}) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
    const u = String(url)
    if (u.includes('/sessions/') && u.endsWith('/close')) {
      return Promise.resolve({ ok: true, json: async () => ({ session_id: sessionId, distillation: 'enqueued' }) })
    }
    if (u.endsWith('/sessions')) {
      return Promise.resolve({ ok: true, json: async () => ({ session_id: sessionId }) })
    }
    if (u.endsWith('/query/stream')) {
      return Promise.resolve(sseResponse(streamEvents))
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
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

it('reuses the same session across multiple questions', async () => {
  const fetchSpy = mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'first')
  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())
  await ask(user, 'second')

  const startCalls = fetchSpy.mock.calls.filter(([u]) => String(u).endsWith('/sessions'))
  expect(startCalls).toHaveLength(1)
})

it('closes the session and opens a new one after refresh', async () => {
  const fetchSpy = mockFetch()
  const user = userEvent.setup()
  renderQueryTab()
  await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())

  await ask(user, 'first')
  await waitFor(() => expect(screen.getByText('Hello there')).toBeInTheDocument())

  // Refresh closes the current session.
  await user.click(screen.getByTitle('Refresh session'))
  await waitFor(() =>
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/sessions\/sess-1\/close$/),
      expect.objectContaining({ method: 'POST' })
    )
  )

  // Next question opens a fresh session.
  await ask(user, 'second')
  const startCalls = fetchSpy.mock.calls.filter(([u]) => String(u).endsWith('/sessions'))
  expect(startCalls).toHaveLength(2)
})
