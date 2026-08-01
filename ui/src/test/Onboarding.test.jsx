import React, { useEffect } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useApp } from '../context'
import { I18nProvider } from '../i18n'
import { renderWithCtx } from './renderWithCtx'
import QueryTab from '../components/tabs/QueryTab'
import SettingsTab from '../components/tabs/SettingsTab'
import OnboardingModal from '../components/modals/OnboardingModal'

// The company profile is fetched by the mounted Layout (App.jsx), not by the tabs
// — so a standalone tab test has to drive that read itself, exactly as Layout does.
function LoadProfile() {
  const { refreshProfile } = useApp()
  useEffect(() => { refreshProfile() }, [refreshProfile])
  return null
}

function SetApp({ name }) {
  const { setCurrentApp } = useApp()
  useEffect(() => { setCurrentApp(name) }, [name, setCurrentApp])
  return null
}

// A streaming Response body carrying the given SSE data objects.
function sseResponse(events) {
  const bytes = new TextEncoder().encode(events.map(e => `data: ${JSON.stringify(e)}\n\n`).join(''))
  let sent = false
  return {
    ok: true,
    body: { getReader: () => ({ read: () => {
      if (sent) return Promise.resolve({ done: true, value: undefined })
      sent = true
      return Promise.resolve({ done: false, value: bytes })
    } }) },
  }
}

// Same, but the body stays unread until `release()` — a turn caught mid-stream.
function deferredSse(events) {
  let release
  const gate = new Promise(r => { release = r })
  const bytes = new TextEncoder().encode(events.map(e => `data: ${JSON.stringify(e)}\n\n`).join(''))
  let sent = false
  return {
    release: () => release(),
    response: {
      ok: true,
      body: { getReader: () => ({ read: async () => {
        if (sent) return { done: true, value: undefined }
        await gate
        sent = true
        return { done: false, value: bytes }
      } }) },
    },
  }
}

const interviewCalls = (spy) =>
  spy.mock.calls.filter(([u]) => String(u).endsWith('/profile/interview/chat/stream'))

// GET /profile answers with `profile`; the query stream answers one canned turn;
// everything else is an empty-but-ok body.
function mockFetch({ profile = { markdown: null, exists: false }, interview = null, profileStatus = 200 } = {}) {
  return vi.spyOn(global, 'fetch').mockImplementation((url, opts = {}) => {
    const u = String(url)
    const method = (opts.method || 'GET').toUpperCase()
    if (u.endsWith('/profile/interview/chat/stream')) {
      if (interview && interview.status && interview.status !== 200) {
        return Promise.resolve({ ok: false, status: interview.status, text: async () => 'no interview script' })
      }
      return Promise.resolve(sseResponse(interview?.events || []))
    }
    if (u.endsWith('/profile')) {
      if (method === 'PUT') {
        const markdown = JSON.parse(opts.body).markdown
        return Promise.resolve({ ok: true, json: async () => ({ markdown, exists: true, source: 'manual', updated_at: '2026-07-31T10:00:00Z' }) })
      }
      if (method === 'DELETE') return Promise.resolve({ ok: true, status: 204, json: async () => ({}) })
      if (profileStatus !== 200) return Promise.resolve({ ok: false, status: profileStatus, json: async () => ({}) })
      return Promise.resolve({ ok: true, json: async () => profile })
    }
    if (u.endsWith('/sessions') && method === 'POST') {
      return Promise.resolve({ ok: true, json: async () => ({ session_id: 'sess-1' }) })
    }
    if (u.endsWith('/query/stream')) {
      return Promise.resolve(sseResponse([{ result: { answer: 'The term auto-renews.', references: {} } }]))
    }
    return Promise.resolve({ ok: true, json: async () => ({}), text: async () => '' })
  })
}

function renderQuery(opts) {
  mockFetch(opts)
  return render(
    <I18nProvider><AppProvider>
      <LoadProfile />
      <SetApp name="contract-analyst" />
      <QueryTab active={true} onOpenOnboarding={opts?.onOpenOnboarding} />
    </AppProvider></I18nProvider>
  )
}

async function ask(user, text = 'Review the contract') {
  await user.type(screen.getByPlaceholderText(/Ask a question/), text)
  await user.click(screen.getByText('Send'))
}

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

describe('onboarding card — Query tab, after the first answer', () => {
  it('stays hidden until an answer has landed, then offers the interview', async () => {
    const user = userEvent.setup()
    renderQuery()
    // Profile is loaded and cold (exists: false), but nothing has been asked yet —
    // the ask has to trade on value already delivered.
    await waitFor(() => expect(screen.queryByText(/No app selected/)).not.toBeInTheDocument())
    expect(screen.queryByText('Tell us about your company')).not.toBeInTheDocument()

    await ask(user)
    await waitFor(() => expect(screen.getByText('The term auto-renews.')).toBeInTheDocument())
    expect(await screen.findByText('Tell us about your company')).toBeInTheDocument()
  })

  it('stays hidden for an account that already has a profile', async () => {
    const user = userEvent.setup()
    renderQuery({ profile: { markdown: '## Who we are', exists: true } })
    await ask(user)
    await waitFor(() => expect(screen.getByText('The term auto-renews.')).toBeInTheDocument())
    expect(screen.queryByText('Tell us about your company')).not.toBeInTheDocument()
  })

  it('stays hidden when the profile state is unknown (the route 503s)', async () => {
    // A deployment with no document store can't hold profiles at all — offering an
    // interview that would 503 on open is worse than not asking.
    const user = userEvent.setup()
    renderQuery({ profileStatus: 503 })
    await ask(user)
    await waitFor(() => expect(screen.getByText('The term auto-renews.')).toBeInTheDocument())
    expect(screen.queryByText('Tell us about your company')).not.toBeInTheDocument()
  })

  it('"Not now" dismisses the card and remembers the deferral for the account', async () => {
    const user = userEvent.setup()
    renderQuery()
    await ask(user)
    await screen.findByText('Tell us about your company')
    await user.click(screen.getByText('Not now'))
    expect(screen.queryByText('Tell us about your company')).not.toBeInTheDocument()
    expect(localStorage.getItem('cogbase.onboardingDismissed.default')).toBe('1')
  })

  it('opens the interview when the card is accepted', async () => {
    const onOpenOnboarding = vi.fn()
    const user = userEvent.setup()
    renderQuery({ onOpenOnboarding })
    await ask(user)
    await user.click(await screen.findByText('Tell us about your company'))
    expect(onOpenOnboarding).toHaveBeenCalled()
  })
})

describe('onboarding interview modal', () => {
  it('opens with the model talking first, without showing a message the user did not write', async () => {
    const fetchSpy = mockFetch({
      interview: { events: [{ token: 'Two minutes or the full tour?' }, { done: true, content: 'Two minutes or the full tour?' }] },
    })
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)

    await waitFor(() => expect(screen.getByText('Two minutes or the full tour?')).toBeInTheDocument())
    // The kickoff went to the server…
    const call = fetchSpy.mock.calls.find(([u]) => String(u).endsWith('/profile/interview/chat/stream'))
    expect(JSON.parse(call[1].body).text).toMatch(/company profile/i)
    expect(JSON.parse(call[1].body).history).toEqual([])
    // …but is not rendered as one of the user's own turns.
    expect(screen.queryByText(/I'd like to set up my company profile/)).not.toBeInTheDocument()
  })

  it('threads the history back on the next turn', async () => {
    const fetchSpy = mockFetch({
      interview: { events: [{ done: true, content: 'Where do you operate?' }] },
    })
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)
    await screen.findByText('Where do you operate?')

    await user.type(screen.getByPlaceholderText(/Type your answer/), 'EU and UK')
    await user.click(screen.getByText('Send'))

    await waitFor(() => {
      const calls = fetchSpy.mock.calls.filter(([u]) => String(u).endsWith('/profile/interview/chat/stream'))
      expect(calls).toHaveLength(2)
      const body = JSON.parse(calls[1][1].body)
      expect(body.text).toBe('EU and UK')
      // The hidden kickoff still has to be in the history — the server needs a
      // coherent user/assistant alternation even for a turn we never showed.
      expect(body.history).toHaveLength(2)
      expect(body.history[1]).toEqual({ role: 'assistant', content: 'Where do you operate?' })
    })
  })

  it('adopts the saved profile so the onboarding card stops being offered', async () => {
    mockFetch({
      interview: { events: [{ done: true, content: 'Got it.', profile_saved: true, markdown: '## Who we are\nA 40-person SaaS company.' }] },
    })
    // Both surfaces share one provider, so the save must reach the Settings editor
    // without a re-read of GET /profile.
    render(
      <I18nProvider><AppProvider>
        <OnboardingModal open={true} onClose={() => {}} />
        <SettingsTab active={false} />
      </AppProvider></I18nProvider>
    )
    await waitFor(() => expect(screen.getByText(/Saved — every app in this account/)).toBeInTheDocument())
    await waitFor(() =>
      expect(screen.getByDisplayValue(/A 40-person SaaS company\./)).toBeInTheDocument()
    )
  })

  it('banks answers on demand and lets the interview carry on', async () => {
    // Only the third turn — the one the button triggers — writes the profile.
    let n = 0
    const fetchSpy = vi.spyOn(global, 'fetch').mockImplementation((url) => {
      if (!String(url).endsWith('/profile/interview/chat/stream')) {
        return Promise.resolve({ ok: true, json: async () => ({ markdown: null, exists: false }), text: async () => '' })
      }
      n += 1
      return Promise.resolve(sseResponse(n === 3
        ? [{ done: true, content: 'Saved.', profile_saved: true, markdown: '## Who we are' }]
        : [{ done: true, content: 'Got it.' }]))
    })
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)
    await screen.findByText('Got it.')
    // Nothing said yet, so there is nothing to bank — the button must not be able
    // to write a profile out of an empty conversation.
    expect(screen.getByText('Save')).toBeDisabled()

    await user.type(screen.getByPlaceholderText(/Type your answer/), 'A 40-person SaaS company')
    await user.click(screen.getByText('Send'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(2))

    await user.click(screen.getByText('Save'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(3))
    const body = JSON.parse(interviewCalls(fetchSpy)[2][1].body)
    expect(body.text).toMatch(/save what you have/i)
    // Shown as the user's own turn: they pressed a button that means this.
    expect(screen.getByText('Save what you have so far.')).toBeInTheDocument()
    // Saving does not end the interview — the input is still live.
    expect(screen.getByPlaceholderText(/Type your answer/)).not.toBeDisabled()
    // …and there is nothing left to bank until they say something new.
    await waitFor(() => expect(screen.getByText('Save')).toBeDisabled())
  })

  it('spends one last turn saving the answers when the user finishes later', async () => {
    const fetchSpy = mockFetch({ interview: { events: [{ done: true, content: 'Where do you operate?' }] } })
    const onClose = vi.fn()
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={onClose} />)
    await screen.findByText('Where do you operate?')

    await user.type(screen.getByPlaceholderText(/Type your answer/), 'EU and UK')
    await user.click(screen.getByText('Send'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(2))

    await user.click(screen.getByText('Finish later'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(3))
    // Only once the save has landed — closing first would drop the user back on a
    // profile card that says nothing was saved, then rewrite it under them.
    await waitFor(() => expect(onClose).toHaveBeenCalled())
    const body = JSON.parse(interviewCalls(fetchSpy)[2][1].body)
    expect(body.text).toMatch(/save what you have/i)
    // Everything answered so far has to ride along, or there is nothing to save.
    expect(body.history).toHaveLength(4)
    expect(body.history[2]).toEqual({ role: 'user', content: 'EU and UK' })
  })

  it('does not save when the user leaves without answering anything', async () => {
    // Only the hidden kickoff has run. Asking the model to save here would have it
    // write a profile from nothing — which reads as "done" everywhere in the UI.
    const fetchSpy = mockFetch({ interview: { events: [{ done: true, content: 'What does your org do?' }] } })
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)
    await screen.findByText('What does your org do?')

    await user.click(screen.getByText('Finish later'))
    await new Promise(r => setTimeout(r, 20))
    expect(interviewCalls(fetchSpy)).toHaveLength(1)
  })

  it('does not save again when the last turn already saved', async () => {
    const fetchSpy = mockFetch({
      interview: { events: [{ done: true, content: 'Got it.', profile_saved: true, markdown: '## Who we are' }] },
    })
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)
    await screen.findByText('Got it.')

    await user.type(screen.getByPlaceholderText(/Type your answer/), 'A 40-person SaaS company')
    await user.click(screen.getByText('Send'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(2))

    await user.click(screen.getByText('Done'))
    await new Promise(r => setTimeout(r, 20))
    expect(interviewCalls(fetchSpy)).toHaveLength(2)
  })

  it('keeps the user in the interview when the closing save fails, instead of losing it quietly', async () => {
    let n = 0
    const fetchSpy = vi.spyOn(global, 'fetch').mockImplementation((url) => {
      if (!String(url).endsWith('/profile/interview/chat/stream')) {
        return Promise.resolve({ ok: true, json: async () => ({ markdown: null, exists: false }), text: async () => '' })
      }
      n += 1
      // The closing turn is the one that fails.
      if (n === 3) return Promise.reject(new Error('network down'))
      return Promise.resolve(sseResponse([{ done: true, content: 'Where do you operate?' }]))
    })
    const onClose = vi.fn()
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={onClose} />)
    await screen.findByText('Where do you operate?')

    await user.type(screen.getByPlaceholderText(/Type your answer/), 'EU and UK')
    await user.click(screen.getByText('Send'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(2))

    await user.click(screen.getByText('Finish later'))
    // Closing here would drop the answers with nobody the wiser.
    await waitFor(() => expect(screen.getByText(/Couldn't save your answers/)).toBeInTheDocument())
    expect(onClose).not.toHaveBeenCalled()
    // Save is the retry; leaving is now a deliberate choice with the cost stated.
    expect(screen.getByText('Save')).not.toBeDisabled()
    await user.click(screen.getByText('Close anyway'))
    expect(onClose).toHaveBeenCalled()
  })

  it('waits for a turn still streaming, so the answer typed just before leaving is saved', async () => {
    // The expensive moment: the user answers, then closes while the model is still
    // replying. That answer is only in the in-flight request, not yet in history.
    const gate = deferredSse([{ done: true, content: 'Noted.' }])
    let n = 0
    const fetchSpy = vi.spyOn(global, 'fetch').mockImplementation((url) => {
      if (!String(url).endsWith('/profile/interview/chat/stream')) {
        return Promise.resolve({ ok: true, json: async () => ({ markdown: null, exists: false }), text: async () => '' })
      }
      n += 1
      if (n === 2) return Promise.resolve(gate.response)
      return Promise.resolve(sseResponse([{ done: true, content: 'Where do you operate?' }]))
    })
    const user = userEvent.setup()
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)
    await screen.findByText('Where do you operate?')

    await user.type(screen.getByPlaceholderText(/Type your answer/), 'EU and UK')
    await user.click(screen.getByText('Send'))
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(2))

    await user.click(screen.getByText('Finish later'))
    // Nothing is sent while that turn is unresolved — a save now would omit it.
    expect(interviewCalls(fetchSpy)).toHaveLength(2)
    // And the wait is visible: the modal is still up, saying what it is doing.
    expect(screen.getByText('Saving your answers…')).toBeInTheDocument()

    gate.release()
    await waitFor(() => expect(interviewCalls(fetchSpy)).toHaveLength(3))
    const body = JSON.parse(interviewCalls(fetchSpy)[2][1].body)
    expect(body.history).toHaveLength(4)
    expect(body.history[2]).toEqual({ role: 'user', content: 'EU and UK' })
    expect(body.history[3]).toEqual({ role: 'assistant', content: 'Noted.' })
  })

  it('reports a deployment with no interview script instead of failing silently', async () => {
    mockFetch({ interview: { status: 503 } })
    renderWithCtx(<OnboardingModal open={true} onClose={() => {}} />)
    await waitFor(() =>
      expect(screen.getByText(/onboarding interview is not available/)).toBeInTheDocument()
    )
  })
})

describe('Settings — company profile card', () => {
  it('nudges an account with no profile and offers the interview', async () => {
    const onOpenOnboarding = vi.fn()
    mockFetch()
    const user = userEvent.setup()
    renderWithCtx(<SettingsTab active={true} onOpenOnboarding={onOpenOnboarding} />)

    expect(await screen.findByText('Complete your profile')).toBeInTheDocument()
    await user.click(screen.getByText('Start the interview'))
    expect(onOpenOnboarding).toHaveBeenCalled()
    // Nothing to re-run or delete before there is a profile.
    expect(screen.queryByText('Re-run the interview')).not.toBeInTheDocument()
  })

  it('loads the saved profile and PUTs an edit', async () => {
    const fetchSpy = mockFetch({ profile: { markdown: '## Who we are', exists: true, source: 'interview', updated_at: '2026-07-31T10:00:00Z' } })
    const user = userEvent.setup()
    renderWithCtx(<SettingsTab active={true} />)

    const box = await screen.findByDisplayValue('## Who we are')
    // Saving is disabled until something actually changes.
    expect(screen.getByText('Save profile')).toBeDisabled()
    await user.type(box, ' — a SaaS company')
    await user.click(screen.getByText('Save profile'))

    await waitFor(() => {
      const put = fetchSpy.mock.calls.find(([u, o]) => String(u).endsWith('/profile') && o?.method === 'PUT')
      expect(JSON.parse(put[1].body).markdown).toBe('## Who we are — a SaaS company')
    })
    expect(await screen.findByText('Saved')).toBeInTheDocument()
  })

  it('reverts an unsaved edit on Reset', async () => {
    mockFetch({ profile: { markdown: '## Who we are', exists: true } })
    const user = userEvent.setup()
    renderWithCtx(<SettingsTab active={true} />)
    const box = await screen.findByDisplayValue('## Who we are')
    await user.type(box, ' oops')
    await user.click(screen.getByText('Reset'))
    expect(box).toHaveValue('## Who we are')
  })

  it('says so when the deployment cannot store profiles at all', async () => {
    mockFetch({ profileStatus: 503 })
    renderWithCtx(<SettingsTab active={true} />)
    expect(await screen.findByText(/cannot store company profiles/)).toBeInTheDocument()
    expect(screen.queryByText('Save profile')).not.toBeInTheDocument()
  })
})
