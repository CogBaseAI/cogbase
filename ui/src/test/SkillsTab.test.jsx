import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithCtx } from './renderWithCtx'
import SkillsTab from '../components/tabs/SkillsTab'

const SKILLS = [
  { id: 'aaa111', name: 'pdf-summarizer', description: 'Summarize PDFs', metadata: {} },
  { id: 'bbb222', name: 'table-extract', description: 'Extract tables', metadata: {} },
]

// With no app selected, the tab fetches /skills only (loadAssigned short-circuits).
function mockSkillsOnly() {
  vi.spyOn(global, 'fetch').mockImplementation((url) => {
    if (String(url).endsWith('/skills')) {
      return Promise.resolve({ ok: true, json: async () => ({ skills: SKILLS, total: SKILLS.length }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
}

afterEach(() => vi.restoreAllMocks())

it('shows loading state initially', () => {
  mockSkillsOnly()
  renderWithCtx(<SkillsTab active={true} />)
  expect(screen.getByText(/Loading/)).toBeInTheDocument()
})

it('renders skills after load', async () => {
  mockSkillsOnly()
  renderWithCtx(<SkillsTab active={true} />)
  await waitFor(() => expect(screen.getByText('pdf-summarizer')).toBeInTheDocument())
  expect(screen.getByText('table-extract')).toBeInTheDocument()
  expect(screen.getByText('Summarize PDFs')).toBeInTheDocument()
})

it('shows empty state when no skills', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ skills: [], total: 0 }) })
  renderWithCtx(<SkillsTab active={true} />)
  await waitFor(() => expect(screen.getByText(/No skills yet/)).toBeInTheDocument())
})

it('shows error when fetch fails', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ ok: false, status: 500, statusText: 'Server Error' })
  renderWithCtx(<SkillsTab active={true} />)
  await waitFor(() => expect(screen.getByText(/Failed:/)).toBeInTheDocument())
})

it('does not load when inactive', () => {
  vi.spyOn(global, 'fetch')
  renderWithCtx(<SkillsTab active={false} />)
  expect(global.fetch).not.toHaveBeenCalled()
})

it('disables Add (assign) when no app is selected', async () => {
  mockSkillsOnly()
  renderWithCtx(<SkillsTab active={true} />)
  await waitFor(() => screen.getByText('pdf-summarizer'))
  const addButtons = screen.getAllByText('Add')
  expect(addButtons[0]).toBeDisabled()
})

describe('delete', () => {
  it('calls DELETE and reloads on confirm', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const fetchSpy = vi.spyOn(global, 'fetch').mockImplementation((url, opts) => {
      if (String(url).endsWith('/skills') && (!opts || opts.method === undefined)) {
        return Promise.resolve({ ok: true, json: async () => ({ skills: SKILLS, total: SKILLS.length }) })
      }
      return Promise.resolve({ ok: true, status: 204, json: async () => ({}) })
    })

    const user = userEvent.setup()
    renderWithCtx(<SkillsTab active={true} />)
    await waitFor(() => screen.getAllByText('Delete'))
    await user.click(screen.getAllByText('Delete')[0])

    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/skills/pdf-summarizer'),
      expect.objectContaining({ method: 'DELETE' })
    )
  })

  it('does nothing when confirm is cancelled', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    mockSkillsOnly()

    const user = userEvent.setup()
    renderWithCtx(<SkillsTab active={true} />)
    await waitFor(() => screen.getAllByText('Delete'))
    global.fetch.mockClear()
    await user.click(screen.getAllByText('Delete')[0])

    expect(global.fetch).not.toHaveBeenCalled()
  })
})

describe('built-in skills', () => {
  const WITH_BUILTIN = [
    ...SKILLS,
    { id: 'ccc333', name: 'edit-docx', description: 'Edit docx', metadata: {}, builtin: true },
  ]

  function mockWithBuiltin() {
    vi.spyOn(global, 'fetch').mockImplementation((url) => {
      if (String(url).endsWith('/skills')) {
        return Promise.resolve({ ok: true, json: async () => ({ skills: WITH_BUILTIN, total: WITH_BUILTIN.length }) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })
  }

  it('shows a Built-in badge on built-in skills', async () => {
    mockWithBuiltin()
    renderWithCtx(<SkillsTab active={true} />)
    await waitFor(() => screen.getByText('edit-docx'))
    expect(screen.getByText('Built-in')).toBeInTheDocument()
  })

  it('renders built-in skills read-only (no Replace/Delete actions)', async () => {
    mockWithBuiltin()
    renderWithCtx(<SkillsTab active={true} />)
    await waitFor(() => screen.getByText('edit-docx'))
    // Only the two non-builtin rows expose the mutating actions.
    expect(screen.getAllByText('Replace')).toHaveLength(2)
    expect(screen.getAllByText('Delete')).toHaveLength(2)
  })
})
