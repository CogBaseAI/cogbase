import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithCtx } from './renderWithCtx'
import AppsTab from '../components/tabs/AppsTab'

const APPS = [
  { name: 'contract-analyst', status: 'active', created_at: '2024-01-15T10:00:00Z' },
  { name: 'vc-portfolio', status: 'error', created_at: null },
]

beforeEach(() => {
  vi.spyOn(global, 'fetch').mockResolvedValue({
    ok: true,
    json: async () => ({ applications: APPS }),
  })
})

afterEach(() => vi.restoreAllMocks())

it('shows loading state initially', () => {
  renderWithCtx(<AppsTab active={true} />)
  expect(screen.getByText(/Loading/)).toBeInTheDocument()
})

it('renders apps after load', async () => {
  renderWithCtx(<AppsTab active={true} />)
  await waitFor(() => expect(screen.getByText('contract-analyst')).toBeInTheDocument())
  expect(screen.getByText('vc-portfolio')).toBeInTheDocument()
  expect(screen.getByText('active')).toBeInTheDocument()
  expect(screen.getByText('error')).toBeInTheDocument()
})

it('shows empty state when no apps', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ applications: [] }) })
  renderWithCtx(<AppsTab active={true} />)
  await waitFor(() => expect(screen.getByText(/No apps yet/)).toBeInTheDocument())
})

it('shows error when fetch fails', async () => {
  vi.spyOn(global, 'fetch').mockResolvedValue({ ok: false, status: 500, statusText: 'Server Error' })
  renderWithCtx(<AppsTab active={true} />)
  await waitFor(() => expect(screen.getByText(/Failed:/)).toBeInTheDocument())
})

it('does not load when inactive', () => {
  renderWithCtx(<AppsTab active={false} />)
  expect(global.fetch).not.toHaveBeenCalled()
})

describe('detail drawer', () => {
  const APP_WITH_CONFIG = {
    name: 'contract-analyst',
    status: 'active',
    created_at: '2024-01-15T10:00:00Z',
    config: {
      vector_collections: [{ name: 'doc_chunks', description: 'Passage chunks' }],
      structured_collections: [{ name: 'contracts', description: 'Contract records', primary_fields: ['doc_id'], schema: '{"type":"object"}' }],
      pipelines: [{ name: 'contract', routing_description: 'All contracts', steps: [{ tool: 'extract-structured', collection: 'contracts', extractor: { record_mode: 'one', extraction_schema: '{"a":1}', prompt: 'Extract fields.' } }] }],
      workflows: [{ name: 'detect-risks', trigger: { type: 'after_ingest' }, steps: [{ id: 'judge', tool: 'llm-structured', prompt: 'Judge risk.' }] }],
    },
  }

  it('opens the drawer with config sections on name click', async () => {
    vi.spyOn(global, 'fetch')
      .mockResolvedValueOnce({ ok: true, json: async () => ({ applications: APPS }) }) // loadApps
      .mockResolvedValueOnce({ ok: true, json: async () => APP_WITH_CONFIG })          // viewApp fetch

    const user = userEvent.setup()
    renderWithCtx(<AppsTab active={true} />)
    await waitFor(() => screen.getByText('contract-analyst'))
    await user.click(screen.getByText('contract-analyst'))

    await waitFor(() => expect(screen.getByText(/contract-analyst — details/)).toBeInTheDocument())
    expect(screen.getByText('Vector collections')).toBeInTheDocument()
    expect(screen.getByText('Structured collections')).toBeInTheDocument()
    expect(screen.getByText('Pipelines')).toBeInTheDocument()
    expect(screen.getByText('Workflows')).toBeInTheDocument()
  })
})

describe('delete', () => {
  it('calls DELETE and reloads on confirm', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    vi.spyOn(global, 'fetch')
      .mockResolvedValueOnce({ ok: true, json: async () => ({ applications: APPS }) })
      .mockResolvedValueOnce({ ok: true })   // DELETE
      .mockResolvedValueOnce({ ok: true, json: async () => ({ applications: [] }) }) // reload

    const user = userEvent.setup()
    renderWithCtx(<AppsTab active={true} />)
    await waitFor(() => screen.getAllByText('Delete'))
    await user.click(screen.getAllByText('Delete')[0])

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/applications/contract-analyst'),
      expect.objectContaining({ method: 'DELETE' })
    )
  })

  it('does nothing when confirm is cancelled', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false)
    vi.spyOn(global, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ applications: APPS }) })

    const user = userEvent.setup()
    renderWithCtx(<AppsTab active={true} />)
    await waitFor(() => screen.getAllByText('Delete'))
    await user.click(screen.getAllByText('Delete')[0])

    // Only the initial loadApps call, no DELETE
    expect(global.fetch).toHaveBeenCalledTimes(1)
  })
})
