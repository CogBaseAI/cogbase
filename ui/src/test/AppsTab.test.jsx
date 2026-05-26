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
