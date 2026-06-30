import React from 'react'
import { describe, it, expect, vi } from 'vitest'
import { render as rtlRender, screen, fireEvent } from '@testing-library/react'
import { I18nProvider } from '../i18n'
import DocModal from '../components/modals/DocModal'

// The I18nProvider renders no DOM wrapper, so container.firstChild assertions
// still reference the component's own root element.
const render = (ui, options) => rtlRender(ui, { wrapper: I18nProvider, ...options })

const DOC = {
  demoKey: 'contract-analyst',
  demoName: 'Contract Analyst',
  docId: 'nda-001',
  meta: { doc_type: 'NDA', year: '2024' },
  text: 'This agreement is between...',
}

it('renders nothing when doc is null', () => {
  const { container } = render(<DocModal doc={null} onClose={() => {}} />)
  expect(container.firstChild).toBeNull()
})

it('renders doc content when open', () => {
  render(<DocModal doc={DOC} onClose={() => {}} />)
  // docId appears in the title h3 and again in the kv grid — use getAllByText
  expect(screen.getAllByText('nda-001').length).toBeGreaterThan(0)
  expect(screen.getByText('This agreement is between...')).toBeInTheDocument()
  expect(screen.getByText('doc_type: NDA')).toBeInTheDocument()
})

it('calls onClose when close button clicked', () => {
  const onClose = vi.fn()
  render(<DocModal doc={DOC} onClose={onClose} />)
  fireEvent.click(screen.getByLabelText('Close document viewer'))
  expect(onClose).toHaveBeenCalledOnce()
})

it('calls onClose when clicking the backdrop', () => {
  const onClose = vi.fn()
  const { container } = render(<DocModal doc={DOC} onClose={onClose} />)
  fireEvent.click(container.firstChild) // the .doc-modal backdrop
  expect(onClose).toHaveBeenCalledOnce()
})

it('calls onClose on Escape key', () => {
  const onClose = vi.fn()
  render(<DocModal doc={DOC} onClose={onClose} />)
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(onClose).toHaveBeenCalledOnce()
})
