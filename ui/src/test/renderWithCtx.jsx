import React from 'react'
import { render } from '@testing-library/react'
import { AppProvider } from '../context'

export function renderWithCtx(ui, options) {
  return render(<AppProvider>{ui}</AppProvider>, options)
}
