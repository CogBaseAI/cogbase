import React from 'react'
import { render } from '@testing-library/react'
import { AppProvider } from '../context'
import { I18nProvider } from '../i18n'

export function renderWithCtx(ui, options) {
  return render(<I18nProvider><AppProvider>{ui}</AppProvider></I18nProvider>, options)
}
