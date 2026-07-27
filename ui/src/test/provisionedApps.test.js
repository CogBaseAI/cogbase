import { describe, it, expect, vi } from 'vitest'
import {
  PROVISIONED_APP_NAME,
  PROVISIONED_PRESETS,
  syncProvisionedAppLanguage,
} from '../provisionedApps'

// A provisioned contract-analyst app carrying the given presentation fields, in
// the same shape the app-list payload delivers (config parsed from config_yaml).
function provisionedApp(overrides = {}) {
  return {
    name: PROVISIONED_APP_NAME,
    namespace: 'legal-team',
    config: { ...PROVISIONED_PRESETS.en, ...overrides },
  }
}

function harness(apps) {
  const authFetch = vi.fn(async () => ({ ok: true }))
  const refreshApps = vi.fn(async () => {})
  return { apps, apiUrl: 'http://api', authFetch, refreshApps }
}

describe('syncProvisionedAppLanguage', () => {
  it('patches the default English app to the target language', async () => {
    const h = harness([provisionedApp()])
    await syncProvisionedAppLanguage({ ...h, lang: 'zh' })

    expect(h.authFetch).toHaveBeenCalledTimes(1)
    const [url, opts] = h.authFetch.mock.calls[0]
    expect(url).toBe('http://api/namespaces/legal-team/applications/contract-analyst/config')
    expect(opts.method).toBe('PATCH')
    expect(JSON.parse(opts.body)).toEqual({
      query_intro: PROVISIONED_PRESETS.zh.query_intro,
      example_queries: PROVISIONED_PRESETS.zh.example_queries,
    })
    expect(h.refreshApps).toHaveBeenCalledTimes(1)
  })

  it('is a no-op when already in the target language', async () => {
    const h = harness([provisionedApp(PROVISIONED_PRESETS.zh)])
    await syncProvisionedAppLanguage({ ...h, lang: 'zh' })
    expect(h.authFetch).not.toHaveBeenCalled()
  })

  it('leaves user-customized fields untouched', async () => {
    const h = harness([provisionedApp({ query_intro: 'my own intro' })])
    await syncProvisionedAppLanguage({ ...h, lang: 'zh' })
    expect(h.authFetch).not.toHaveBeenCalled()
  })

  it('does nothing when the provisioned app is absent', async () => {
    const h = harness([{ name: 'other-app', namespace: 'x', config: {} }])
    await syncProvisionedAppLanguage({ ...h, lang: 'zh' })
    expect(h.authFetch).not.toHaveBeenCalled()
  })

  it('does nothing for a language without a preset', async () => {
    const h = harness([provisionedApp()])
    await syncProvisionedAppLanguage({ ...h, lang: 'fr' })
    expect(h.authFetch).not.toHaveBeenCalled()
  })

  it('does not refresh when the patch fails', async () => {
    const h = harness([provisionedApp()])
    h.authFetch.mockResolvedValueOnce({ ok: false })
    await syncProvisionedAppLanguage({ ...h, lang: 'zh' })
    expect(h.refreshApps).not.toHaveBeenCalled()
  })
})
