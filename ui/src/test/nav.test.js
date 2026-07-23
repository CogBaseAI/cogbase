import { describe, it, expect } from 'vitest'
import { buildHash, parseHash } from '../nav'

describe('buildHash', () => {
  it('encodes each tier as its own shape', () => {
    expect(buildHash({ focus: 'account', activeTab: 'settings' })).toBe('#/account/settings')
    expect(buildHash({ focus: 'namespace', namespaceName: 'legal', activeTab: 'apps' })).toBe('#/ns/legal/apps')
    expect(buildHash({ focus: 'application', namespaceName: 'legal', currentApp: 'contracts', activeTab: 'query' }))
      .toBe('#/ns/legal/app/contracts/query')
  })

  it('drops the app segment when the application tier has no selection', () => {
    expect(buildHash({ focus: 'application', namespaceName: 'legal', currentApp: '', activeTab: 'query' }))
      .toBe('#/ns/legal/app')
  })

  it('percent-encodes namespace and app names', () => {
    expect(buildHash({ focus: 'application', namespaceName: 'a b', currentApp: 'c/d', activeTab: 'data' }))
      .toBe('#/ns/a%20b/app/c%2Fd/data')
  })
})

describe('parseHash', () => {
  it('round-trips the three tier shapes', () => {
    expect(parseHash('#/account/skills')).toEqual({ focus: 'account', activeTab: 'skills' })
    expect(parseHash('#/ns/legal/build')).toEqual({ focus: 'namespace', namespaceName: 'legal', activeTab: 'build' })
    expect(parseHash('#/ns/legal/app/contracts/memory'))
      .toEqual({ focus: 'application', namespaceName: 'legal', currentApp: 'contracts', activeTab: 'memory' })
  })

  it('parses the empty application tier to a cleared selection on its default tab', () => {
    expect(parseHash('#/ns/legal/app')).toEqual({ focus: 'application', namespaceName: 'legal', currentApp: '', activeTab: 'query' })
  })

  it('omits currentApp for account and namespace hashes so a live selection is preserved', () => {
    expect(parseHash('#/account/settings')).not.toHaveProperty('currentApp')
    expect(parseHash('#/ns/legal/apps')).not.toHaveProperty('currentApp')
  })

  it('decodes percent-encoded segments', () => {
    expect(parseHash('#/ns/a%20b/app/c%2Fd/data'))
      .toEqual({ focus: 'application', namespaceName: 'a b', currentApp: 'c/d', activeTab: 'data' })
  })

  it('rejects garbage and tab/tier mismatches', () => {
    expect(parseHash('')).toBeNull()
    expect(parseHash('#/')).toBeNull()
    expect(parseHash('#/nonsense')).toBeNull()
    expect(parseHash('#/account/query')).toBeNull()      // query is an application tab
    expect(parseHash('#/ns/legal/settings')).toBeNull()  // settings is an account tab
    expect(parseHash('#/ns/legal/app/contracts/apps')).toBeNull()  // apps is a namespace tab
  })
})
