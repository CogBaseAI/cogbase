// Scope-tier navigation model + hash routing (docs/ui-navigation.md, milestone B).
//
// TAB_TIER is the single source of the tab→tier grouping; DEFAULT_TAB is where
// focusing a tier lands. buildHash/parseHash project the (focus, namespace, app,
// tab) tuple onto location.hash and back, giving deep-linking and refresh-survival
// (step 5). They are pure so the Layout effects around them stay thin.

export const TAB_TIER = {
  build: 'namespace', apps: 'namespace', demos: 'namespace',
  ingest: 'application', data: 'application', query: 'application', memory: 'application',
  namespaces: 'account', skills: 'account', settings: 'account',
}
export const DEFAULT_TAB = { account: 'namespaces', namespace: 'apps', application: 'query' }

// Suggestions for a free-text namespace picker (the sidebar switcher and the Build
// tab's deploy target): the account's real namespaces, plus whatever is currently
// selected — so the active value is always offered, including a not-yet-created
// deploy target being typed (the deploy flow creates it). De-duplicated. There is
// no implicit 'default' any more; the server has no default namespace, so offering
// one would just point at a namespace that doesn't exist until something is
// deployed into it.
export function nsOptions(names, current) {
  return [...new Set([current, ...names])].filter(Boolean)
}

// State → hash. Each tier maps onto a distinct shape so the hash is self-describing:
//   #/account/settings
//   #/ns/legal/apps
//   #/ns/legal/app/contracts/query   (application tier, app selected)
//   #/ns/legal/app                    (application tier, nothing selected)
// The account tier isn't namespace-scoped, so its hash omits the namespace (which
// persists in context/localStorage independently).
export function buildHash({ focus, namespaceName, currentApp, activeTab }) {
  if (focus === 'account') return `#/account/${activeTab}`
  const ns = encodeURIComponent(namespaceName)
  if (focus === 'application') {
    return currentApp ? `#/ns/${ns}/app/${encodeURIComponent(currentApp)}/${activeTab}` : `#/ns/${ns}/app`
  }
  return `#/ns/${ns}/${activeTab}`
}

// Hash → partial state, or null when unparseable (caller leaves state untouched).
// Keys appear only when the hash actually carries them, so a mirror-back of the
// hash never clobbers state it doesn't address: an account hash omits namespace and
// app; a namespace-tier hash omits app (preserving a live selection); only
// `.../app` (empty) and `.../app/<name>` set currentApp explicitly.
export function parseHash(hash) {
  const parts = (hash || '').replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  if (!parts.length) return null
  const belongs = (tab, tier) => TAB_TIER[tab] === tier
  if (parts[0] === 'account') {
    return belongs(parts[1], 'account') ? { focus: 'account', activeTab: parts[1] } : null
  }
  if (parts[0] === 'ns' && parts[1]) {
    const namespaceName = parts[1]
    if (parts[2] === 'app') {
      const currentApp = parts[3] || ''
      if (!currentApp) return { focus: 'application', namespaceName, currentApp: '', activeTab: DEFAULT_TAB.application }
      return belongs(parts[4], 'application') ? { focus: 'application', namespaceName, currentApp, activeTab: parts[4] } : null
    }
    return belongs(parts[2], 'namespace') ? { focus: 'namespace', namespaceName, activeTab: parts[2] } : null
  }
  return null
}
