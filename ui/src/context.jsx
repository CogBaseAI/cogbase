import React, { createContext, useContext, useState, useCallback, useMemo } from 'react'

const AppCtx = createContext(null)

// Tenancy defaults mirror the API (api/dependencies.py): a request that omits the
// X-Account-Id header lands in account "default", and an unqualified path lands in
// namespace "default". Both are trust-on-declaration dev knobs for now.
const DEFAULT_ACCOUNT_ID = 'default'
const DEFAULT_NAMESPACE = 'default'

// Persist the tenant selection across reloads so a dev working in a non-default
// account/namespace doesn't have to re-enter it every session.
function persisted(key, fallback) {
  if (typeof window === 'undefined') return fallback
  try {
    return window.localStorage.getItem(key) || fallback
  } catch {
    return fallback
  }
}

function persist(key, value) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(key, value)
  } catch {
    /* private mode / disabled storage — non-fatal */
  }
}

export function AppProvider({ children }) {
  // Default to the origin the UI was served from so it works whether run
  // locally or on a remote node. When running the vite dev server (port 5173),
  // the API lives separately on localhost:8000.
  const defaultApiUrl =
    typeof window !== 'undefined' && window.location.port !== '5173'
      ? window.location.origin
      : 'http://localhost:8000'
  const [apiUrl, setApiUrl] = useState(defaultApiUrl)
  const [accountId, setAccountIdState] = useState(() => persisted('cogbase.accountId', DEFAULT_ACCOUNT_ID))
  // The API addresses namespaces by their user-facing *name* (the {namespace} URL
  // path segment); the internal namespace_id is a server-side concept the client
  // never sends (api/dependencies.py resolve_namespace_id maps name -> id). So this
  // holds a name, not an id.
  const [namespaceName, setNamespaceNameState] = useState(() => persisted('cogbase.namespaceName', DEFAULT_NAMESPACE))
  // The selected app is a (namespace, name) pair — a name is only unique within a
  // namespace, so operating on it must carry its own namespace, independent of the
  // header-selected working namespace. currentApp stays the bare name for display
  // and compatibility; currentAppNs pins the namespace it was selected from.
  const [currentApp, setCurrentAppState] = useState('')
  const [currentAppNs, setCurrentAppNs] = useState(namespaceName)
  const [namespaces, setNamespaces] = useState([])
  const [apps, setApps] = useState([])   // apps in the selected namespace, for the App switcher
  const [demoCatalog, setDemoCatalog] = useState([])
  const [llmConfigured, setLlmConfigured] = useState(false)
  const [embConfigured, setEmbConfigured] = useState(false)

  const setAccountId = useCallback((v) => {
    const next = (v || DEFAULT_ACCOUNT_ID).trim() || DEFAULT_ACCOUNT_ID
    persist('cogbase.accountId', next)
    setAccountIdState(next)
  }, [])
  const setNamespaceName = useCallback((v) => {
    const next = (v || DEFAULT_NAMESPACE).trim() || DEFAULT_NAMESPACE
    persist('cogbase.namespaceName', next)
    setNamespaceNameState(next)
  }, [])

  // Namespace-scoped bases. Name-addressed application routes moved under
  // /namespaces/{namespace}/applications (api/routers/applications.py); nsBase
  // also fronts the namespace-scoped generate/deploy route. Account-wide routes
  // (GET /applications, /skills, /generate/chat, /system) keep the bare apiUrl.
  // appBase is scoped to the header-selected working namespace (used for creating
  // /deploying); currentAppBase is scoped to the selected app's own namespace
  // (used for operating on it — query, ingest, workflows, ...).
  const nsBase = `${apiUrl}/namespaces/${encodeURIComponent(namespaceName)}`
  const appBase = `${nsBase}/applications`
  const currentAppBase = `${apiUrl}/namespaces/${encodeURIComponent(currentAppNs)}/applications`

  // Select an app: pin the namespace it was chosen from so later operations
  // address it correctly regardless of the header's working namespace. Callers
  // that omit the namespace (e.g. a fresh deploy) inherit the working namespace.
  const setCurrentApp = useCallback((name, namespace) => {
    setCurrentAppState(name || '')
    if (name) setCurrentAppNs(namespace || namespaceName)
  }, [namespaceName])

  // Every request carries the account as the X-Account-Id header (the security
  // boundary). authFetch injects it while leaving each call site's URL/options
  // otherwise untouched, so streaming and multipart uploads pass straight through.
  const authFetch = useCallback((url, opts = {}) => {
    return fetch(url, { ...opts, headers: { 'X-Account-Id': accountId, ...(opts.headers || {}) } })
  }, [accountId])

  // The account's namespaces, for the header switcher. The header drives the fetch
  // (on mount and whenever the account changes) so tab-level renders that don't
  // mount the header stay side-effect-free. A new account may have no namespaces
  // until an app is created, so callers merge in 'default' + the current selection.
  const refreshNamespaces = useCallback(async () => {
    try {
      const resp = await authFetch(`${apiUrl}/namespaces`)
      if (resp.ok) {
        const { namespaces: items = [] } = await resp.json()
        setNamespaces(items)
      } else {
        setNamespaces([])
      }
    } catch {
      setNamespaces([])
    }
  }, [apiUrl, authFetch])

  // Apps in the selected namespace, for the App switcher. Namespace-scoped (the
  // breadcrumb's account ▸ namespace ▸ app path), unlike the Apps tab's account-wide
  // listing. Re-fetched whenever the namespace (nsBase) or account changes.
  const refreshApps = useCallback(async () => {
    try {
      const resp = await authFetch(`${nsBase}/applications`)
      if (resp.ok) {
        const { applications: items = [] } = await resp.json()
        setApps(items)
      } else {
        setApps([])
      }
    } catch {
      setApps([])
    }
  }, [nsBase, authFetch])

  const value = useMemo(() => ({
    apiUrl, setApiUrl,
    accountId, setAccountId, namespaceName, setNamespaceName,
    namespaces, refreshNamespaces,
    apps, refreshApps,
    nsBase, appBase, currentAppBase, authFetch,
    currentApp, currentAppNs, setCurrentApp,
    demoCatalog, setDemoCatalog,
    llmConfigured, setLlmConfigured, embConfigured, setEmbConfigured,
  }), [apiUrl, accountId, namespaceName, namespaces, refreshNamespaces, apps, refreshApps, nsBase, appBase, currentAppBase, authFetch, currentApp, currentAppNs, setCurrentApp, demoCatalog, llmConfigured, embConfigured, setAccountId, setNamespaceName])

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>
}

export function useApp() { return useContext(AppCtx) }
