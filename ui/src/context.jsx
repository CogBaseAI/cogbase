import React, { createContext, useContext, useState, useCallback, useMemo } from 'react'

const AppCtx = createContext(null)

// Tenancy defaults (api/dependencies.py): a request that omits the X-Account-Id
// header lands in account "default" (a trust-on-declaration dev knob). There is no
// server-side default *namespace* — one must be created before it can hold apps
// (the server rejects a deploy into an unknown namespace with 404). So the working
// namespace has no default: it starts empty (no namespace selected) and is set only
// to a namespace the account actually has, reconciled against the live list in
// App.jsx. An empty account shows a "create a namespace" prompt rather than a
// phantom selection.
const DEFAULT_ACCOUNT_ID = 'default'

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
  // holds a name, not an id. Empty string means "no working namespace" (a fresh
  // account with none yet); App.jsx reconciles it against the live list.
  const [namespaceName, setNamespaceNameState] = useState(() => persisted('cogbase.namespaceName', ''))
  // Whether the account's namespace list has been fetched at least once, so the UI
  // can tell "loaded, none exist" (show the create-a-namespace prompt) from "not
  // loaded yet" (show nothing) instead of flashing an empty state on first paint.
  const [namespacesLoaded, setNamespacesLoaded] = useState(false)
  // The selected app is a (namespace, name) pair — a name is only unique within a
  // namespace. Under the unified namespace model (docs/ui-navigation.md, milestone
  // B step 4) selecting an app snaps the working namespace to the app's own, so the
  // app always lives in `namespaceName`; there is no longer a separate "app
  // namespace" to track. currentApp is the bare name for display and addressing.
  const [currentApp, setCurrentAppState] = useState('')
  const [namespaces, setNamespaces] = useState([])
  const [apps, setApps] = useState([])   // apps in the selected namespace, for the App switcher
  const [appsNs, setAppsNs] = useState(null)  // which namespace `apps` was loaded for
  const [demoCatalog, setDemoCatalog] = useState([])
  const [llmConfigured, setLlmConfigured] = useState(false)
  const [embConfigured, setEmbConfigured] = useState(false)
  // Deployment mode from GET /whoami. 'dev' means the account is trust-on-
  // declaration (the header we send is echoed back), so the UI keeps an editable
  // account field. Any other mode (saas/single_tenant/demo) means the server
  // resolves the account authoritatively, so the UI treats it as read-only.
  const [mode, setMode] = useState('dev')

  const setAccountId = useCallback((v) => {
    const next = (v || DEFAULT_ACCOUNT_ID).trim() || DEFAULT_ACCOUNT_ID
    persist('cogbase.accountId', next)
    setAccountIdState(next)
  }, [])
  const setNamespaceName = useCallback((v) => {
    // Empty is a valid state (no working namespace on a fresh account); don't
    // coerce it to a phantom default.
    const next = (v || '').trim()
    persist('cogbase.namespaceName', next)
    setNamespaceNameState(next)
  }, [])

  // Namespace-scoped bases. Name-addressed application routes moved under
  // /namespaces/{namespace}/applications (api/routers/applications.py); nsBase
  // also fronts the namespace-scoped generate/deploy route. Account-wide routes
  // (GET /applications, /skills, /generate/chat, /system) keep the bare apiUrl.
  // appBase is scoped to the working namespace; under the unified model the
  // selected app always lives there, so it doubles as the base for operating on
  // the current app (query, ingest, workflows, ...).
  const nsBase = `${apiUrl}/namespaces/${encodeURIComponent(namespaceName)}`
  const appBase = `${nsBase}/applications`

  // Select an app: snap the working namespace to the app's own so the whole
  // account ▸ namespace ▸ app path stays coherent (the breadcrumb reads as one
  // path, and appBase addresses the selection). Callers that omit the namespace
  // (e.g. a fresh deploy into the current namespace) keep the working namespace.
  // Clearing the selection (empty name) leaves the namespace untouched.
  const setCurrentApp = useCallback((name, namespace) => {
    setCurrentAppState(name || '')
    if (name && namespace) setNamespaceName(namespace)
  }, [setNamespaceName])

  // Every request carries the account as the X-Account-Id header (the security
  // boundary). authFetch injects it while leaving each call site's URL/options
  // otherwise untouched, so streaming and multipart uploads pass straight through.
  const authFetch = useCallback((url, opts = {}) => {
    return fetch(url, { ...opts, headers: { 'X-Account-Id': accountId, ...(opts.headers || {}) } })
  }, [accountId])

  // Bootstrap the calling identity from the server: GET /whoami returns the
  // account the server resolved (which we adopt) and the deployment mode (which
  // decides whether the account is editable). The UI never sources an account
  // itself — in 'dev' this echoes the header we sent, and in managed modes it
  // becomes the authoritative account once auth binds it server-side. Like
  // refreshNamespaces, this is driven by the mounted header (App.jsx), not a
  // provider mount effect, so tab-level renders stay side-effect-free.
  const bootstrap = useCallback(async () => {
    try {
      const resp = await authFetch(`${apiUrl}/whoami`)
      if (!resp.ok) return
      const data = await resp.json()
      if (data.mode) setMode(data.mode)
      if (data.account_id && data.account_id !== accountId) setAccountId(data.account_id)
    } catch {
      /* no /whoami (old server) — keep dev defaults */
    }
  }, [apiUrl, authFetch, accountId, setAccountId])

  // The account's namespaces, for the header switcher. The header drives the fetch
  // (on mount and whenever the account changes) so tab-level renders that don't
  // mount the header stay side-effect-free. A fresh account may have none until one
  // is created; `namespacesLoaded` lets the UI distinguish that from "not fetched
  // yet" so it can prompt for creation rather than flash a phantom selection.
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
    } finally {
      setNamespacesLoaded(true)
    }
  }, [apiUrl, authFetch])

  // Ensure a namespace exists so an app can be deployed into it. The server no
  // longer auto-registers namespaces on deploy (api/routers/app_generate.py), so
  // the free-text namespace pickers create the target on demand. Idempotent: a 409
  // (already exists) is treated as success. Returns true once the namespace exists.
  const ensureNamespace = useCallback(async (name) => {
    const n = (name || '').trim()
    if (!n) return false
    try {
      const resp = await authFetch(`${apiUrl}/namespaces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: n }),
      })
      // 201 created, or 409 it already existed — either way it now exists.
      if (resp.ok || resp.status === 409) {
        await refreshNamespaces()
        return true
      }
      return false
    } catch {
      return false
    }
  }, [apiUrl, authFetch, refreshNamespaces])

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
    } finally {
      // Stamp the list with its namespace so consumers can tell "loaded, app
      // absent" from "not loaded yet" (see the currentApp reconciliation in App.jsx).
      setAppsNs(namespaceName)
    }
  }, [nsBase, authFetch, namespaceName])

  const value = useMemo(() => ({
    apiUrl, setApiUrl,
    accountId, setAccountId, mode, bootstrap, namespaceName, setNamespaceName,
    namespaces, namespacesLoaded, refreshNamespaces, ensureNamespace,
    apps, appsNs, refreshApps,
    nsBase, appBase, authFetch,
    currentApp, setCurrentApp,
    demoCatalog, setDemoCatalog,
    llmConfigured, setLlmConfigured, embConfigured, setEmbConfigured,
  }), [apiUrl, accountId, mode, bootstrap, namespaceName, namespaces, namespacesLoaded, refreshNamespaces, ensureNamespace, apps, appsNs, refreshApps, nsBase, appBase, authFetch, currentApp, setCurrentApp, demoCatalog, llmConfigured, embConfigured, setAccountId, setNamespaceName])

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>
}

export function useApp() { return useContext(AppCtx) }
