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
  const [namespaceId, setNamespaceIdState] = useState(() => persisted('cogbase.namespaceId', DEFAULT_NAMESPACE))
  const [currentApp, setCurrentApp] = useState('')
  const [demoCatalog, setDemoCatalog] = useState([])
  const [llmConfigured, setLlmConfigured] = useState(false)
  const [embConfigured, setEmbConfigured] = useState(false)

  const setAccountId = useCallback((v) => {
    const next = (v || DEFAULT_ACCOUNT_ID).trim() || DEFAULT_ACCOUNT_ID
    persist('cogbase.accountId', next)
    setAccountIdState(next)
  }, [])
  const setNamespaceId = useCallback((v) => {
    const next = (v || DEFAULT_NAMESPACE).trim() || DEFAULT_NAMESPACE
    persist('cogbase.namespaceId', next)
    setNamespaceIdState(next)
  }, [])

  // Namespace-scoped bases. Name-addressed application routes moved under
  // /namespaces/{namespace}/applications (api/routers/applications.py); nsBase
  // also fronts the namespace-scoped generate/deploy route. Account-wide routes
  // (GET /applications, /skills, /generate/chat, /system) keep the bare apiUrl.
  const nsBase = `${apiUrl}/namespaces/${encodeURIComponent(namespaceId)}`
  const appBase = `${nsBase}/applications`

  // Every request carries the account as the X-Account-Id header (the security
  // boundary). authFetch injects it while leaving each call site's URL/options
  // otherwise untouched, so streaming and multipart uploads pass straight through.
  const authFetch = useCallback((url, opts = {}) => {
    return fetch(url, { ...opts, headers: { 'X-Account-Id': accountId, ...(opts.headers || {}) } })
  }, [accountId])

  const value = useMemo(() => ({
    apiUrl, setApiUrl,
    accountId, setAccountId, namespaceId, setNamespaceId,
    nsBase, appBase, authFetch,
    currentApp, setCurrentApp,
    demoCatalog, setDemoCatalog,
    llmConfigured, setLlmConfigured, embConfigured, setEmbConfigured,
  }), [apiUrl, accountId, namespaceId, nsBase, appBase, authFetch, currentApp, demoCatalog, llmConfigured, embConfigured, setAccountId, setNamespaceId])

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>
}

export function useApp() { return useContext(AppCtx) }
