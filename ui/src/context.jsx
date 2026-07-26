import React, { createContext, useContext, useState, useCallback, useMemo, useRef, useEffect } from 'react'

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

// The remembered "last used" namespace + app are keyed by account. Namespace and
// app names are only unique within an account, so on a browser shared by several
// accounts (saas, multiple logins) one account's selection must never seed
// another's — that would land a returning user in a same-named-but-wrong workspace.
// So we don't persist a single global selection; we persist one per account and
// restore it whenever the active account resolves (login / account switch).
const nsKey = (account) => `cogbase.ns.${account}`
const appKey = (account) => `cogbase.app.${account}`

export function AppProvider({ children }) {
  // Default to the origin the UI was served from so it works whether run
  // locally or on a remote node. When running the vite dev server (port 5173),
  // the API lives separately on localhost:8000.
  const defaultApiUrl =
    typeof window !== 'undefined' && window.location.port !== '5173'
      ? window.location.origin
      : 'http://localhost:8000'
  const [apiUrl, setApiUrl] = useState(defaultApiUrl)
  const initialAccount = persisted('cogbase.accountId', DEFAULT_ACCOUNT_ID)
  const [accountId, setAccountIdState] = useState(initialAccount)
  // The persist helpers below write under the *current* account's key. Hold the
  // account in a ref so those callbacks stay identity-stable (they don't need to
  // re-create when the account changes) while still targeting the right key.
  const accountIdRef = useRef(accountId)
  accountIdRef.current = accountId
  // The API addresses namespaces by their user-facing *name* (the {namespace} URL
  // path segment); the internal namespace_id is a server-side concept the client
  // never sends (api/dependencies.py resolve_namespace_id maps name -> id). So this
  // holds a name, not an id. Empty string means "no working namespace" (a fresh
  // account with none yet); App.jsx reconciles it against the live list.
  const [namespaceName, setNamespaceNameState] = useState(() => persisted(nsKey(initialAccount), ''))
  // Whether the account's namespace list has been fetched at least once, so the UI
  // can tell "loaded, none exist" (show the create-a-namespace prompt) from "not
  // loaded yet" (show nothing) instead of flashing an empty state on first paint.
  const [namespacesLoaded, setNamespacesLoaded] = useState(false)
  // The selected app is a (namespace, name) pair — a name is only unique within a
  // namespace. Under the unified namespace model (docs/ui-navigation.md, milestone
  // B step 4) selecting an app snaps the working namespace to the app's own, so the
  // app always lives in `namespaceName`; there is no longer a separate "app
  // namespace" to track. currentApp is the bare name for display and addressing.
  const [currentApp, setCurrentAppState] = useState(() => persisted(appKey(initialAccount), ''))
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
  const [mode, setMode] = useState(() => persisted('cogbase.mode', 'dev'))

  // First-party auth (saas mode). Tokens persist across reloads so a refresh
  // doesn't bounce the user to the login screen. Persisting in localStorage is a
  // deliberate pilot trade-off (an XSS bug could read them); a hardening step
  // moves the refresh token to an httpOnly cookie. In dev mode these stay empty
  // and the header path is used, so nothing here affects local development.
  const [accessToken, setAccessTokenState] = useState(() => persisted('cogbase.accessToken', ''))
  const [refreshToken, setRefreshTokenState] = useState(() => persisted('cogbase.refreshToken', ''))
  const [email, setEmailState] = useState(() => persisted('cogbase.email', ''))
  const [role, setRoleState] = useState(() => persisted('cogbase.role', ''))
  // Whether /whoami has resolved at least once, so the gate can tell "still
  // checking" from "checked, not signed in" and avoid flashing the login screen.
  const [authChecked, setAuthChecked] = useState(false)
  // One-shot flag raised by a brand-new-account signup. A fresh account is seeded
  // server-side with a starter workspace (a legal-team namespace + contract-analyst
  // app; see api/provisioning.py). The namespace auto-selects via App.jsx's reconcile,
  // and this tells App.jsx to also adopt the provisioned app once its namespace's apps
  // load, so the owner lands in the app instead of the pick-an-app empty state.
  const [autoSelectApp, setAutoSelectApp] = useState(false)

  const setAccountId = useCallback((v) => {
    const next = (v || DEFAULT_ACCOUNT_ID).trim() || DEFAULT_ACCOUNT_ID
    persist('cogbase.accountId', next)
    setAccountIdState(next)
  }, [])
  const setNamespaceName = useCallback((v) => {
    // Empty is a valid state (no working namespace on a fresh account); don't
    // coerce it to a phantom default.
    const next = (v || '').trim()
    persist(nsKey(accountIdRef.current), next)
    setNamespaceNameState(next)
  }, [])

  // Persist + set the whole authenticated session in one place (login/signup), and
  // its inverse for sign-out. Keeping tokens, account, and profile in lockstep
  // avoids a half-cleared state where a stale token races a new account.
  const applySession = useCallback((data) => {
    persist('cogbase.accessToken', data.access_token || '')
    persist('cogbase.refreshToken', data.refresh_token || '')
    persist('cogbase.email', data.email || '')
    persist('cogbase.role', data.role || '')
    persist('cogbase.accountId', data.account_id || DEFAULT_ACCOUNT_ID)
    setAccessTokenState(data.access_token || '')
    setRefreshTokenState(data.refresh_token || '')
    setEmailState(data.email || '')
    setRoleState(data.role || '')
    setAccountIdState(data.account_id || DEFAULT_ACCOUNT_ID)
  }, [])

  const clearSession = useCallback(() => {
    persist('cogbase.accessToken', '')
    persist('cogbase.refreshToken', '')
    persist('cogbase.email', '')
    persist('cogbase.role', '')
    setAccessTokenState('')
    setRefreshTokenState('')
    setEmailState('')
    setRoleState('')
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
    const next = name || ''
    persist(appKey(accountIdRef.current), next)
    setCurrentAppState(next)
    if (next && namespace) setNamespaceName(namespace)
  }, [setNamespaceName])

  // Restore the account's last-used namespace + app whenever the active account
  // resolves to a *different* one — the login/signup handoff, an account switch, or
  // /whoami adopting the server-resolved account. This is what lands a returning
  // user back where they left off. Uses the raw state setters (not the persisting
  // wrappers) so restoring doesn't rewrite what it just read. The initial mount is
  // skipped: state is already seeded from `initialAccount` above, and App.jsx's hash
  // router may deep-link a different view on first paint. App.jsx then reconciles the
  // restored values against the account's live namespace/app lists, dropping any that
  // no longer exist.
  const acctMountedRef = useRef(false)
  useEffect(() => {
    if (!acctMountedRef.current) { acctMountedRef.current = true; return }
    setNamespaceNameState(persisted(nsKey(accountId), ''))
    setCurrentAppState(persisted(appKey(accountId), ''))
  }, [accountId])

  // Exchange the refresh token for a fresh access token. Returns the new token, or
  // '' when refresh fails (expired/revoked) — in which case the session is cleared
  // so the app falls back to the login gate. Uses raw fetch to avoid recursing
  // through authFetch's own 401 handling.
  const refreshAccess = useCallback(async () => {
    if (!refreshToken) return ''
    try {
      const resp = await fetch(`${apiUrl}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
      if (!resp.ok) { clearSession(); return '' }
      const data = await resp.json()
      persist('cogbase.accessToken', data.access_token || '')
      setAccessTokenState(data.access_token || '')
      return data.access_token || ''
    } catch {
      return ''
    }
  }, [apiUrl, refreshToken, clearSession])

  // Every request carries the tenant identity. In dev mode that's the X-Account-Id
  // header (trust-on-declaration); in saas mode it's the Bearer access token (the
  // server derives the account from it and ignores the header). Options are left
  // otherwise untouched so streaming and multipart uploads pass straight through.
  // On a 401 with a refresh token available, transparently refresh once and retry —
  // but only for requests we can safely replay (no consumable body), so an upload's
  // FormData isn't re-sent half-read.
  const authFetch = useCallback(async (url, opts = {}) => {
    const headersFor = (tok) => ({
      ...(accountId ? { 'X-Account-Id': accountId } : {}),
      ...(tok ? { Authorization: `Bearer ${tok}` } : {}),
      ...(opts.headers || {}),
    })
    let resp = await fetch(url, { ...opts, headers: headersFor(accessToken) })
    const replayable = !opts.body || typeof opts.body === 'string'
    if (resp.status === 401 && refreshToken && replayable) {
      const fresh = await refreshAccess()
      if (fresh) resp = await fetch(url, { ...opts, headers: headersFor(fresh) })
    }
    return resp
  }, [accountId, accessToken, refreshToken, refreshAccess])

  // Auth actions the login screen and header call. login/signup adopt the returned
  // session; logout best-effort revokes the refresh token server-side then clears
  // local state so the gate reappears.
  const login = useCallback(async (emailArg, password) => {
    const resp = await fetch(`${apiUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: emailArg, password }),
    })
    if (!resp.ok) {
      const msg = await resp.json().catch(() => ({}))
      throw new Error(msg.detail || 'Login failed')
    }
    applySession(await resp.json())
  }, [apiUrl, applySession])

  const signup = useCallback(async (emailArg, password, inviteToken) => {
    const resp = await fetch(`${apiUrl}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: emailArg, password, invite_token: inviteToken || undefined }),
    })
    if (!resp.ok) {
      const msg = await resp.json().catch(() => ({}))
      // FastAPI validation errors arrive as an array under detail.
      const detail = Array.isArray(msg.detail) ? msg.detail.map(d => d.msg).join('; ') : msg.detail
      throw new Error(detail || 'Signup failed')
    }
    applySession(await resp.json())
    // Only a no-invite signup mints a fresh account (an invitee joins an existing
    // one and gets no provisioning), so only that path seeds the starter workspace.
    // Raise the one-shot so App.jsx auto-selects the provisioned contract-analyst app.
    if (!inviteToken) setAutoSelectApp(true)
  }, [apiUrl, applySession])

  const logout = useCallback(async () => {
    if (refreshToken) {
      try {
        await fetch(`${apiUrl}/auth/logout`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        })
      } catch { /* best effort — clear locally regardless */ }
    }
    clearSession()
  }, [apiUrl, refreshToken, clearSession])

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
      if (data.mode) { setMode(data.mode); persist('cogbase.mode', data.mode) }
      if (data.mode === 'saas') {
        // Server is authoritative: adopt the resolved identity, or — when the
        // token is absent/invalid (no account_id) — clear the local session so
        // the login gate takes over.
        if (data.account_id) {
          setAccountId(data.account_id)
          persist('cogbase.email', data.email || '')
          persist('cogbase.role', data.role || '')
          setEmailState(data.email || '')
          setRoleState(data.role || '')
        } else {
          clearSession()
        }
      } else if (data.account_id && data.account_id !== accountId) {
        setAccountId(data.account_id)
      }
    } catch {
      /* no /whoami (old server) — keep dev defaults */
    } finally {
      setAuthChecked(true)
    }
  }, [apiUrl, authFetch, accountId, setAccountId, clearSession])

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
    // auth (saas mode)
    accessToken, email, role, authChecked, login, signup, logout,
    autoSelectApp, setAutoSelectApp,
  }), [apiUrl, accountId, mode, bootstrap, namespaceName, namespaces, namespacesLoaded, refreshNamespaces, ensureNamespace, apps, appsNs, refreshApps, nsBase, appBase, authFetch, currentApp, setCurrentApp, demoCatalog, llmConfigured, embConfigured, setAccountId, setNamespaceName, accessToken, email, role, authChecked, login, signup, logout, autoSelectApp])

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>
}

export function useApp() { return useContext(AppCtx) }
