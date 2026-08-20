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
// "Not now" on the onboarding card. Per-account for the same reason as the two
// above, and persisted so a deferral survives a reload — the card is a nudge, and
// a nudge that comes back on every refresh is a modal with extra steps. Settings
// keeps the standing "complete your profile" prompt regardless.
const onbKey = (account) => `cogbase.onboardingDismissed.${account}`

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

  // The account's company profile (GET /profile), account-scoped like the account
  // itself — one document shared by every namespace and app. Three states, and the
  // UI branches on all three:
  //   null + !profileLoaded → not fetched yet (render nothing)
  //   null + profileLoaded  → the fetch failed or the deployment can't hold profiles
  //                           (503, no document store) — "unknown", never onboard
  //   { exists: false }     → known cold start, the one case that offers onboarding
  const [profile, setProfile] = useState(null)
  const [profileLoaded, setProfileLoaded] = useState(false)
  const [onboardingDismissed, setOnboardingDismissedState] = useState(
    () => persisted(onbKey(initialAccount), '') === '1'
  )

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
    // The profile belongs to the account that was signed in, so drop it back to
    // "not fetched" rather than letting the previous account's answer decide
    // whether this one gets onboarded. refreshProfile re-runs on its own (its
    // identity tracks authFetch, which tracks the account).
    setProfile(null)
    setProfileLoaded(false)
    setOnboardingDismissedState(persisted(onbKey(accountId), '') === '1')
  }, [accountId])

  // Every request carries the tenant identity via the X-Account-Id header
  // (trust-on-declaration in dev/single_tenant mode — this UI never runs in
  // saas mode, so there is no bearer-token path here). Options are left
  // otherwise untouched so streaming and multipart uploads pass straight through.
  const authFetch = useCallback(async (url, opts = {}) => {
    const headers = {
      ...(accountId ? { 'X-Account-Id': accountId } : {}),
      ...(opts.headers || {}),
    }
    return fetch(url, { ...opts, headers })
  }, [accountId])

  // Bootstrap the calling identity from the server: GET /whoami returns the
  // account the server resolved (which we adopt) and the deployment mode. The UI
  // never sources an account itself — the server echoes back the header we sent.
  // Driven by the mounted header (App.jsx), not a provider mount effect, so
  // tab-level renders stay side-effect-free.
  const bootstrap = useCallback(async () => {
    try {
      const resp = await authFetch(`${apiUrl}/whoami`)
      if (!resp.ok) return
      const data = await resp.json()
      if (data.mode) { setMode(data.mode); persist('cogbase.mode', data.mode) }
      if (data.account_id && data.account_id !== accountId) {
        setAccountId(data.account_id)
      }
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

  // ── Company profile (account-scoped; api/routers/profile.py) ──
  // Driven by the mounted Layout, like refreshNamespaces, so tab-level renders
  // stay side-effect-free. A non-200 (503 on a deployment with no document store,
  // or any transport failure) resolves to "unknown", which reads as *not* a cold
  // start — better to skip an onboarding prompt than to offer one that 503s the
  // moment it is opened.
  const refreshProfile = useCallback(async () => {
    try {
      const resp = await authFetch(`${apiUrl}/profile`)
      setProfile(resp.ok ? await resp.json() : null)
    } catch {
      setProfile(null)
    } finally {
      setProfileLoaded(true)
    }
  }, [apiUrl, authFetch])

  // Replace the profile. Returns { ok } or { ok: false, error } so the caller can
  // render the failure inline (413 over the size cap is the expected one).
  const saveProfile = useCallback(async (markdown) => {
    try {
      const resp = await authFetch(`${apiUrl}/profile`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ markdown }),
      })
      const data = await resp.json().catch(() => ({}))
      if (!resp.ok) {
        const detail = Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : data.detail
        return { ok: false, error: detail || `HTTP ${resp.status}` }
      }
      setProfile(data)
      return { ok: true }
    } catch (e) {
      return { ok: false, error: e.message }
    }
  }, [apiUrl, authFetch])

  const deleteProfile = useCallback(async () => {
    try {
      const resp = await authFetch(`${apiUrl}/profile`, { method: 'DELETE' })
      if (!resp.ok && resp.status !== 204) return { ok: false, error: `HTTP ${resp.status}` }
      setProfile({ markdown: null, exists: false, updated_at: null, updated_by: null, source: null })
      return { ok: true }
    } catch (e) {
      return { ok: false, error: e.message }
    }
  }, [apiUrl, authFetch])

  // The interview writes the profile server-side (its save_company_profile tool),
  // so the client adopts the markdown it reports rather than re-reading it — the
  // turn already carries the authoritative text.
  const adoptSavedProfile = useCallback((markdown) => {
    setProfile(prev => ({
      ...(prev || {}),
      markdown,
      exists: true,
      source: 'interview',
      updated_at: new Date().toISOString(),
    }))
    setProfileLoaded(true)
  }, [])

  const dismissOnboarding = useCallback(() => {
    persist(onbKey(accountIdRef.current), '1')
    setOnboardingDismissedState(true)
  }, [])

  const value = useMemo(() => ({
    apiUrl, setApiUrl,
    accountId, setAccountId, mode, bootstrap, namespaceName, setNamespaceName,
    namespaces, namespacesLoaded, refreshNamespaces, ensureNamespace,
    apps, appsNs, refreshApps,
    nsBase, appBase, authFetch,
    currentApp, setCurrentApp,
    demoCatalog, setDemoCatalog,
    llmConfigured, setLlmConfigured, embConfigured, setEmbConfigured,
    // company profile
    profile, profileLoaded, refreshProfile, saveProfile, deleteProfile, adoptSavedProfile,
    onboardingDismissed, dismissOnboarding,
  }), [apiUrl, accountId, mode, bootstrap, namespaceName, namespaces, namespacesLoaded, refreshNamespaces, ensureNamespace, apps, appsNs, refreshApps, nsBase, appBase, authFetch, currentApp, setCurrentApp, demoCatalog, llmConfigured, embConfigured, setAccountId, setNamespaceName, profile, profileLoaded, refreshProfile, saveProfile, deleteProfile, adoptSavedProfile, onboardingDismissed, dismissOnboarding])

  return <AppCtx.Provider value={value}>{children}</AppCtx.Provider>
}

export function useApp() { return useContext(AppCtx) }
