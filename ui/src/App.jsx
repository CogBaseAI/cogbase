import React, { useState, useEffect, useRef } from 'react'
import { AppProvider, useApp } from './context'
import { TAB_TIER, DEFAULT_TAB, buildHash, parseHash, nsOptions } from './nav'
import { I18nProvider, useT, LANGUAGES } from './i18n'
import { syncProvisionedAppLanguage } from './provisionedApps'
import BuildTab from './components/tabs/BuildTab'
import AppsTab from './components/tabs/AppsTab'
import NamespacesTab from './components/tabs/NamespacesTab'
import NamespaceSelect from './components/NamespaceSelect'
import DemosTab from './components/tabs/DemosTab'
import IngestTab from './components/tabs/IngestTab'
import DataTab from './components/tabs/DataTab'
import QueryTab from './components/tabs/QueryTab'
import MemoryTab from './components/tabs/MemoryTab'
import SkillsTab from './components/tabs/SkillsTab'
import SettingsTab from './components/tabs/SettingsTab'
import LoginScreen from './components/LoginScreen'
import DocModal from './components/modals/DocModal'
import ConfigModal from './components/modals/ConfigModal'
import WfModal from './components/modals/WfModal'
import TaskProgressModal from './components/modals/TaskProgressModal'

// TAB_TIER / DEFAULT_TAB (the tab→tier grouping and each tier's landing tab) live
// in ./nav alongside the hash router that also consumes them.

function Layout() {
  const { accountId, mode, email, logout, namespaceName, setNamespaceName, namespaces, namespacesLoaded, refreshNamespaces, apps, appsNs, refreshApps, currentApp, setCurrentApp, autoSelectApp, setAutoSelectApp, apiUrl, authFetch } = useApp()
  const { t, lang, setLang } = useT()

  // Keep the auto-provisioned contract-analyst's Query starter panel (query_intro
  // / example_queries) in the active UI language. App config stays single-language;
  // this re-localizes just those UI-only fields via the light config PATCH, and is
  // a no-op for any other or user-customized app. Runs on language change and when
  // the app list (re)loads.
  useEffect(() => {
    syncProvisionedAppLanguage({ apps, lang, apiUrl, authFetch, refreshApps })
  }, [lang, apps, apiUrl, authFetch, refreshApps])
  const [activeTab, setActiveTab] = useState('build')
  const [focus, setFocus] = useState(TAB_TIER['build'])   // which tier's sub-nav shows
  // Sidebar collapse: hiding drops it to a thin rail with a show toggle. Replaces
  // the old per-tab chats collapse (the chats/collections lists now live in the
  // sidebar's lower slot, so one control hides the whole thing — the ChatGPT/Claude
  // pattern).
  const [navHidden, setNavHidden] = useState(false)
  // The DOM node of the sidebar's lower contextual slot. The application-tier tabs
  // portal their secondary list into it (Query → chats, Data → collections). Held
  // in state (via a callback ref) so the tabs re-render once it mounts.
  const [navSlot, setNavSlot] = useState(null)
  const [docModal, setDocModal] = useState(null)        // null | doc object
  const [configModal, setConfigModal] = useState(null)  // null | { demo }
  const [wfModal, setWfModal] = useState(null)          // null | { appName, workflowName, paramKey, label, values, desc, allDone, fromIngest }
  const [taskProgress, setTaskProgress] = useState(null) // null | { appName, workflowName, docId }
  const [ingestRefreshKey, setIngestRefreshKey] = useState(0)
  const [dataRefreshKey, setDataRefreshKey] = useState(0)
  const [wfCompleteCollection, setWfCompleteCollection] = useState(null)
  // A query the Ingest tab's post-upload CTA hands off to the Query tab: jump to
  // Query and either auto-send it or just prefill the input. Shape: null (plain
  // navigate) | { text, send }. QueryTab clears it once consumed.
  const [pendingQuery, setPendingQuery] = useState(null)

  // Navigate to the Query tab, optionally seeding a query. `send` true auto-sends
  // it (single-doc review); false just prefills the input so the user can pick
  // which contract, then send (multi-doc review). `text` null/empty just switches
  // tab (the plain "Ask a question" action).
  function startQuery(text, send = true) {
    setPendingQuery(text && text.trim() ? { text: text.trim(), send } : null)
    goTab('query')
  }

  // Two navigation actions keep focus and activeTab in lockstep: selecting a tab
  // snaps focus to its tier; focusing a tier lands on that tier's default tab.
  function goTab(name) {
    setActiveTab(name)
    setFocus(TAB_TIER[name])
  }
  function goFocus(tier) {
    setFocus(tier)
    setActiveTab(DEFAULT_TAB[tier])
    if (tier === 'application') refreshApps()   // keep the App switcher current
  }

  // In saas mode the Settings tab is hidden (providers come from the service
  // config). A stale #settings hash could still land there, so bounce it to the
  // account tier's default tab once mode resolves.
  useEffect(() => {
    if (mode === 'saas' && activeTab === 'settings') goTab(DEFAULT_TAB['account'])
  }, [mode, activeTab])

  // Reflect the resolved account in the browser tab title so multiple accounts
  // open side by side are distinguishable. The default account is unnamed context
  // (dev/single-tenant), so it falls back to the bare brand.
  useEffect(() => {
    document.title = accountId && accountId !== 'default' ? `CogBase — ${accountId}` : 'CogBase'
  }, [accountId])

  // Populate the namespace switcher on mount and whenever the account changes
  // (refreshNamespaces' identity tracks the account via authFetch).
  useEffect(() => { refreshNamespaces() }, [refreshNamespaces])

  // Reconcile the working namespace against the account's real namespaces. The
  // selection persists in localStorage, so a value from another account/server — or
  // a since-deleted namespace — can survive into an account that doesn't have it,
  // surfacing in the switcher as a phantom. Once the list has loaded: if the account
  // has namespaces but the current one isn't among them, snap to a real one; if the
  // account has none, clear the selection so the sidebar shows the create-a-namespace
  // prompt instead of a phantom. Read the selection via a ref and depend only on the
  // loaded list, so this reacts to (re)loads, not keystrokes — typing a new deploy
  // target the deploy flow will create isn't clobbered mid-edit.
  const nsSelRef = useRef(namespaceName)
  nsSelRef.current = namespaceName
  useEffect(() => {
    if (!namespacesLoaded) return
    const cur = nsSelRef.current
    if (namespaces.length) {
      if (!namespaces.some(n => n.name === cur)) setNamespaceName(namespaces[0].name)
    } else if (cur) {
      setNamespaceName('')
    }
  }, [namespaces, namespacesLoaded, setNamespaceName])

  // Populate the App switcher on mount and whenever the namespace/account changes
  // (refreshApps' identity tracks nsBase).
  useEffect(() => { refreshApps() }, [refreshApps])

  // Reconcile the selection with the working namespace: switching namespaces leaves
  // currentApp pointing at the old app, which may not exist here. Once the new
  // namespace's apps have loaded (appsNs caught up), drop a selection that's absent
  // so the application tier falls back to its empty state instead of querying a
  // phantom app. Gated on appsNs === namespaceName so a deep-linked app isn't
  // wiped before its namespace list arrives.
  //
  // Suspended while a fresh-signup auto-select is pending: a stale application-tier
  // hash can survive into the seeded workspace (a prior session's deep link — the
  // login gate doesn't clear the URL), making currentApp a phantom the moment the
  // seeded namespace's apps load. Clearing it here would race the auto-select below,
  // which itself replaces that phantom with the seeded app. Let the one-shot own the
  // decision until it's consumed.
  useEffect(() => {
    if (autoSelectApp) return
    if (appsNs === namespaceName && currentApp && !apps.some(a => a.name === currentApp)) {
      setCurrentApp('')
    }
  }, [autoSelectApp, appsNs, apps, currentApp, namespaceName, setCurrentApp])

  // First landing after a brand-new-account signup: the server seeded a starter
  // workspace (legal-team namespace + contract-analyst app; api/provisioning.py). The
  // namespace auto-selects via the reconcile above; this adopts the provisioned app and
  // drops the owner on Ingest — the seeded app has no documents yet, so uploading is the
  // one thing to do next (rather than the pick-an-app empty state or the Build tab).
  // One-shot: raised only by the signup path (context.jsx) and consumed the moment the
  // seeded app loads, so a normal namespace switch never auto-picks an app or hijacks
  // navigation — that keeps the deliberate pick-an-app empty state for everyone else.
  useEffect(() => {
    if (!autoSelectApp || !namespaceName || appsNs !== namespaceName) return
    // Consume the flag only once the resolved namespace's apps have actually
    // arrived (apps.length) — otherwise the transient empty default-namespace
    // window would clear it before the seeded workspace loads. If provisioning
    // yielded no app, it simply lingers until one appears, then adopts it.
    if (apps.length) {
      // Adopt the seeded app on Ingest. Set the whole (app, focus, tab) tuple
      // imperatively AND mirror it into the hash in one shot: the imperative state
      // is what keeps currentApp a real selection (not the stale phantom the URL may
      // have deep-linked), so the reconcile above never clears it out from under us —
      // the failure mode when this relied on the hash round-trip alone, which left
      // currentApp stale until an async hashchange applied and let the reconcile +
      // hash-writer strand the owner on the pick-an-app empty state. Writing the hash
      // here too keeps the URL authoritative, so any stale in-flight hashchange (e.g.
      // the namespace-tier route queued when the namespace resolved) reads the live
      // app route rather than reverting the selection.
      const target = apps.some(a => a.name === currentApp) ? currentApp : apps[0].name
      setCurrentApp(target, namespaceName)
      setFocus('application')
      setActiveTab('ingest')
      window.location.hash = buildHash({ focus: 'application', namespaceName, currentApp: target, activeTab: 'ingest' })
      setAutoSelectApp(false)
    }
  }, [autoSelectApp, appsNs, namespaceName, apps, currentApp, setCurrentApp, setAutoSelectApp])

  // ── Hash routing (docs/ui-navigation.md, milestone B step 5) ──
  // A pure mirror of the (focus, namespace, app, tab) tuple onto location.hash, so
  // views deep-link and survive refresh. Applying a parsed hash only touches the
  // pieces it carries — a namespace/account hash leaves the selected app alone.
  function applyRoute(r) {
    if (r.namespaceName != null && r.namespaceName !== namespaceName) setNamespaceName(r.namespaceName)
    if (r.currentApp != null && r.currentApp !== currentApp) setCurrentApp(r.currentApp, r.namespaceName)
    setActiveTab(r.activeTab)
    setFocus(r.focus)
  }
  // Keep the latest applyRoute reachable from the mount-only listener without
  // re-subscribing (which would re-run the initial parse and revert live state).
  const applyRouteRef = useRef(applyRoute)
  applyRouteRef.current = applyRoute

  // Read: restore from the URL on mount (or seed it from default state), then
  // follow back/forward. Mount-only; the handler reads fresh state via the ref.
  useEffect(() => {
    const r0 = parseHash(window.location.hash)
    if (r0) applyRouteRef.current(r0)
    else window.location.hash = buildHash({ focus, namespaceName, currentApp, activeTab })
    const onHash = () => { const r = parseHash(window.location.hash); if (r) applyRouteRef.current(r) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Write: mirror subsequent changes into the hash. Skip the mount run — the reader
  // already reconciled the URL, and a just-applied initial route hasn't committed
  // to state yet, so writing here would clobber it. Guard on equality so the
  // reader's echo (and back/forward) don't loop.
  const hashMountedRef = useRef(false)
  useEffect(() => {
    if (!hashMountedRef.current) { hashMountedRef.current = true; return }
    const h = buildHash({ focus, namespaceName, currentApp, activeTab })
    if (window.location.hash !== h) window.location.hash = h
  }, [focus, namespaceName, currentApp, activeTab])

  // Sidebar nav grouped by scope tier: account ▸ namespace (workspace) ▸ the
  // selected application. Each group header focuses its tier; only the focused
  // tier's items are shown, so out-of-scope actions stay hidden rather than empty
  // (docs/ui-navigation.md, milestone B).
  // In saas mode the LLM/embedding providers come from the service config; an
  // account can't configure its own, so the Settings tab is hidden.
  const accountTabs = mode === 'saas' ? ['namespaces', 'skills'] : ['namespaces', 'skills', 'settings']
  const navGroups = [
    { tier: 'namespace',   label: t('nav.groupWorkspace'),   tabs: ['build', 'apps', 'demos'] },
    { tier: 'application', label: t('nav.groupApplication'), tabs: ['ingest', 'data', 'query', 'memory'] },
    { tier: 'account',     label: t('nav.groupAccount'),     tabs: accountTabs },
  ]

  // The application tier needs a selected app; until one is picked, its panels are
  // replaced by an empty state prompting selection.
  const appReady = !!currentApp
  const showEmpty = focus === 'application' && !appReady
  // The App switcher lists the working namespace's apps. Under the unified model
  // the selection lives in that namespace, but a manual namespace switch can leave
  // currentApp pointing elsewhere — show it selected only while it's in the list.
  const appSelectValue = apps.some(a => a.name === currentApp) ? currentApp : ''

  const nsSuggestions = nsOptions(namespaces.map(n => n.name), namespaceName)

  return (
    <>
      {/* Top bar: brand + app pill + language. The API base is auto-resolved from
          the serving origin (see context.jsx), so it's not surfaced here. */}
      <header>
        <h1>⚡ Cog<span>Base</span></h1>
        <div className={`app-pill ${currentApp ? 'on' : ''}`} title={currentApp ? t('header.appInNamespace', { namespace: namespaceName }) : undefined}>
          <span className="dot" />
          <span>{currentApp || t('header.noApp')}</span>
          {currentApp && <span className="app-pill-ns">{namespaceName}</span>}
        </div>
        {/* Signed-in user + sign-out, only in the authenticated (saas) mode. */}
        {mode === 'saas' && (
          <div className="user-row" title={email}>
            {email && <span className="user-email">{email}</span>}
            <button className="btn btn-sm" onClick={logout}>Sign out</button>
          </div>
        )}
        <div className="lang-row" title={t('header.language')}>
          <select className="lang-select" value={lang} onChange={e => setLang(e.target.value)} aria-label={t('header.language')}>
            {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
        </div>
      </header>

      <div className="shell">
        {/* Collapsed to a thin rail: a single toggle restores the sidebar. Mirrors
            the collapsed-panel rail pattern used for the chat/doc panels. */}
        {navHidden && (
          <div className="sidebar-rail">
            <button className="aside-toggle" title={t('nav.show')} aria-label={t('nav.show')} onClick={() => setNavHidden(false)}><NavPanelIcon /></button>
          </div>
        )}
        {/* Focus-driven sidebar nav. Kept mounted even when hidden (collapsed via
            CSS) so the nav-secondary portal slot below never detaches — otherwise a
            hide would strand the ported chats/collections list. */}
        <aside className={`sidebar${navHidden ? ' collapsed' : ''}`}>
          {/* Hide control: collapses the whole sidebar (nav + contextual list) to
              the rail above. */}
          <div className="sidebar-hd">
            <button className="aside-toggle" title={t('nav.hide')} aria-label={t('nav.hide')} onClick={() => setNavHidden(true)}><NavPanelIcon /></button>
          </div>
          {/* Namespace switcher scopes every tier below. The account is the
              tenant/security boundary, not a nav dimension — it's resolved by the
              server via /whoami and shown read-only in the top bar, not here. */}
          <div className="side-switch">
            <label htmlFor="namespaceName">{t('header.namespaceLabel')}</label>
            {namespacesLoaded && namespaces.length === 0 ? (
              /* Fresh account with no namespaces yet — prompt to create one rather
                 than present a phantom selection. Creating a namespace here (or
                 deploying an app, which creates its target) populates the switcher. */
              <div className="ns-empty">
                <p className="side-hint">{t('nav.nsNoneHint')}</p>
                <button className="btn btn-primary btn-sm" onClick={() => goTab('namespaces')}>
                  {t('nav.nsNoneCta')}
                </button>
              </div>
            ) : (
              /* Filtering combobox: lists the account's namespaces and substring-
                 filters them as you type, but an arbitrary namespace can still be
                 committed (e.g. to deploy into one that doesn't exist yet — the
                 deploy flow creates it first). See components/NamespaceSelect.jsx. */
              <NamespaceSelect id="namespaceName" value={namespaceName} options={nsSuggestions} onChange={setNamespaceName} />
            )}
          </div>

          {navGroups.map(group => {
            const open = focus === group.tier
            return (
              <div className={`nav-group ${open ? 'open' : ''}`} key={group.tier}>
                <button className={`nav-group-header ${open ? 'active' : ''}`} onClick={() => goFocus(group.tier)}>
                  {group.label}
                </button>
                {/* The App switcher scopes the application tier, so it heads its items */}
                {open && group.tier === 'application' && (
                  <div className="side-switch nested">
                    <label htmlFor="appSelect">{t('nav.appLabel')}</label>
                    <select id="appSelect" value={appSelectValue} onChange={e => setCurrentApp(e.target.value)}>
                      <option value="" disabled hidden>{t('nav.appPlaceholder')}</option>
                      {apps.map(a => <option key={a.name} value={a.name}>{a.name}</option>)}
                    </select>
                    {apps.length === 0 && <div className="side-hint">{t('nav.appNoneInNs')}</div>}
                  </div>
                )}
                {open && group.tabs.map(tab => (
                  <button key={tab} className={`side-item ${activeTab === tab ? 'active' : ''}`} onClick={() => goTab(tab)}>
                    {t(`nav.${tab}`)}
                  </button>
                ))}
              </div>
            )
          })}

          {/* Contextual list slot: the application-tier tabs portal their secondary
              list here — Query its chats, Data its collections. Empty (and invisible)
              for tabs without one. Scrolls on its own so the nav above stays put. */}
          <div className="nav-secondary" ref={setNavSlot} />
        </aside>

        <main>
        <div className={`panel ${activeTab === 'build' ? 'active' : ''}`}>
          <BuildTab active={activeTab === 'build'} />
        </div>
        <div className={`panel ${activeTab === 'apps' ? 'active' : ''}`}>
          <AppsTab active={activeTab === 'apps'} onSwitchTab={goTab} />
        </div>
        <div className={`panel ${activeTab === 'namespaces' ? 'active' : ''}`}>
          <NamespacesTab active={activeTab === 'namespaces'} />
        </div>
        <div className={`panel ${activeTab === 'demos' ? 'active' : ''}`}>
          <DemosTab
            active={activeTab === 'demos'}
            onOpenDocModal={setDocModal}
            onOpenConfigModal={demo => setConfigModal({ demo })}
            onOpenWfModal={setWfModal}
            onSwitchTab={goTab}
          />
        </div>
        <div className={`panel ${activeTab === 'ingest' && !showEmpty ? 'active' : ''}`}>
          <IngestTab
            active={activeTab === 'ingest' && !showEmpty}
            refreshKey={ingestRefreshKey}
            onOpenTaskProgress={setTaskProgress}
            onOpenWfModal={setWfModal}
            onDocsChanged={() => setDataRefreshKey(k => k + 1)}
            onStartQuery={startQuery}
          />
        </div>
        <div className={`panel ${activeTab === 'data' && !showEmpty ? 'active' : ''}`}>
          <DataTab
            active={activeTab === 'data' && !showEmpty}
            refreshKey={dataRefreshKey}
            onOpenWfModal={setWfModal}
            wfCompleteCollection={wfCompleteCollection}
            onWfCompleteHandled={() => setWfCompleteCollection(null)}
            navSlot={navSlot}
          />
        </div>
        <div className={`panel ${activeTab === 'query' && !showEmpty ? 'active' : ''}`}>
          <QueryTab
            active={activeTab === 'query' && !showEmpty}
            pendingQuery={pendingQuery}
            onPendingConsumed={() => setPendingQuery(null)}
            navSlot={navSlot}
          />
        </div>
        <div className={`panel ${activeTab === 'memory' && !showEmpty ? 'active' : ''}`}>
          <MemoryTab active={activeTab === 'memory' && !showEmpty} />
        </div>
        {/* Application tier with no app selected → prompt to pick one */}
        <div className={`panel ${showEmpty ? 'active' : ''}`}>
          {showEmpty && (
            <div className="app-empty">
              <div className="app-empty-icon">📦</div>
              <p>{t('nav.appTierEmptyTitle')}</p>
              <button className="btn btn-primary" onClick={() => goTab('apps')}>{t('nav.appTierEmptyCta')}</button>
            </div>
          )}
        </div>
        <div className={`panel ${activeTab === 'skills' ? 'active' : ''}`}>
          <SkillsTab active={activeTab === 'skills'} />
        </div>
        <div className={`panel ${activeTab === 'settings' ? 'active' : ''}`}>
          <SettingsTab active={activeTab === 'settings'} onAutoSwitch={() => goTab('settings')} />
        </div>
        </main>
      </div>

      {/* Modals */}
      <DocModal doc={docModal} onClose={() => setDocModal(null)} />
      <ConfigModal data={configModal} onClose={() => setConfigModal(null)} />
      <WfModal
        state={wfModal}
        onClose={() => {
          const fromIngest = wfModal && wfModal.fromIngest
          const saveCollection = wfModal && wfModal.saveCollection
          setWfModal(null)
          if (fromIngest) setIngestRefreshKey(k => k + 1)
          if (saveCollection) setWfCompleteCollection(saveCollection)
        }}
      />
      <TaskProgressModal
        data={taskProgress}
        onClose={() => setTaskProgress(null)}
        onDone={() => setIngestRefreshKey(k => k + 1)}
      />
    </>
  )
}

// Sidebar-collapse glyph, matching the framed-rectangle-with-divided-column icon
// ChatGPT/Claude use for the same control (and the panel toggles elsewhere in the UI).
function NavPanelIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <line x1="9" y1="3" x2="9" y2="21" />
    </svg>
  )
}

// Auth gate: resolves the deployment mode + identity from /whoami on mount, then
// either shows the login screen (saas mode with no valid session) or the app. In
// dev/single_tenant/demo modes there is no gate — the app renders straight away,
// exactly as before. Bootstrap lives here (not in Layout) so it also runs while
// unauthenticated, before Layout ever mounts.
function AuthGate() {
  const { mode, accessToken, bootstrap } = useApp()
  useEffect(() => { bootstrap() }, [bootstrap])

  // Only saas mode gates. mode is seeded from the last-known value (persisted in
  // context) and confirmed by /whoami, so a returning saas visitor sees the login
  // screen immediately rather than a flash of the app shell. dev/single_tenant/demo
  // render the app straight through, exactly as before.
  if (mode === 'saas' && !accessToken) {
    return <LoginScreen />
  }
  return <Layout />
}

export default function App() {
  return <I18nProvider><AppProvider><AuthGate /></AppProvider></I18nProvider>
}
