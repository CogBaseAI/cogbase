import React, { useState, useEffect, useRef } from 'react'
import { AppProvider, useApp } from './context'
import { TAB_TIER, DEFAULT_TAB, buildHash, parseHash, nsOptions } from './nav'
import { I18nProvider, useT, LANGUAGES } from './i18n'
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
import DocModal from './components/modals/DocModal'
import ConfigModal from './components/modals/ConfigModal'
import WfModal from './components/modals/WfModal'
import TaskProgressModal from './components/modals/TaskProgressModal'

// TAB_TIER / DEFAULT_TAB (the tab→tier grouping and each tier's landing tab) live
// in ./nav alongside the hash router that also consumes them.

function Layout() {
  const { apiUrl, setApiUrl, accountId, setAccountId, namespaceName, setNamespaceName, namespaces, refreshNamespaces, apps, appsNs, refreshApps, currentApp, setCurrentApp } = useApp()
  const { t, lang, setLang } = useT()
  const [activeTab, setActiveTab] = useState('build')
  const [focus, setFocus] = useState(TAB_TIER['build'])   // which tier's sub-nav shows
  const [docModal, setDocModal] = useState(null)        // null | doc object
  const [configModal, setConfigModal] = useState(null)  // null | { demo }
  const [wfModal, setWfModal] = useState(null)          // null | { appName, workflowName, paramKey, label, values, desc, allDone, fromIngest }
  const [taskProgress, setTaskProgress] = useState(null) // null | { appName, workflowName, docId }
  const [ingestRefreshKey, setIngestRefreshKey] = useState(0)
  const [dataRefreshKey, setDataRefreshKey] = useState(0)
  const [wfCompleteCollection, setWfCompleteCollection] = useState(null)

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

  // Populate the namespace switcher on mount and whenever the account changes
  // (refreshNamespaces' identity tracks the account via authFetch).
  useEffect(() => { refreshNamespaces() }, [refreshNamespaces])

  // Populate the App switcher on mount and whenever the namespace/account changes
  // (refreshApps' identity tracks nsBase).
  useEffect(() => { refreshApps() }, [refreshApps])

  // Reconcile the selection with the working namespace: switching namespaces leaves
  // currentApp pointing at the old app, which may not exist here. Once the new
  // namespace's apps have loaded (appsNs caught up), drop a selection that's absent
  // so the application tier falls back to its empty state instead of querying a
  // phantom app. Gated on appsNs === namespaceName so a deep-linked app isn't
  // wiped before its namespace list arrives.
  useEffect(() => {
    if (appsNs === namespaceName && currentApp && !apps.some(a => a.name === currentApp)) {
      setCurrentApp('')
    }
  }, [appsNs, apps, currentApp, namespaceName, setCurrentApp])

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
  const navGroups = [
    { tier: 'namespace',   label: t('nav.groupWorkspace'),   tabs: ['build', 'apps', 'demos'] },
    { tier: 'application', label: t('nav.groupApplication'), tabs: ['ingest', 'data', 'query', 'memory'] },
    { tier: 'account',     label: t('nav.groupAccount'),     tabs: ['namespaces', 'skills', 'settings'] },
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
      {/* Top bar: brand + environment (API url) + language */}
      <header>
        <h1>⚡ Cog<span>Base</span></h1>
        <div className="api-row">
          <label htmlFor="apiUrl">{t('header.apiLabel')}</label>
          <input id="apiUrl" type="text" value={apiUrl} onChange={e => setApiUrl(e.target.value.replace(/\/$/, ''))} placeholder={t('header.apiPlaceholder')} />
        </div>
        <div className={`app-pill ${currentApp ? 'on' : ''}`} title={currentApp ? t('header.appInNamespace', { namespace: namespaceName }) : undefined}>
          <span className="dot" />
          <span>{currentApp || t('header.noApp')}</span>
          {currentApp && <span className="app-pill-ns">{namespaceName}</span>}
        </div>
        <div className="lang-row" title={t('header.language')}>
          <select className="lang-select" value={lang} onChange={e => setLang(e.target.value)} aria-label={t('header.language')}>
            {LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
          </select>
        </div>
      </header>

      <div className="shell">
        {/* Focus-driven sidebar nav */}
        <aside className="sidebar">
          {/* Context switchers — account and namespace scope every tier below.
              (In milestone B step 3 these graduate into the top-bar breadcrumb.) */}
          <div className="side-switch">
            <label htmlFor="accountId">{t('header.accountLabel')}</label>
            <input id="accountId" type="text" value={accountId} onChange={e => setAccountId(e.target.value)} />
          </div>
          <div className="side-switch">
            <label htmlFor="namespaceName">{t('header.namespaceLabel')}</label>
            {/* Filtering combobox: lists the account's namespaces and substring-
                filters them as you type, but an arbitrary namespace can still be
                committed (e.g. to deploy into one that doesn't exist yet — deploy
                registers it). See components/NamespaceSelect.jsx. */}
            <NamespaceSelect id="namespaceName" value={namespaceName} options={nsSuggestions} onChange={setNamespaceName} />
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
                      <option value="">{t('nav.appPlaceholder')}</option>
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
          />
        </div>
        <div className={`panel ${activeTab === 'data' && !showEmpty ? 'active' : ''}`}>
          <DataTab
            active={activeTab === 'data' && !showEmpty}
            refreshKey={dataRefreshKey}
            onOpenWfModal={setWfModal}
            wfCompleteCollection={wfCompleteCollection}
            onWfCompleteHandled={() => setWfCompleteCollection(null)}
          />
        </div>
        <div className={`panel ${activeTab === 'query' && !showEmpty ? 'active' : ''}`}>
          <QueryTab active={activeTab === 'query' && !showEmpty} />
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

export default function App() {
  return <I18nProvider><AppProvider><Layout /></AppProvider></I18nProvider>
}
