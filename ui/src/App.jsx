import React, { useState, useEffect } from 'react'
import { AppProvider, useApp } from './context'
import { I18nProvider, useT, LANGUAGES } from './i18n'
import BuildTab from './components/tabs/BuildTab'
import AppsTab from './components/tabs/AppsTab'
import NamespacesTab from './components/tabs/NamespacesTab'
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

// Each tab belongs to one scope tier; the sidebar shows one tier ("focus") at a
// time (docs/ui-navigation.md, milestone B). This map is the single source of the
// tab→tier grouping, and DEFAULT_TAB is where focusing a tier lands you.
const TAB_TIER = {
  build: 'namespace', apps: 'namespace', demos: 'namespace',
  ingest: 'application', data: 'application', query: 'application', memory: 'application',
  namespaces: 'account', skills: 'account', settings: 'account',
}
const DEFAULT_TAB = { account: 'namespaces', namespace: 'apps', application: 'query' }

function Layout() {
  const { apiUrl, setApiUrl, accountId, setAccountId, namespaceName, setNamespaceName, namespaces, refreshNamespaces, apps, refreshApps, currentApp, setCurrentApp, llmConfigured, embConfigured } = useApp()
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

  // Namespace suggestions: the account's namespaces, plus 'default' and whatever
  // is currently typed, de-duplicated so the active value is always offered.
  const nsOptions = [...new Set(['default', namespaceName, ...namespaces.map(n => n.name)])].filter(Boolean)

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
            {/* Free-text input backed by a datalist: the account's namespaces are
                offered as suggestions, but an arbitrary namespace can still be typed
                (e.g. to deploy into one that doesn't exist yet — deploy registers it). */}
            <input id="namespaceName" type="text" list="ns-options" value={namespaceName} onChange={e => setNamespaceName(e.target.value)} />
            <datalist id="ns-options">
              {nsOptions.map(ns => <option key={ns} value={ns} />)}
            </datalist>
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
