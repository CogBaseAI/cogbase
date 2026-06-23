import React, { useState } from 'react'
import { AppProvider, useApp } from './context'
import BuildTab from './components/tabs/BuildTab'
import AppsTab from './components/tabs/AppsTab'
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

function Layout() {
  const { apiUrl, setApiUrl, currentApp, llmConfigured, embConfigured } = useApp()
  const [activeTab, setActiveTab] = useState('build')
  const [docModal, setDocModal] = useState(null)        // null | doc object
  const [configModal, setConfigModal] = useState(null)  // null | { demo }
  const [wfModal, setWfModal] = useState(null)          // null | { appName, workflowName, paramKey, label, values, desc, allDone, fromIngest }
  const [taskProgress, setTaskProgress] = useState(null) // null | { appName, workflowName, docId }
  const [ingestRefreshKey, setIngestRefreshKey] = useState(0)
  const [wfCompleteCollection, setWfCompleteCollection] = useState(null)

  function switchTab(name) {
    setActiveTab(name)
  }

  const tabs = ['build', 'apps', 'demos', 'ingest', 'data', 'query', 'memory', 'skills', 'settings']

  return (
    <>
      {/* Header */}
      <header>
        <h1>⚡ Cog<span>Base</span></h1>
        <div className="api-row">
          <label htmlFor="apiUrl">API</label>
          <input id="apiUrl" type="text" value={apiUrl} onChange={e => setApiUrl(e.target.value.replace(/\/$/, ''))} placeholder="http://localhost:8000" />
        </div>
        <div className={`app-pill ${currentApp ? 'on' : ''}`}>
          <span className="dot" />
          <span>{currentApp || 'No app selected'}</span>
        </div>
      </header>

      {/* Tab nav */}
      <nav>
        {tabs.map(tab => (
          <button key={tab} className={activeTab === tab ? 'active' : ''} onClick={() => switchTab(tab)}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      <main>
        <div className={`panel ${activeTab === 'build' ? 'active' : ''}`}>
          <BuildTab active={activeTab === 'build'} />
        </div>
        <div className={`panel ${activeTab === 'apps' ? 'active' : ''}`}>
          <AppsTab active={activeTab === 'apps'} onSwitchTab={switchTab} />
        </div>
        <div className={`panel ${activeTab === 'demos' ? 'active' : ''}`}>
          <DemosTab
            active={activeTab === 'demos'}
            onOpenDocModal={setDocModal}
            onOpenConfigModal={demo => setConfigModal({ demo })}
            onOpenWfModal={setWfModal}
            onSwitchTab={switchTab}
          />
        </div>
        <div className={`panel ${activeTab === 'ingest' ? 'active' : ''}`}>
          <IngestTab
            active={activeTab === 'ingest'}
            refreshKey={ingestRefreshKey}
            onOpenTaskProgress={setTaskProgress}
            onOpenWfModal={setWfModal}
          />
        </div>
        <div className={`panel ${activeTab === 'data' ? 'active' : ''}`}>
          <DataTab
            active={activeTab === 'data'}
            onOpenWfModal={setWfModal}
            wfCompleteCollection={wfCompleteCollection}
            onWfCompleteHandled={() => setWfCompleteCollection(null)}
          />
        </div>
        <div className={`panel ${activeTab === 'query' ? 'active' : ''}`}>
          <QueryTab active={activeTab === 'query'} />
        </div>
        <div className={`panel ${activeTab === 'memory' ? 'active' : ''}`}>
          <MemoryTab active={activeTab === 'memory'} />
        </div>
        <div className={`panel ${activeTab === 'skills' ? 'active' : ''}`}>
          <SkillsTab active={activeTab === 'skills'} />
        </div>
        <div className={`panel ${activeTab === 'settings' ? 'active' : ''}`}>
          <SettingsTab active={activeTab === 'settings'} onAutoSwitch={() => switchTab('settings')} />
        </div>
      </main>

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
  return <AppProvider><Layout /></AppProvider>
}
