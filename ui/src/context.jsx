import React, { createContext, useContext, useState } from 'react'

const AppCtx = createContext(null)

export function AppProvider({ children }) {
  // Default to the origin the UI was served from so it works whether run
  // locally or on a remote node. When running the vite dev server (port 5173),
  // the API lives separately on localhost:8000.
  const defaultApiUrl =
    typeof window !== 'undefined' && window.location.port !== '5173'
      ? window.location.origin
      : 'http://localhost:8000'
  const [apiUrl, setApiUrl] = useState(defaultApiUrl)
  const [currentApp, setCurrentApp] = useState('')
  const [demoCatalog, setDemoCatalog] = useState([])
  const [llmConfigured, setLlmConfigured] = useState(false)
  const [embConfigured, setEmbConfigured] = useState(false)

  return (
    <AppCtx.Provider value={{ apiUrl, setApiUrl, currentApp, setCurrentApp, demoCatalog, setDemoCatalog, llmConfigured, setLlmConfigured, embConfigured, setEmbConfigured }}>
      {children}
    </AppCtx.Provider>
  )
}

export function useApp() { return useContext(AppCtx) }
