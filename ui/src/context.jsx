import React, { createContext, useContext, useState } from 'react'

const AppCtx = createContext(null)

export function AppProvider({ children }) {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
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
