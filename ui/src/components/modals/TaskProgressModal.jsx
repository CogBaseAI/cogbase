import React, { useState, useEffect, useRef } from 'react'
import { useApp } from '../../context'

export default function TaskProgressModal({ data, onClose, onDone }) {
  const { apiUrl } = useApp()
  const [task, setTask] = useState(null)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!data) { clearInterval(timerRef.current); return }
    setTask(null); setError(null)
    async function poll() {
      try {
        const resp = await fetch(
          `${apiUrl}/applications/${encodeURIComponent(data.appName)}/tasks?task_type=workflow&task_name=${encodeURIComponent(data.workflowName)}&doc_id=${encodeURIComponent(data.docId)}`
        )
        if (!resp.ok) { setError(`Failed to load task (${resp.status}).`); clearInterval(timerRef.current); return }
        const d = await resp.json()
        const tasks = (d.tasks || []).slice().sort((a, b) => a.created_at < b.created_at ? -1 : 1)
        if (!tasks.length) { setError('No task found for this document.'); clearInterval(timerRef.current); return }
        const t = tasks[tasks.length - 1]
        setTask(t)
        const active = t.status === 'pending' || t.status === 'running'
        if (!active) {
          clearInterval(timerRef.current)
          if (t.status === 'done' && onDone) onDone()
        }
      } catch (e) { setError('Error: ' + e.message); clearInterval(timerRef.current) }
    }
    poll()
    timerRef.current = setInterval(poll, 2000)
    return () => clearInterval(timerRef.current)
  }, [data])

  if (!data) return null
  const active = task && (task.status === 'pending' || task.status === 'running')

  return (
    <div className="task-progress-modal show" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="task-progress-panel" role="dialog" aria-modal="true">
        <div className="task-progress-hd">
          <h3>{data.workflowName}</h3>
          <button className="task-progress-close" onClick={onClose}>✕</button>
        </div>
        {error && <p style={{ color: 'var(--red)', fontSize: 12 }}>{error}</p>}
        {!task && !error && <p style={{ color: 'var(--muted)', fontSize: 12 }}>Loading…</p>}
        {task && (
          <>
            <div className="task-progress-row"><strong>Document:</strong> {data.docId}</div>
            <div className="task-progress-row" style={{ marginBottom: 12 }}>
              {active && <span className="task-progress-spinner" />}
              <span className={`ingest-status ${task.status}`}>{task.status}</span>
              {task.error && <span style={{ color: 'var(--red)', fontSize: 11, marginLeft: 6 }}>{task.error}</span>}
            </div>
            {task.started_at && <div className="task-progress-row"><strong>Started:</strong> {task.started_at.replace('T', ' ').slice(0, 16)}</div>}
            {task.completed_at && <div className="task-progress-row"><strong>Completed:</strong> {task.completed_at.replace('T', ' ').slice(0, 16)}</div>}
          </>
        )}
      </div>
    </div>
  )
}
