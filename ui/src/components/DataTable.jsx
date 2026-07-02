import React, { useState, useMemo } from 'react'
import { useT } from '../i18n'

// Flatten a cell value to a plain string for search/sort.
function cellText(val) {
  if (val === null || val === undefined) return ''
  if (typeof val === 'object') return JSON.stringify(val)
  return String(val)
}

// Reusable table carrying the Data-tab affordances: a search box, click-to-sort
// headers, drag-to-resize columns, and a floating full-text preview for
// truncated cells. Columns are declared with optional render/value/text/sortValue
// so each tab keeps its custom cells (badges, buttons) while sharing the
// behaviour.
//
//   column: {
//     key,                unique id (required)
//     label,              header text
//     render(row, i),     custom cell node (badges, buttons); opts out of the hover tip
//     value(row),         raw value for a plain cell (falls back to row[key]); keeps
//                         null/object styling and gets the hover preview
//     text(row),          string for search + default sort (falls back to the value)
//     sortValue(row),     comparison key for sort (falls back to text/value)
//     sortable,           default: false for render-only columns, true otherwise
//     searchable,         default: false for render-only columns, true otherwise
//     width, align, cellClassName
//   }
export default function DataTable({
  columns, rows, rowKey, rowNum = true, rowClassName, onRowClick,
  searchable = true, filters = null, count = null, searchPlaceholder,
}) {
  const { t } = useT()
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState(null)   // null | { key, dir: 'asc'|'desc' }
  const [colWidths, setColWidths] = useState({}) // { key -> px }
  const [tip, setTip] = useState(null)     // null | { text, x, y }

  const rawValue = (col, row) => (col.value ? col.value(row) : row[col.key])
  const textOf = (col, row) =>
    col.text ? col.text(row) : (col.render ? '' : cellText(rawValue(col, row)))
  const sortValOf = (col, row) =>
    col.sortValue ? col.sortValue(row) : (col.text ? col.text(row) : rawValue(col, row))

  const canSort = col => col.sortable ?? (!col.render || !!col.text || !!col.sortValue)
  const canSearch = col => col.searchable ?? (!col.render || !!col.text)

  function toggleSort(key) {
    setSort(prev => {
      if (!prev || prev.key !== key) return { key, dir: 'asc' }
      if (prev.dir === 'asc') return { key, dir: 'desc' }
      return null // third click clears the sort
    })
  }

  // Drag the handle on a header's right edge to set an explicit width; widening
  // a column lets its cells show more text before truncating.
  function startResize(key, e) {
    e.preventDefault()
    e.stopPropagation()
    const startX = e.clientX
    const startW = e.currentTarget.parentElement.getBoundingClientRect().width
    const onMove = ev => setColWidths(prev => ({ ...prev, [key]: Math.max(60, Math.round(startW + ev.clientX - startX)) }))
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      document.body.classList.remove('col-resizing')
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    document.body.classList.add('col-resizing')
  }

  const searchCols = useMemo(() => columns.filter(canSearch), [columns])

  // Filter (across searchable columns) then sort — both client-side.
  const viewRows = useMemo(() => {
    let rs = rows || []
    const q = search.trim().toLowerCase()
    rs = q
      ? rs.filter(row => searchCols.some(c => textOf(c, row).toLowerCase().includes(q)))
      : rs.slice()
    if (sort) {
      const col = columns.find(c => c.key === sort.key)
      if (col) {
        const { dir } = sort
        rs.sort((a, b) => {
          const av = sortValOf(col, a), bv = sortValOf(col, b)
          // Nulls always sort last, regardless of direction.
          const aNull = av === null || av === undefined || av === ''
          const bNull = bv === null || bv === undefined || bv === ''
          if (aNull || bNull) return aNull === bNull ? 0 : aNull ? 1 : -1
          let cmp
          if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv
          else cmp = cellText(av).localeCompare(cellText(bv), undefined, { numeric: true, sensitivity: 'base' })
          return dir === 'asc' ? cmp : -cmp
        })
      }
    }
    return rs
  }, [rows, columns, searchCols, search, sort])

  const showSearch = searchable && rows && rows.length > 0

  return (
    <>
      {(showSearch || filters || count != null) && (
        <div className="dt-toolbar">
          <div className="dt-toolbar-left">
            {filters}
            {showSearch && (
              <input
                className="data-search"
                type="search"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder={searchPlaceholder || t('data.searchPlaceholder')}
              />
            )}
          </div>
          {count != null && <span className="meta">{count}</span>}
        </div>
      )}
      <div className="dt-wrap">
        <table className="data-tbl">
          <thead>
            <tr>
              {rowNum && <th className="row-num-th">#</th>}
              {columns.map(col => {
                const sortable = canSort(col)
                const dir = sort?.key === col.key ? sort.dir : null
                const w = colWidths[col.key] || col.width
                return (
                  <th
                    key={col.key}
                    className={sortable ? 'sortable' : undefined}
                    style={w ? { width: w, minWidth: w, maxWidth: w } : undefined}
                    onClick={sortable ? () => toggleSort(col.key) : undefined}
                  >
                    <span className="th-label" title={col.label}>{col.label}</span>
                    {sortable && <span className="sort-caret">{dir === 'asc' ? ' ▲' : dir === 'desc' ? ' ▼' : ''}</span>}
                    <span className="col-resize-handle" onClick={e => e.stopPropagation()} onMouseDown={e => startResize(col.key, e)} />
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {viewRows.map((row, i) => (
              <tr
                key={rowKey(row, i)}
                className={rowClassName ? rowClassName(row) : undefined}
                onClick={onRowClick ? () => { setTip(null); onRowClick(row) } : undefined}
              >
                {rowNum && <td className="row-num">{i + 1}</td>}
                {columns.map(col => (
                  <Cell key={col.key} col={col} row={row} i={i} width={colWidths[col.key] || col.width} rawValue={rawValue} setTip={setTip} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows && rows.length > 0 && viewRows.length === 0 && (
          <div className="empty"><div className="ei">🔍</div><p>{t('data.noMatch')}</p></div>
        )}
      </div>
      {tip && <CellTip text={tip.text} x={tip.x} y={tip.y} />}
    </>
  )
}

// One table cell: custom render passes through untouched; a plain value cell
// stays single-line (truncated with ellipsis) and reveals its full content in a
// floating preview on hover.
function Cell({ col, row, i, width, rawValue, setTip }) {
  const style = width ? { maxWidth: width, width } : undefined
  const cls = [col.cellClassName, col.align ? `ta-${col.align}` : ''].filter(Boolean).join(' ') || undefined
  if (col.render) return <td className={cls} style={style}>{col.render(row, i)}</td>

  const val = rawValue(col, row)
  if (val === null || val === undefined) return <td className={[cls, 'null-val'].filter(Boolean).join(' ')} style={style}>—</td>
  const isObj = typeof val === 'object'
  const str = isObj ? JSON.stringify(val) : String(val)
  const objCls = isObj ? 'obj-val' : ''
  const fullCls = [cls, objCls].filter(Boolean).join(' ') || undefined
  if (str.length > 80) {
    const full = isObj ? JSON.stringify(val, null, 2) : str
    return (
      <td
        className={fullCls}
        style={style}
        onMouseEnter={e => setTip({ text: full, x: e.clientX, y: e.clientY })}
        onMouseMove={e => setTip(prev => (prev ? { ...prev, x: e.clientX, y: e.clientY } : prev))}
        onMouseLeave={() => setTip(null)}
      >{str}</td>
    )
  }
  return <td className={fullCls} style={style} title={str}>{str}</td>
}

// Floating preview of a truncated cell, following the cursor and flipping away
// from the viewport edges. pointer-events:none (in CSS) keeps it from stealing
// the hover, so the cell's own mouseleave still fires.
function CellTip({ text, x, y }) {
  const W = 380, GAP = 14
  const left = x + GAP + W > window.innerWidth ? Math.max(8, x - GAP - W) : x + GAP
  const top = Math.min(y + GAP, Math.max(8, window.innerHeight - 8 - 240))
  return <div className="cell-tip" style={{ left, top, maxWidth: W }}>{text}</div>
}
