import { useCallback, useEffect, useState } from 'react'
import Papa from 'papaparse'

const API_URL = 'http://localhost:8000'
const SAMPLE_SIZE = 10

const VISIBLE_COLUMNS = [
  'project',
  'process',
  'hazard name',
  'barrier',
  'critical surfaces',
  'interaction',
  'visibility',
  'distance to object',
  'size',
  'weight',
  'handling'
]

function moveItem(list, fromIndex, toIndex) {
  const next = [...list]
  const [moved] = next.splice(fromIndex, 1)
  next.splice(toIndex, 0, moved)
  return next
}

function getNewlyOrderedList(list, fromIndex, toIndex) {
  return moveItem(list, fromIndex, toIndex)
}

export default function App() {
  const [apiColumns, setApiColumns] = useState([])
  const [items, setItems] = useState([])
  const [dragIndex, setDragIndex] = useState(null)
  const [status, setStatus] = useState('Loading scenarios...')

  const fetchScenarios = useCallback(async () => {
    setStatus('Loading scenarios...')
    try {
      const response = await fetch(`${API_URL}/scenarios?n=${SAMPLE_SIZE}`)
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`)
      }

      const json = await response.json()
      const rows = json.rows ?? []
      const cols = json.columns ?? (rows.length > 0 ? Object.keys(rows[0]) : [])

      setApiColumns(cols)
      setItems(rows.map((row, index) => ({ id: `${Date.now()}-${index}`, row })))
      setStatus(`Loaded ${rows.length} random scenarios (${json.total} total)`)
    } catch (error) {
      setStatus(`Failed to load scenarios: ${error.message}`)
    }
  }, [])

  useEffect(() => {
    fetchScenarios()
  }, [fetchScenarios])

  const handleDrop = (dropIndex) => {
    if (dragIndex === null || dragIndex === dropIndex) {
      setDragIndex(null)
      return
    }

    const reordered = getNewlyOrderedList(items, dragIndex, dropIndex)
    setItems(reordered)
    setDragIndex(null)

    // This returns the reordered rows.
    console.log(
      'Newly ordered list:',
      reordered.map((item) => item.row)
    )
  }

  const handleGetNewData = () => {
    fetchScenarios()
  }

  const handleDone = () => {
    if (items.length === 0) {
      setStatus('No rows to export.')
      return
    }

    const headers = apiColumns.length > 0 ? apiColumns : Object.keys(items[0].row)
    const data = items.map((item) => headers.map((header) => item.row[header] ?? ''))
    const csv = Papa.unparse({
      fields: headers,
      data
    })

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'risk_summary_reordered.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)

    setStatus('Exported reordered data to risk_summary_reordered.csv')
  }

  const allHeaders = apiColumns.length > 0 ? apiColumns : Object.keys(items[0]?.row ?? {})
  const visibleHeaders = allHeaders.filter(h => VISIBLE_COLUMNS.includes(h.toLowerCase()))

  return (
    <main className="app">
      <section className="card">
        <img src="/logo-innerspace.png" alt="Logo" className="headerLogo" />
        <h1>Order Hazard Scenarios</h1>
        <p className="hint">Showing 10 random scenarios from the dataset.</p>
        <p className="statusText">{status}</p>
        <div className="buttonRow">
          <button className="refreshButton" type="button" onClick={handleGetNewData}>
            Get New Data
          </button>
          <button className="doneButton" type="button" onClick={handleDone}>
            Done
          </button>
        </div>

        <div className="tableWrap" aria-label="Reorderable table">
          <table className="dataTable">
            <thead>
              <tr>
                <th className="gripHeader" aria-label="Drag handle">
                  Drag
                </th>
                {visibleHeaders.map((header) => (
                  <th key={header}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => (
                <tr
                  key={item.id}
                  className={`tableRow ${dragIndex === index ? 'dragging' : ''}`}
                  draggable
                  onDragStart={() => setDragIndex(index)}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={() => handleDrop(index)}
                  onDragEnd={() => setDragIndex(null)}
                >
                  <td className="gripCell" aria-hidden="true">
                    ::
                  </td>
                  {visibleHeaders.map((header) => (
                    <td key={`${item.id}-${header}`} className="rowCell">
                      {String(item.row[header] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* <h2>Newly Ordered List</h2> */}
        {/* <pre className="output">{JSON.stringify(items.map((item) => item.row), null, 2)}</pre> */}
      </section>
    </main>
  )
}
