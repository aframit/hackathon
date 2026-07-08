import { useCallback, useEffect, useState } from 'react'

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

function SummaryPage({ items, visibleHeaders, result, status, onBack, onStartNew }) {
  return (
    <main className="app">
      <section className="card">
        <img src="/logo-innerspace.png" alt="Logo" className="headerLogo" />
        <h1>Ranking Summary</h1>

        {result && (
          <div className="results">
            <h2>Re-fit results ({result.param})</h2>

            <h3>Fitted encoding (label → score)</h3>
            <ul>
              {Object.entries(result.encoding).map(([label, score]) => (
                <li key={label}>{label}: {Number(score).toFixed(3)}</li>
              ))}
            </ul>

            <h3>Re-ranked list (most critical first)</h3>
            <ol>
              {result.fitted_list.map((row) => (
                <li key={row.scenario_id}>
                  {row.hazard} — score {row.score.toFixed(3)}
                </li>
              ))}
            </ol>

            <h3>Most frustrated (model can’t match the ordering)</h3>
            <ol>
              {result.frustration_list.map((row) => (
                <li key={row.scenario_id}>
                  {row.hazard} — frustration {row.frustration.toFixed(3)}
                </li>
              ))}
            </ol>
          </div>
        )}

        <div className="buttonRow">
          <button className="nextButton" type="button" onClick={onStartNew}>
            Start New Reordering
          </button>
        </div>
      </section>
    </main>
  )
}

export default function App() {
  const [page, setPage] = useState('reorder')
  const [apiColumns, setApiColumns] = useState([])
  const [items, setItems] = useState([])
  const [dragIndex, setDragIndex] = useState(null)
  const [status, setStatus] = useState('Loading scenarios...')
  const [result, setResult] = useState(null)

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

  // On "Done" we send the ordering (top row = most critical) to the backend,
  // which re-fits the studied parameter and returns the re-ranked + frustration
  // lists plus the fitted encoding.
  const handleDone = async () => {
    if (items.length === 0) {
      setStatus('No rows to order.')
      return null
    }

    const orderedIds = items.map((item) => item.row.scenario_id)
    setStatus('Running model re-fit...')
    try {
      const response = await fetch(`${API_URL}/refit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ordered_ids: orderedIds })
      })
      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
      setStatus(
        `Re-fit done - ${data.n_orderings} orderings, ${data.n_pairs} constraints, ` +
        `train acc ${data.train_accuracy.toFixed(3)}`
      )
      return data
    } catch (error) {
      setStatus(`Re-fit failed: ${error.message}`)
      return null
    }
  }

  const handleNextPage = async () => {
    const data = await handleDone()
    if (data) {
      setPage('summary')
    }
  }

  const allHeaders = apiColumns.length > 0 ? apiColumns : Object.keys(items[0]?.row ?? {})
  const visibleHeaders = allHeaders.filter(h => VISIBLE_COLUMNS.includes(h.toLowerCase()))

  if (page === 'summary') {
    return (
      <SummaryPage
        items={items}
        visibleHeaders={visibleHeaders}
        result={result}
        status={status}
        onBack={() => setPage('reorder')}
        onStartNew={() => {
          setResult(null)
          fetchScenarios()
          setPage('reorder')
        }}
      />
    )
  }

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
          <button className="nextButton" type="button" onClick={handleNextPage}>
            Done (Re-fit Model)
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

      </section>
    </main>
  )
}
