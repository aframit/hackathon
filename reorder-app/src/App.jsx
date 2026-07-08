import { useCallback, useEffect, useState } from 'react'

const API_URL = 'http://localhost:8000'
const SAMPLE_SIZE = 10

// The parameter groupings the re-fit can target. Value must match the backend CASES.
const CASE_OPTIONS = [
  { value: 'interaction_with_critical_surfaces', label: 'Interaction with critical surfaces' },
  { value: 'distance_to_object', label: 'Distance to object (integer bins)' },
  { value: 'all', label: '🎢 All parameters (everything at once)' }
]

const VISIBLE_COLUMNS = [
  'project',
  'process',
  'hazard scenario',
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

function getFrustrationRowClass(value) {
  if (value > 0.6) {
    return 'frustrationHigh'
  }
  if (value < 0.35) {
    return 'frustrationLow'
  }
  return 'frustrationMedium'
}

function shouldShowConfetti(result) {
  const rows = result?.frustration_list ?? []
  return rows.length > 0 && rows.every((row) => Number(row.frustration) < 0.6)
}

function ConfettiBurst({ active }) {
  if (!active) {
    return null
  }

  const pieces = Array.from({ length: 36 }, (_, index) => {
    const colors = ['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#ef476f', '#f7b801']
    return {
      id: index,
      left: `${Math.random() * 100}%`,
      delay: `${(index % 12) * 0.08}s`,
      duration: `${2.8 + (index % 6) * 0.18}s`,
      rotate: `${Math.random() * 360}deg`,
      color: colors[index % colors.length]
    }
  })

  return (
    <div className="confettiLayer" aria-hidden="true">
      {pieces.map((piece) => (
        <span
          key={piece.id}
          className="confettiPiece"
          style={{
            left: piece.left,
            animationDelay: piece.delay,
            animationDuration: piece.duration,
            backgroundColor: piece.color,
            transform: `rotate(${piece.rotate})`
          }}
        />
      ))}
    </div>
  )
}

function SummaryPage({ items, visibleHeaders, result, status, onBack, onStartNew, showConfetti }) {
  return (
    <main className="app">
      <section className="card">
        <ConfettiBurst active={showConfetti} />
        <img src="/logo-innerspace.png" alt="Logo" className="headerLogo" />
        <h1>Ranking Summary</h1>

        {result && (
          <div className="results">
            <h2>Re-fit results ({result.case})</h2>

            <div className="resultsTopGrid">
              <div className="resultPanel">
                <h3>Re-ranked list (most critical first)</h3>
                <div className="tableWrap resultTableWrap" aria-label="Re-ranked list table">
                  <table className="dataTable resultDataTable">
                    <thead>
                      <tr>
                        <th className="rankHeader">#</th>
                        <th>Hazard Scenario</th>
                        <th>WHC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.fitted_list.map((row, index) => (
                        <tr key={row.scenario_id} className="tableRow">
                          <td className="rankCell">{index + 1}</td>
                          <td className="rowCell">{row.hazard}</td>
                          <td className="rowCell">{Number(row.whc).toFixed(1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="resultPanel">
                <h3>Most frustrated (model can’t match ordering)</h3>
                <div className="tableWrap resultTableWrap" aria-label="Most frustrated table">
                  <table className="dataTable resultDataTable">
                    <thead>
                      <tr>
                        <th className="rankHeader">#</th>
                        <th>Hazard Scenario</th>
                        <th>Frustration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.frustration_list.map((row, index) => (
                        <tr
                          key={row.scenario_id}
                          className={`tableRow ${getFrustrationRowClass(Number(row.frustration))}`}
                        >
                          <td className="rankCell">{index + 1}</td>
                          <td className="rowCell">{row.hazard}</td>
                          <td className="rowCell">{row.frustration.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {Object.entries(result.encodings ?? {}).map(([param, mapping]) => {
              const isDistance = param === 'distance_to_object'
              return (
                <div key={param} className="encodingBlock">
                  <h3>
                    {isDistance
                      ? 'distance_to_object — 10 bin scores'
                      : `Fitted encoding — ${param} (label → new score)`}
                  </h3>
                  <div className="tableWrap resultTableWrap" aria-label={`${param} encoding table`}>
                    <table className="dataTable resultDataTable">
                      <thead>
                        <tr>
                          {!isDistance && <th>Label</th>}
                          <th>New Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(mapping).map(([label, score]) => (
                          <tr key={label} className="tableRow">
                            {!isDistance && <td className="rowCell">{label}</td>}
                            <td className="rowCell">
                              {isDistance ? Math.round(Number(score)) : Number(score).toFixed(3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )
            })}
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
  const [allOrderedBatches, setAllOrderedBatches] = useState([])
  const [dragIndex, setDragIndex] = useState(null)
  const [status, setStatus] = useState('Loading scenarios...')
  const [result, setResult] = useState(null)
  const [showConfetti, setShowConfetti] = useState(false)
  const [selectedCase, setSelectedCase] = useState(CASE_OPTIONS[0].value)

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

  useEffect(() => {
    if (!showConfetti) {
      return
    }
    const timerId = setTimeout(() => setShowConfetti(false), 3600)
    return () => clearTimeout(timerId)
  }, [showConfetti])

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
    if (items.length > 0) {
      setAllOrderedBatches((prev) => [...prev, items])
      setStatus(`Saved batch ${allOrderedBatches.length + 1}. Loading next 10 scenarios...`)
    }
    fetchScenarios()
  }

  // On "Done" we send the ordering (top row = most critical) to the backend,
  // which re-fits the studied parameter and returns the re-ranked + frustration
  // lists plus the fitted encoding.
  const handleDone = async () => {
    const allBatches = [...allOrderedBatches, items].filter((batch) => batch.length > 0)
    if (allBatches.length === 0) {
      setStatus('No rows to order.')
      return null
    }

    const orderedIdBatches = allBatches.map((batch) =>
      batch.map((item) => item.row.scenario_id)
    )

    const totalItems = orderedIdBatches.reduce((sum, batch) => sum + batch.length, 0)
    setStatus(
      `Running model re-fit on ${allBatches.length} batches (${totalItems} rows total)...`
    )
    try {
      const response = await fetch(`${API_URL}/refit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ordered_ids: orderedIdBatches, case: selectedCase })
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
      setShowConfetti(shouldShowConfetti(data))
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
        showConfetti={showConfetti}
        onStartNew={() => {
          setResult(null)
          setShowConfetti(false)
          setAllOrderedBatches([])
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
        <p className="hint">Saved batches: {allOrderedBatches.length}</p>
        <div className="buttonRow">
          <label className="caseSelect">
            Re-fit:{' '}
            <select value={selectedCase} onChange={(event) => setSelectedCase(event.target.value)}>
              {CASE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <button className="refreshButton" type="button" onClick={handleGetNewData}>
            Save Batch & Get New Data
          </button>
          <button className="nextButton" type="button" onClick={handleNextPage}>
            Finalize & Re-fit Model
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
                  <th
                    key={header}
                    className={header.toLowerCase() === 'hazard scenario' ? 'hazardScenarioCol' : ''}
                  >
                    {header}
                  </th>
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
                    <td
                      key={`${item.id}-${header}`}
                      className={`rowCell ${header.toLowerCase() === 'hazard scenario' ? 'hazardScenarioCol' : ''}`}
                    >
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
