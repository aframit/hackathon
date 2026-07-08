import { useEffect, useState } from 'react'
import Papa from 'papaparse'

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

function sampleRandomRows(rows, count) {
  const shuffled = [...rows]
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled.slice(0, Math.min(count, shuffled.length))
}

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
  const [allRows, setAllRows] = useState([])
  const [csvHeaders, setCsvHeaders] = useState([])
  const [items, setItems] = useState([])
  const [dragIndex, setDragIndex] = useState(null)
  const [status, setStatus] = useState('Loading CSV data...')

  const loadRandomSample = (rows) => {
    const sampledRows = sampleRandomRows(rows, SAMPLE_SIZE).map((row, index) => ({
      id: `${Date.now()}-${index}`,
      row
    }))
    setItems(sampledRows)
    setStatus(`Loaded ${sampledRows.length} random rows from risk_summary.csv`)
  }

  useEffect(() => {
    const loadCsvRows = async () => {
      try {
        const response = await fetch('/risk_summary.csv')
        if (!response.ok) {
          throw new Error('Could not read risk_summary.csv from public folder.')
        }

        const csvText = await response.text()
        const parsed = Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true
        })

        if (parsed.errors.length > 0) {
          throw new Error(parsed.errors[0].message)
        }

        const headers = parsed.meta.fields ?? []
        setCsvHeaders(headers)

        const validRows = parsed.data.filter((row) =>
          Object.values(row).some((value) => String(value ?? '').trim() !== '')
        )

        setAllRows(validRows)
        loadRandomSample(validRows)
      } catch (error) {
        setStatus(`Failed to load CSV: ${error.message}`)
      }
    }

    loadCsvRows()
  }, [])

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
    if (allRows.length === 0) {
      setStatus('No CSV rows are loaded yet.')
      return
    }

    loadRandomSample(allRows)
  }

  const handleDone = () => {
    if (items.length === 0) {
      setStatus('No rows to export.')
      return
    }

    const headers = csvHeaders.length > 0 ? csvHeaders : Object.keys(items[0].row)
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

  const visibleHeaders = csvHeaders.length > 0 
    ? csvHeaders.filter(h => VISIBLE_COLUMNS.includes(h.toLowerCase()))
    : Object.keys(items[0]?.row ?? {}).filter(h => VISIBLE_COLUMNS.includes(h.toLowerCase()))

  return (
    <main className="app">
      <section className="card">
        <img src="/logo-innerspace.png" alt="Logo" className="headerLogo" />
        <h1>Order Hazard Scenarios</h1>
        <p className="hint">Showing 10 random rows from the CSV file.</p>
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
