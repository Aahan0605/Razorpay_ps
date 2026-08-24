import { useEffect, useState } from 'react'

const BANDS = {
  auto_block: { label: 'Auto-block', chip: 'bg-red-50 text-red-700 ring-red-200' },
  hold_for_review: { label: 'Hold for review', chip: 'bg-amber-50 text-amber-800 ring-amber-200' },
  auto_clear: { label: 'Auto-clear', chip: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
}

const inr = n => '₹' + n.toLocaleString('en-IN', { maximumFractionDigits: 0 })
const pct = n => (n * 100).toFixed(1) + '%'

function Stat({ label, value, sub }) {
  return (
    <div className="flex-1 min-w-32 px-5 py-4">
      <div className="text-[11px] font-medium uppercase tracking-wider text-navy/50">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-navy/50">{sub}</div>}
    </div>
  )
}

function Band({ band }) {
  const b = BANDS[band]
  return (
    <span className={`inline-flex rounded-[4px] px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${b.chip}`}>
      {b.label}
    </span>
  )
}

function ScoreBar({ score }) {
  const tone = score >= 0.85 ? 'bg-red-500' : score >= 0.3 ? 'bg-amber-500' : 'bg-emerald-500'
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 rounded-full bg-line">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.max(score * 100, 2)}%` }} />
      </div>
      <span className="w-11 text-right text-xs tabular-nums text-navy/70">{score.toFixed(3)}</span>
    </div>
  )
}

function Verdict({ txn, onVerify, busy, error }) {
  const v = txn.verdict
  if (!v) {
    return (
      <div className="rounded-card border border-line bg-white p-5">
        <div className="text-sm text-navy/60">
          {txn.flagged
            ? 'This transaction was flagged by the classifier. Run the verifier for a plain-language explanation and a recommended action.'
            : 'Scored below the flag threshold. You can still ask the verifier to review it.'}
        </div>
        {error && <div className="mt-3 rounded-[4px] bg-red-50 px-3 py-2 text-xs text-red-700">{error}</div>}
        <button
          onClick={onVerify}
          disabled={busy}
          className="mt-4 rounded-[5px] bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent/90 disabled:opacity-50"
        >
          {busy ? 'Verifying…' : 'Run LLM verifier'}
        </button>
      </div>
    )
  }
  return (
    <div className="rounded-card border border-line bg-white">
      <div className="flex items-center justify-between border-b border-line px-5 py-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-navy/50">Verifier recommendation</span>
        <Band band={v.recommended_action} />
      </div>
      <div className="space-y-4 px-5 py-4">
        <p className="text-sm leading-relaxed">{v.explanation}</p>
        <div>
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-navy/50">Key signals</div>
          <ul className="space-y-1.5">
            {v.key_signals.map((s, i) => (
              <li key={i} className="flex gap-2 text-sm text-navy/80">
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent" />
                {s}
              </li>
            ))}
          </ul>
        </div>
        <div className="border-t border-line pt-3 text-xs text-navy/50">
          Confidence: <span className="font-medium text-navy/70">{v.confidence}</span> · {v.model}
        </div>
      </div>
    </div>
  )
}

function Detail({ txn, onVerify, busy, error }) {
  if (!txn) {
    return (
      <div className="rounded-card border border-dashed border-line bg-white/50 p-10 text-center text-sm text-navy/40">
        Select a transaction to inspect it.
      </div>
    )
  }
  // ids stay verbatim; only the human-readable enums get title-cased
  const fields = [
    ['Amount', inr(txn.amount), false],
    ['Method', txn.payment_method, true],
    ['Category', txn.merchant_category.replace(/_/g, ' '), true],
    ['Merchant', txn.merchant_id, false],
    ['Customer', txn.customer_id, false],
    ['Time', new Date(txn.timestamp).toLocaleString('en-IN'), false],
  ]
  return (
    <div className="space-y-4">
      <div className="rounded-card border border-line bg-white">
        <div className="flex items-start justify-between border-b border-line px-5 py-4">
          <div>
            <div className="font-mono text-sm">{txn.txn_id}</div>
            <div className="mt-1 text-2xl font-semibold">{inr(txn.amount)}</div>
          </div>
          <div className="text-right">
            <Band band={txn.band} />
            <div className="mt-2 flex justify-end"><ScoreBar score={txn.score} /></div>
          </div>
        </div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 px-5 py-4">
          {fields.map(([k, v, cap]) => (
            <div key={k}>
              <dt className="text-[11px] uppercase tracking-wider text-navy/45">{k}</dt>
              <dd className={`mt-0.5 text-sm ${cap ? 'capitalize' : 'font-mono text-[13px]'}`}>{v}</dd>
            </div>
          ))}
        </dl>
      </div>
      <Verdict txn={txn} onVerify={onVerify} busy={busy} error={error} />
    </div>
  )
}

export default function App() {
  const [data, setData] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [selected, setSelected] = useState(null)
  const [flaggedOnly, setFlaggedOnly] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`/api/transactions?limit=80&flagged_only=${flaggedOnly}`).then(r => r.json()).then(setData)
    setSelected(null)
  }, [flaggedOnly])
  useEffect(() => { fetch('/api/metrics').then(r => r.json()).then(setMetrics) }, [])

  async function verify() {
    setBusy(true); setError(null)
    try {
      const r = await fetch(`/api/verify/${selected.txn_id}`, { method: 'POST' })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail || 'Verifier failed')
      const next = { ...selected, verdict: body }
      setSelected(next)
      setData(d => ({ ...d, transactions: d.transactions.map(t => t.txn_id === next.txn_id ? next : t) }))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const cm = metrics?.confusion_matrix

  return (
    <div className="min-h-screen">
      <header className="bg-navy px-8 py-5">
        <div className="mx-auto flex max-w-7xl items-baseline justify-between">
          <div>
            <h1 className="text-lg font-semibold text-white">Fraud Risk Detector</h1>
            <p className="mt-0.5 text-sm text-white/55">ML classifier + LLM verifier · held-out test set</p>
          </div>
          {data && (
            <div className="text-sm text-white/70 tabular-nums">
              {data.flagged_total} flagged of {data.total.toLocaleString()} · threshold {data.threshold}
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-8 py-7">
        {metrics && (
          <div className="mb-7 flex flex-wrap divide-x divide-line rounded-card border border-line bg-white">
            <Stat label="Precision" value={pct(metrics.test.precision)} sub={`${cm.tp} true / ${cm.fp} false`} />
            <Stat label="Recall" value={pct(metrics.test.recall)} sub={`${cm.fn} fraud missed`} />
            <Stat label="F1" value={metrics.test.f1.toFixed(3)} />
            <Stat label="PR-AUC" value={metrics.pr_auc.toFixed(3)} sub={`base rate ${pct(metrics.base_rate)}`} />
            <Stat label="Good traffic held" value={pct(cm.fp / (cm.tn + cm.fp))} sub={`${cm.fp} of ${(cm.tn + cm.fp).toLocaleString()}`} />
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
          <section className="overflow-hidden rounded-card border border-line bg-white">
            <div className="flex items-center justify-between border-b border-line px-5 py-3">
              <h2 className="text-sm font-semibold">Transaction feed</h2>
              <label className="flex cursor-pointer items-center gap-2 text-xs text-navy/60">
                <input type="checkbox" checked={flaggedOnly} onChange={e => setFlaggedOnly(e.target.checked)}
                       className="accent-[#0D94FB]" />
                Flagged only
              </label>
            </div>
            <div className="max-h-[34rem] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-canvas text-[11px] uppercase tracking-wider text-navy/45">
                  <tr>
                    <th className="px-5 py-2 text-left font-medium">Transaction</th>
                    <th className="px-3 py-2 text-right font-medium">Amount</th>
                    <th className="px-3 py-2 text-left font-medium">Risk</th>
                    <th className="px-5 py-2 text-left font-medium">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {data?.transactions.map(t => (
                    <tr key={t.txn_id} onClick={() => { setSelected(t); setError(null) }}
                        className={`cursor-pointer transition hover:bg-canvas ${selected?.txn_id === t.txn_id ? 'bg-accent/5' : ''}`}>
                      <td className="px-5 py-2.5">
                        <div className="font-mono text-xs">{t.txn_id}</div>
                        <div className="text-xs text-navy/45 capitalize">
                          {t.merchant_category.replace(/_/g, ' ')} · {t.payment_method}
                        </div>
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums">{inr(t.amount)}</td>
                      <td className="px-3 py-2.5"><ScoreBar score={t.score} /></td>
                      <td className="px-5 py-2.5"><Band band={t.band} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {data && !data.transactions.length && (
                <div className="px-5 py-10 text-center text-sm text-navy/40">No transactions.</div>
              )}
            </div>
          </section>

          <section><Detail txn={selected} onVerify={verify} busy={busy} error={error} /></section>
        </div>
      </main>
    </div>
  )
}
