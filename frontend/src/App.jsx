import React, { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, CheckCircle2, CircleStop, Gauge, RefreshCw, ShieldCheck, WalletCards } from "lucide-react";
import { BarChart, Bar, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { analyzeCase, executeCase, getAnalytics, getCases, getDashboard, getPayment } from "./api";

const money = n => `₹${Number(n || 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
const pct = n => `${(Number(n || 0) * 100).toFixed(1)}%`;

function Status({ children }) {
  return <span className={`status status-${String(children).toLowerCase().replaceAll(" ", "-")}`}>{children}</span>;
}

function App() {
  const [page, setPage] = useState("overview");
  const [dashboard, setDashboard] = useState(null);
  const [cases, setCases] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const [d, c, a] = await Promise.all([getDashboard(), getCases(), getAnalytics()]);
      setDashboard(d); setCases(c); setAnalytics(a);
    } finally { setLoading(false); }
  };

  useEffect(() => { refresh(); }, []);

  const openCase = async (caseId) => {
    const data = await getPayment(cases.find(c => c.case_id === caseId)?.payment_id);
    setSelected(data);
    setPage("case");
  };

  const runAnalysis = async () => {
    if (!selected?.case_id) return;
    setLoading(true);
    try {
      const result = await analyzeCase(selected.case_id);
      const fresh = await getPayment(selected.payment_id);
      setSelected(fresh);
      await refresh();
    } finally { setLoading(false); }
  };

  const execute = async () => {
    if (!selected?.case_id) return;
    setLoading(true);
    try {
      await executeCase(selected.case_id);
      const fresh = await getPayment(selected.payment_id);
      setSelected(fresh);
      await refresh();
    } finally { setLoading(false); }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">Recover<span>Z</span></div>
        <div className="brand-sub">Revenue recovery operations</div>

        <nav>
          {[
            ["overview", "Overview"],
            ["cases", "Recovery cases"],
            ["analytics", "Evaluation"],
          ].map(([key, label]) => (
            <button className={page === key ? "nav active" : "nav"} onClick={() => setPage(key)} key={key}>
              {label}
            </button>
          ))}
        </nav>

        <div className="side-note">
          <div className="eyebrow">CONTROL MODEL</div>
          <strong>LLM recommends.</strong>
          <strong>Policy authorizes.</strong>
          <strong>Executor acts.</strong>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="eyebrow">MERCHANT OPERATIONS</div>
            <h1>{page === "case" ? "Recovery case" : page === "analytics" ? "Evaluation" : page === "cases" ? "Recovery cases" : "Revenue recovery"}</h1>
          </div>
          <button className="refresh" onClick={refresh}><RefreshCw size={15}/> {loading ? "Working..." : "Refresh"}</button>
        </header>

        {page === "overview" && <Overview dashboard={dashboard} cases={cases} openCase={openCase} goCases={() => setPage("cases")} />}
        {page === "cases" && <Cases cases={cases} openCase={openCase} />}
        {page === "case" && selected && <CaseDetail selected={selected} runAnalysis={runAnalysis} execute={execute} loading={loading} />}
        {page === "analytics" && <Analytics analytics={analytics} />}
      </main>
    </div>
  );
}

function Overview({ dashboard, cases, openCase, goCases }) {
  if (!dashboard) return <div className="loading">Loading RecoverZ...</div>;
  const queue = cases.slice(0, 6);
  return <>
    <section className="metric-grid">
      <Metric label="Revenue at risk" value={money(dashboard.revenue_at_risk)} icon={<WalletCards size={18}/>} />
      <Metric label="Revenue recovered" value={money(dashboard.revenue_recovered)} icon={<CheckCircle2 size={18}/>} />
      <Metric label="Recovery rate" value={pct(dashboard.recovery_rate)} icon={<Gauge size={18}/>} />
      <Metric label="Escalations" value={dashboard.escalations} icon={<ShieldCheck size={18}/>} />
    </section>

    <section className="section-grid">
      <div className="panel">
        <div className="panel-head"><div><div className="eyebrow">LIVE QUEUE</div><h2>Recovery cases</h2></div><button className="text-btn" onClick={goCases}>View all <ArrowRight size={14}/></button></div>
        <CaseTable cases={queue} openCase={openCase}/>
      </div>

      <div className="panel funnel">
        <div className="eyebrow">RECOVERY PIPELINE</div>
        <h2>From failure to verified outcome</h2>
        {[
          ["Failed payments", dashboard.cases],
          ["Cases opened", cases.length],
          ["Actions executed", dashboard.executed_actions],
          ["Successful recoveries", dashboard.successful_recoveries],
        ].map(([label, value], i) => (
          <div className="funnel-row" key={label}>
            <div><span>{String(i + 1).padStart(2, "0")}</span>{label}</div><strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  </>;
}

function Metric({ label, value, icon }) {
  return <div className="metric">
    <div className="metric-icon">{icon}</div>
    <div className="eyebrow">{label}</div>
    <div className="metric-value">{value}</div>
  </div>;
}

function Cases({ cases, openCase }) {
  return <div className="panel full">
    <div className="panel-head"><div><div className="eyebrow">OPERATIONS QUEUE</div><h2>Failed payment cases</h2></div><span className="demo-tag">DEMO DATA</span></div>
    <CaseTable cases={cases} openCase={openCase}/>
  </div>;
}

function CaseTable({ cases, openCase }) {
  return <div className="table-wrap">
    <table>
      <thead><tr><th>Payment</th><th>Amount</th><th>Failure</th><th>Recovery</th><th>Action</th><th>Status</th><th></th></tr></thead>
      <tbody>
        {cases.map(c => <tr key={c.case_id}>
          <td><button className="link-btn" onClick={() => openCase(c.case_id)}>{c.payment_id}</button><small>{c.customer_id}</small></td>
          <td>{money(c.amount)}</td>
          <td><span className="muted">{c.failure_code.replaceAll("_", " ")}</span></td>
          <td>{c.recovery_probability == null ? "—" : pct(c.recovery_probability)}</td>
          <td>{c.final_action || "Not analyzed"}</td>
          <td><Status>{c.outcome || c.status}</Status></td>
          <td><ArrowRight size={14} className="muted"/></td>
        </tr>)}
      </tbody>
    </table>
  </div>;
}

function CaseDetail({ selected, runAnalysis, execute, loading }) {
  const analyzed = selected.recovery_probability != null;
  const reasons = selected.audit?.find(a => a.event_type === "ANALYZED")?.metadata_json;
  let policyReasons = [];
  try { policyReasons = JSON.parse(reasons || "{}").policy_reasons || []; } catch {}

  return <div className="case-layout">
    <div>
      <div className="case-title">
        <div><div className="eyebrow">{selected.payment_id}</div><h2>{money(selected.amount)}</h2></div>
        <Status>{selected.outcome || selected.status}</Status>
      </div>

      <div className="panel">
        <div className="eyebrow">PAYMENT CONTEXT</div>
        <div className="detail-grid">
          <Detail label="Customer" value={selected.customer_id}/>
          <Detail label="Method" value={selected.payment_method}/>
          <Detail label="Failure" value={selected.failure_code.replaceAll("_", " ")}/>
          <Detail label="Attempts" value={selected.attempt_count}/>
          <Detail label="Previous success" value={pct(selected.previous_success_rate)}/>
          <Detail label="Customer activity" value={pct(selected.customer_activity_score)}/>
        </div>
      </div>

      <div className="panel decision-panel">
        <div className="panel-head">
          <div><div className="eyebrow">RECOVERZ DECISION</div><h2>{analyzed ? selected.final_action : "Analysis required"}</h2></div>
          {analyzed && <span className={`decision-mark ${selected.policy_result === "APPROVED" ? "approved" : "blocked"}`}>
            {selected.policy_result === "APPROVED" ? <CheckCircle2 size={18}/> : <AlertTriangle size={18}/>}
            {selected.policy_result}
          </span>}
        </div>

        {!analyzed ? <div className="empty-action"><p>Run analysis to calculate recovery probability, diagnosis, expected value and policy checks.</p><button className="primary" onClick={runAnalysis} disabled={loading}>Run RecoverZ analysis</button></div> :
        <>
          <div className="decision-grid">
            <div><span>Recovery probability</span><strong>{pct(selected.recovery_probability)}</strong></div>
            <div><span>Expected recovery value</span><strong>{money(selected.expected_recovery_value)}</strong></div>
            <div><span>AI recommendation</span><strong>{selected.recommended_action}</strong></div>
            <div><span>Confidence</span><strong>{pct(selected.confidence)}</strong></div>
          </div>
          <div className="diagnosis">
            <div className="eyebrow">DIAGNOSIS</div>
            <p>{selected.diagnosis}</p>
          </div>
          <div className="policy-list">
            <div className="eyebrow">POLICY CHECKS</div>
            {policyReasons.map((r, i) => <div className="policy-row" key={i}><CheckCircle2 size={15}/>{r}</div>)}
            {selected.policy_result === "BLOCKED" && <div className="policy-row blocked"><CircleStop size={15}/>The recommendation was blocked; the final action is controlled by policy.</div>}
          </div>
          <button className="primary" onClick={execute} disabled={loading || ["STOP","ESCALATE"].includes(selected.final_action)}>
            {["STOP","ESCALATE"].includes(selected.final_action) ? "No money movement — policy outcome" : "Execute approved recovery"}
          </button>
          {selected.outcome && <div className="result-box"><strong>{selected.outcome}</strong><span>{selected.recovered_amount > 0 ? `${money(selected.recovered_amount)} recovered` : "No money movement"}</span></div>}
        </>}
      </div>
    </div>

    <aside className="panel audit">
      <div className="eyebrow">AUDIT TRAIL</div>
      <h2>Decision timeline</h2>
      {(selected.audit || []).map(a => <div className="audit-item" key={a.id}><div className="dot"></div><div><strong>{a.event_type.replaceAll("_", " ")}</strong><p>{a.message}</p><small>{a.created_at}</small></div></div>)}
    </aside>
  </div>;
}

function Detail({ label, value }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function Analytics({ analytics }) {
  if (!analytics) return <div className="loading">Loading evaluation...</div>;
  const e = analytics.evaluation || {};
  const n = e.naive || {};
  const r = e.recoverz || {};
  const m = e.model_metrics || {};

  return <div className="analytics-grid">
    <div className="panel full">
      <div className="eyebrow">BUSINESS OUTCOME</div>
      <h2>RecoverZ vs a bounded naive retry</h2>
      <p className="muted benchmark-copy">
        Both strategies are evaluated on the same safe autonomous universe. RecoverZ adds a probability gate before intervening.
      </p>
      <div className="comparison">
        <Compare label="No intervention" value={e.no_intervention_recovered} />
        <Compare label="Naive retry baseline" value={e.naive_recovered} />
        <Compare label="RecoverZ" value={e.recoverz_recovered} strong />
      </div>
    </div>

    <div className="panel">
      <div className="eyebrow">EFFICIENCY</div>
      <h2>Same risk boundary, fewer interventions</h2>
      <div className="detail-grid">
        <Detail label="Naive interventions" value={n.interventions ?? "—"}/>
        <Detail label="RecoverZ interventions" value={r.interventions ?? "—"}/>
        <Detail label="Intervention reduction" value={pct(e.intervention_reduction_vs_naive)}/>
        <Detail label="Naive recovery rate" value={pct(n.recovery_rate)}/>
        <Detail label="RecoverZ recovery rate" value={pct(r.recovery_rate)}/>
        <Detail label="Revenue retained vs naive" value={pct(e.revenue_retained_vs_naive)}/>
        <Detail label="Naive ₹ / intervention" value={money(n.recovered_value_per_intervention)}/>
        <Detail label="RecoverZ ₹ / intervention" value={money(r.recovered_value_per_intervention)}/>
        <Detail label="False-intervention reduction" value={pct(e.false_intervention_reduction_vs_naive)}/>
      </div>
    </div>

    <div className="panel">
      <div className="eyebrow">MODEL QUALITY</div>
      <h2>Recovery probability model</h2>
      <div className="detail-grid">
        <Detail label="ROC AUC" value={m.roc_auc ?? "—"}/>
        <Detail label="Precision" value={pct(m.precision)}/>
        <Detail label="Recall" value={pct(m.recall)}/>
        <Detail label="F1" value={m.f1 ?? "—"}/>
        <Detail label="Accuracy" value={pct(m.accuracy)}/>
        <Detail label="Training rows" value={m.rows ?? "—"}/>
      </div>
      <div className="benchmark-box">
        <strong>Policy gates</strong>
        <span>≤ {money(e.policy?.max_unattended_amount)} unattended amount</span>
        <span>&lt; {e.policy?.max_autonomous_attempts} autonomous attempts</span>
        <span>≥ {pct(e.policy?.min_recovery_probability)} probability floor</span>
      </div>
    </div>

    <div className="panel chart-panel">
      <div className="eyebrow">RECOVERY BY FAILURE</div><h2>Where the recovery comes from</h2>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={analytics.by_failure}>
          <CartesianGrid strokeDasharray="3 3" vertical={false}/>
          <XAxis dataKey="failure_code" tick={{fontSize: 11}}/>
          <YAxis tick={{fontSize: 11}}/>
          <Tooltip formatter={(v) => money(v)}/>
          <Bar dataKey="recovered" fill="#181818" radius={[3,3,0,0]}/>
        </BarChart>
      </ResponsiveContainer>
      <p className="footnote">{e.note}</p>
    </div>
  </div>;
}
function Compare({ label, value, strong }) {
  return <div className={strong ? "compare strong" : "compare"}><span>{label}</span><strong>{money(value)}</strong></div>;
}

export default App;
