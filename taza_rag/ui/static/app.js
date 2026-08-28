const EXAMPLES = [
  { intent: "entity", q: "SoftBank Group" },
  { intent: "topical", q: "private credit market trends" },
  { intent: "executive", q: "Jerome Powell" },
  { intent: "geographic", q: "Brazil deforestation trends" },
  { intent: "industry", q: "pharmaceutical patent cliff" },
  { intent: "event", q: "latest OPEC+ production decision" },
  { intent: "known item", q: "IMF World Economic Outlook latest projections" },
  { intent: "risk", q: "sanctions compliance risk for exporters" },
  { intent: "competitive", q: "competition between Airbus and Boeing on aircraft orders" },
  { intent: "brand", q: "Boeing reputation after safety incidents" },
];

const RESEARCH_EXAMPLES = [
  { intent: "exposure", q: "How exposed is SoftBank Group to its AI bets, and what do its own numbers say?" },
  { intent: "cost", q: "What is Deutsche Bank restructuring, and what has it cost so far?" },
  { intent: "compare", q: "Compare Airbus and Boeing on aircraft orders and delivery performance." },
  { intent: "risk", q: "Why is private credit under scrutiny, and what are the specific risks?" },
];

const RAIL = {
  retrieve: [
    ["plan", "Plan"],
    ["variants", "Variants"],
    ["spend", "Spend"],
    ["pack", "Options"],
    ["answer", "Answer"],
  ],
  research: [
    ["plan", "Plan"],
    ["steps", "Steps"],
    ["spend", "Spend"],
    ["pack", "Evidence"],
    ["answer", "Answer"],
  ],
};

const state = { legend: { scores: [], tiers: [] }, run: null, openai: false, mode: "retrieve" };

const $ = (id) => document.getElementById(id);

async function boot() {
  paintExamples();
  $("examples").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    $("query").value = btn.dataset.q;
    $("query").focus();
  });
  $("modes").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (btn) setMode(btn.dataset.mode);
  });
  setMode("retrieve");
  renderRail("plan");
  try {
    const health = await get("/api/health");
    document.querySelectorAll("#health dd").forEach((dd) => {
      const on = health[dd.dataset.k];
      dd.textContent = on ? "configured" : "missing";
      dd.className = on ? "on" : "off";
    });
    state.openai = !!health.openai;
    syncButtons(false);
    state.legend = await get("/api/legend");
  } catch (err) {
    showStatus(String(err), true);
  }
}

function paintExamples() {
  const list = state.mode === "research" ? RESEARCH_EXAMPLES : EXAMPLES;
  $("examples").innerHTML =
    `<span class="lead">Try</span>` +
    list
      .map(
        (ex) =>
          `<span class="ex"><i>${escapeHtml(ex.intent)}</i>` +
          `<button type="button" data-q="${escapeAttr(ex.q)}">${escapeHtml(ex.q)}</button></span>`
      )
      .join("");
}

function setMode(mode) {
  state.mode = mode === "research" ? "research" : "retrieve";
  state.run = null;
  document.body.className = `mode-${state.mode}`;
  document.querySelectorAll("#modes button").forEach((b) => {
    b.classList.toggle("on", b.dataset.mode === state.mode);
  });
  const research = state.mode === "research";
  $("query-label").textContent = "Task";
  $("topk-label").textContent = research ? "Per query" : "Pack";
  $("top-k").value = research ? 6 : 10;
  $("query").placeholder = research
    ? RESEARCH_EXAMPLES[0].q
    : "How exposed is SoftBank Group to its AI bets?";
  $("pack-title").textContent = research ? "Evidence pack" : "Content options";
  paintExamples();
  ["usage-panel", "plan-panel", "funnel-panel", "pack-panel", "answer-panel", "steps-panel",
   "rounds-panel", "ledger-panel", "conflicts-panel"].forEach((id) => hide($(id)));
  showStatus("");
  syncButtons(false);
  renderRail("plan");
}

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.mode === "research") await research();
  else await retrieve();
});
$("answer-btn").addEventListener("click", () => answer());
$("research-btn").addEventListener("click", () => research());

async function retrieve() {
  const query = $("query").value.trim();
  const top_k = Number($("top-k").value || 10);
  const raw = $("raw").checked;
  if (!query) return;
  busy(true);
  hide($("answer-panel"));
  try {
    showStatus("Planning query…");
    const plan = await post("/api/plan", { query });
    paintPlan(plan);
    renderRail("funnel");
    showStatus(raw ? "Single Factiva call…" : "Retrieving in parallel from Factiva…");
    const run = await post("/api/retrieve", { query, top_k, raw });
    state.run = run;
    paintUsage(run.usage);
    paintRun(run);
    showStatus("");
  } catch (err) {
    showStatus(err.message || String(err), true);
  } finally {
    busy(false);
  }
}

async function answer() {
  const query = $("query").value.trim();
  const top_k = Number($("top-k").value || 10);
  const raw = $("raw").checked;
  if (!query) return;
  busy(true);
  try {
    if (!state.run) {
      showStatus("Planning query…");
      paintPlan(await post("/api/plan", { query }));
    }
    showStatus("Retrieving, extracting facts, composing, verifying…");
    renderRail("answer");
    const payload = await post("/api/answer", { query, top_k, raw });
    if (!state.run) {
      paintPlan(await post("/api/plan", { query }));
      paintHits(payload.hits || []);
    }
    paintUsage(payload.usage);
    paintAnswer(payload);
    showStatus("");
  } catch (err) {
    showStatus(err.message || String(err), true);
  } finally {
    busy(false);
  }
}

async function research() {
  const query = $("query").value.trim();
  if (!query) return;
  busy(true);
  try {
    showStatus("Planning the research…");
    paintPlan(await post("/api/plan", { query }));
    showStatus("Searching in parallel, judging coverage, refining what is missing…");
    renderRail("rounds");
    const run = await post("/api/research", {
      query,
      top_k: Number($("top-k").value || 6),
      max_rounds: Number($("max-rounds").value || 3),
      max_chunks: Number($("max-chunks").value || 40),
      purchase_gate: $("purchase-gate").checked,
    });
    state.run = run;
    paintUsage(run.usage);
    paintResearch(run);
    showStatus("");
  } catch (err) {
    showStatus(err.message || String(err), true);
  } finally {
    busy(false);
  }
}

function paintResearch(run) {
  // The query-plan panel was already painted from /api/plan, which is the only place the
  // date window comes from. Repainting it here would blank that, and would duplicate the
  // steps that the Research plan panel owns with more detail.
  const plan = run.plan || {};

  const gapSet = new Set((run.gaps || []).map((g) => `${g.sub_question_id}::${g.aspect}`));
  show($("steps-panel"));
  $("steps-hint").textContent =
    `${plan.method || "?"} plan · coverage ${fmt(run.coverage)} · stopped on ${run.stop_reason}`;
  $("steps").innerHTML = (plan.sub_questions || [])
    .map((sub) => {
      const cov = (run.sub_coverage || {})[sub.id];
      const aspects = (sub.aspects || [])
        .map((a) => {
          const missing = gapSet.has(`${sub.id}::${a}`);
          return `<span class="aspect ${missing ? "gap" : "met"}">${escapeHtml(a)}</span>`;
        })
        .join("");
      return `<li class="step">
        <span class="sid">${escapeHtml(sub.id)}</span>
        <div><h3>${escapeHtml(sub.question)}</h3>${aspects || '<span class="aspect">no aspects</span>'}</div>
        <span class="cov">${cov === undefined ? "—" : fmt(cov)}</span>
      </li>`;
    })
    .join("");

  show($("rounds-panel"));
  $("rounds-hint").textContent = "each round shows what it cost and what it bought";
  $("rounds").innerHTML = (run.rounds || [])
    .map((r) => {
      const failed = (r.failed_queries || []).length
        ? `<span class="fail"> · ${r.failed_queries.length} failed upstream</span>`
        : "";
      const queries = (r.queries || []).map((q) => escapeHtml(q)).join("<br>") || "<em>none</em>";
      return `<li class="round">
        <span class="rid">round ${r.index}</span>
        <div>
          <p class="queries">${queries}</p>
          <p class="nums">${r.chunks_returned} offered · ${r.new_chunks} bought ·
            +${r.new_findings} facts · coverage ${fmt(r.coverage)}
            (${r.coverage_delta >= 0 ? "+" : ""}${fmt(r.coverage_delta)}) ·
            ${r.latency_ms} ms${failed}</p>
        </div>
      </li>`;
    })
    .join("");

  const led = run.ledger || {};
  if ((led.decisions || []).length) {
    show($("ledger-panel"));
    $("ledger-hint").textContent =
      `${led.charged} bought of ${led.offered} offered · ${led.already_held} already held · ` +
      `${led.rejected} refused · scored on headline and lead only`;
    $("refusals").innerHTML = Object.entries(led.rejection_reasons || {})
      .sort((a, b) => b[1] - a[1])
      .map(([reason, n]) => `<li><b>×${n}</b> refused — ${escapeHtml(reason)}</li>`)
      .join("");
    $("ledger").innerHTML = (led.decisions || [])
      .map(
        (d) => `<li>
        <span class="verdict ${d.admitted ? "buy" : "pass"}">${d.admitted ? "buy" : "pass"}</span>
        <span class="val">${d.value.toFixed(2)}</span>
        <span><span class="what">${escapeHtml(d.title || "")}</span>
          <span class="why">— ${escapeHtml(d.source || "")} · ${escapeHtml(d.reason || "")}</span></span>
      </li>`
      )
      .join("");
  } else {
    hide($("ledger-panel"));
  }

  const disagreements = (run.conflicts || []).filter((c) => c.kind === "disagreement");
  if (disagreements.length) {
    show($("conflicts-panel"));
    $("conflicts").innerHTML = disagreements
      .map((c) => {
        const side = (f, lead) =>
          `<p class="side ${lead ? "lead" : ""}"><b>[${escapeHtml(f.label)}]</b> ${escapeHtml(f.text)}` +
          (lead ? " <em>(leads)</em>" : "") +
          `</p>`;
        return `<li>
          ${side(c.left, c.preferred_label === c.left.label)}
          ${side(c.right, c.preferred_label === c.right.label)}
          <p class="why">${escapeHtml(c.reason)}</p>
        </li>`;
      })
      .join("");
  } else {
    hide($("conflicts-panel"));
  }

  paintHits(run.evidence || []);
  $("pack-hint").textContent =
    `${run.cost.unique_chunks} passages bought of ${run.cost.chunks_returned} offered · ` +
    `${run.cost.evidence_tokens} tokens · ${run.latency_ms.total} ms total`;

  paintAnswer(run);

  if ((run.gaps || []).length) {
    show($("gaps"));
    $("gaps").innerHTML = run.gaps
      .map((g) => `<li>${escapeHtml(g.aspect)} <span>(${escapeHtml(g.sub_question_id)})</span></li>`)
      .join("");
  } else {
    hide($("gaps"));
  }
  renderRail("answer");
}

function fmt(n) {
  return Number(n || 0).toFixed(2);
}

function paintUsage(u) {
  if (!u) {
    hide($("usage-panel"));
    return;
  }
  show($("usage-panel"));
  $("usage").innerHTML = [
    metric(u.offered, "Offered"),
    metric(u.bought, "Bought"),
    metric(u.refused, "Refused"),
    metric(u.cited, "Cited"),
  ].join("");
  const bits = [];
  if (u.budget != null) bits.push(`budget ${u.budget}`);
  if (u.retrieval_calls) bits.push(`${u.retrieval_calls} retrievals`);
  if (u.llm_calls) bits.push(`${u.llm_calls} model calls`);
  $("usage-hint").textContent = bits.length
    ? bits.join(" · ")
    : "what this call offered, bought, and cited";
}

function paintPlan(plan) {
  show($("plan-panel"));
  const same = plan.normalized !== plan.query;
  $("plan-lede").textContent = same
    ? `Normalized “${plan.query}” → “${plan.normalized}”. Intent ${label(plan.intent)}.`
    : `Intent ${label(plan.intent)}. Window ${plan.days_range}.`;
  $("plan-facts").innerHTML = [
    kv("Intent", label(plan.intent)),
    kv("Window", plan.days_range),
    kv("Entities", chips(plan.entities)),
    kv("Topics", chips(plan.topics)),
  ].join("");
  $("variants").innerHTML = (plan.variants || [])
    .map((v, i) => `<li><span class="n">v${i + 1}</span><span>${escapeHtml(v)}</span></li>`)
    .join("");
  // In research mode the sub-questions are the real plan, shown in their own panel with
  // coverage; the heuristic expansion variants would only be noise beside them.
  $("variants").hidden = state.mode === "research";
  renderRail("variants");
}

function paintRun(run) {
  paintPlan({
    query: run.query,
    normalized: run.query,
    intent: run.intent,
    days_range: run.days_range || "",
    entities: run.entities,
    topics: run.topics,
    variants: run.variants,
  });
  if (run.failed_variants?.length) {
    $("variants").insertAdjacentHTML(
      "beforeend",
      run.failed_variants
        .map((v) => `<li class="fail"><span class="n">×</span><span>failed: ${escapeHtml(v)}</span></li>`)
        .join("")
    );
  }
  show($("funnel-panel"));
  const lat = run.latency_ms || {};
  $("funnel").innerHTML = [
    metric(run.variants?.length || 0, "Variants"),
    metric(run.candidates, "Articles"),
    metric(run.passages, "Passages"),
    metric(run.hits?.length || 0, "Pack"),
  ].join("");
  show($("pack-panel"));
  $("pack-hint").textContent = `${run.config} · ${lat.total ?? "—"} ms total`;
  $("hits").innerHTML = (run.hits || []).map(hitCard).join("");
  $("hits").onclick = onHitClick;
  renderRail("pack", lat);
}

function paintHits(hits) {
  show($("pack-panel"));
  $("pack-hint").textContent = "Evidence used to write the answer";
  $("hits").innerHTML = hits.map(hitCard).join("");
  $("hits").onclick = onHitClick;
}

function hitCard(h) {
  const meters = (state.legend.scores || [])
    .map((spec) => meter(spec, (h.scores || {})[spec.key]))
    .join("");
  const p = h.passage || { index: 1, of: 1 };
  const snippet = (h.text || "").slice(0, 280).replace(/\s+/g, " ");
  return `<li class="hit" id="hit-${h.label}" data-label="${h.label}">
    <div class="hit-top">
      <span class="rank">${String(h.rank).padStart(2, "0")}</span>
      <h3>${escapeHtml(h.title || "")}</h3>
      <span class="composite">${h.score.toFixed(3)}</span>
    </div>
    <p class="meta">
      ${escapeHtml(h.source || "")} · ${escapeHtml(h.published_at || "n/a")} ·
      ${escapeHtml(h.doc_id || "")} · ${escapeHtml(h.kind || "article")} ·
      passage ${p.index}/${p.of}
      <span class="tier" title="${escapeAttr(h.tier_help || "")}">${escapeHtml(h.tier_label || "")}</span>
    </p>
    <div class="scores">${meters}</div>
    <details class="excerpt">
      <summary>Passage</summary>
      <p>${escapeHtml(h.text || snippet)}</p>
    </details>
  </li>`;
}

function meter(spec, value) {
  const n = Number(value || 0);
  let pct = 0;
  let tick = "";
  if (spec.kind === "factor") {
    pct = Math.max(0, Math.min(1, (n - 0.9) / 0.22)) * 100;
    tick = `<em style="left:45%" title="neutral 1.00"></em>`;
  } else if (spec.kind === "cost") {
    pct = Math.max(0, Math.min(1, n / 1.1)) * 100;
  } else {
    pct = Math.max(0, Math.min(1, n)) * 100;
  }
  return `<div class="score" title="${escapeAttr(spec.help)}">
    <label>${escapeHtml(spec.label)}</label>
    <div class="bar">${tick}<i style="width:${pct}%"></i></div>
    <output>${n.toFixed(2)}</output>
  </div>`;
}

function paintAnswer(payload) {
  show($("answer-panel"));
  $("answer-config").textContent = `${payload.config}${payload.abstained ? " · abstained" : ""}`;
  const v = payload.verification;
  if (v) {
    show($("verify"));
    const cls = v.resolved ? "ok" : "bad";
    $("verify").innerHTML = `
      <span class="${cls}">${v.resolved ? "Grounding resolved" : "Unresolved claims"}</span>
      <span>Repairs ${v.repairs_applied}</span>
      <span>Flags ${v.initial_problems} → ${v.final_problems}</span>`;
  } else {
    hide($("verify"));
  }
  $("answer").innerHTML = citeHtml(payload.answer || "");
  $("cites").innerHTML = (payload.citations || [])
    .map(
      (c) =>
        `<li>${c.label ? `<b>[${escapeHtml(c.label)}]</b> ` : ""}` +
        `<b>${escapeHtml(c.source || "")}</b> — ${escapeHtml(c.title || "")}
         <span>(${escapeHtml(c.published_at || "n/a")} · ${escapeHtml(c.doc_id || "")})</span></li>`
    )
    .join("");
  $("answer").onclick = (e) => {
    const btn = e.target.closest(".cite");
    if (!btn) return;
    const el = document.getElementById(`hit-${btn.dataset.label}`);
    if (!el) return;
    selectHit(el);
    el.scrollIntoView({ behavior: "smooth", block: "center" });
  };
  renderRail("answer");
}

function citeHtml(text) {
  return escapeHtml(text).replace(/\[(c\d+)\]/gi, (_, lab) => {
    const label = lab.toLowerCase();
    return `<button type="button" class="cite" data-label="${label}">[${label}]</button>`;
  });
}

function onHitClick(e) {
  const hit = e.target.closest(".hit");
  if (!hit) return;
  selectHit(hit);
}

function selectHit(el) {
  document.querySelectorAll(".hit").forEach((n) => n.classList.toggle("selected", n === el));
}

function renderRail(active, latency) {
  $("rail").innerHTML = (RAIL[state.mode] || RAIL.retrieve).map(([id, label]) => {
    const cls = id === active ? "active" : "";
    const ms = latency && id === "spend" && latency.factiva_multi != null
      ? `<span class="ms">${latency.factiva_multi} ms Factiva</span>`
      : latency && id === "pack" && latency.rank != null
        ? `<span class="ms">${latency.rank} ms rank</span>`
        : "";
    return `<li class="${cls}"><span class="mark">${id === active ? "→" : "·"}</span>
      <span>${label}${ms}</span></li>`;
  }).join("");
}

function kv(k, v) {
  return `<div><dt>${escapeHtml(k)}</dt><dd>${v}</dd></div>`;
}
function chips(xs) {
  if (!xs || !xs.length) return "—";
  return xs.map((x) => `<span class="chip">${escapeHtml(x)}</span>`).join("");
}
function metric(n, label) {
  return `<li><b>${n}</b><span>${label}</span></li>`;
}
function label(intent) {
  return (intent || "").replaceAll("_", " ");
}
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }
function busy(on) {
  syncButtons(on);
}
function syncButtons(on) {
  $("retrieve-btn").disabled = on;
  $("answer-btn").disabled = on || !state.openai;
  $("research-btn").disabled = on || !state.openai;
  const missing = "OPENAI_API_KEY is not set";
  $("answer-btn").title = state.openai ? "" : missing;
  $("research-btn").title = state.openai ? "" : missing;
}
function showStatus(msg, err) {
  const el = $("status");
  if (!msg) { hide(el); el.textContent = ""; return; }
  el.textContent = msg;
  el.className = "status" + (err ? " err" : "");
  show(el);
}

async function get(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}
async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}
function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
function escapeAttr(s) {
  return escapeHtml(s).replaceAll("'", "&#39;");
}

boot();
