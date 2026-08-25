const EXAMPLES = [
  "SoftBank Group",
  "Deutche Bank restructuring",
  "EU AI Act compliance",
  "Nvidia chip export restrictions",
];

const RAIL = [
  ["plan", "Plan"],
  ["variants", "Variants"],
  ["funnel", "Funnel"],
  ["pack", "Pack"],
  ["answer", "Answer"],
];

const state = { legend: { scores: [], tiers: [] }, run: null, openai: false };

const $ = (id) => document.getElementById(id);

async function boot() {
  $("examples").innerHTML =
    "Try " +
    EXAMPLES.map((q) => `<button type="button" data-q="${escapeAttr(q)}">${escapeHtml(q)}</button>`).join(" · ");
  $("examples").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    $("query").value = btn.dataset.q;
    $("query").focus();
  });
  renderRail("plan");
  try {
    const health = await get("/api/health");
    document.querySelectorAll("#health dd").forEach((dd) => {
      const on = health[dd.dataset.k];
      dd.textContent = on ? "configured" : "missing";
      dd.className = on ? "on" : "off";
    });
    state.openai = !!health.openai;
    $("answer-btn").disabled = !state.openai;
    if (!state.openai) $("answer-btn").title = "OPENAI_API_KEY is not set";
    state.legend = await get("/api/legend");
  } catch (err) {
    showStatus(String(err), true);
  }
}

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  await retrieve();
});
$("answer-btn").addEventListener("click", () => answer());

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
    paintAnswer(payload);
    showStatus("");
  } catch (err) {
    showStatus(err.message || String(err), true);
  } finally {
    busy(false);
  }
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
        `<li><b>${escapeHtml(c.source || "")}</b> — ${escapeHtml(c.title || "")}
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
  $("rail").innerHTML = RAIL.map(([id, label]) => {
    const cls = id === active ? "active" : "";
    const ms = latency && id === "funnel" && latency.factiva_multi != null
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
  $("retrieve-btn").disabled = on;
  $("answer-btn").disabled = on || !state.openai;
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
