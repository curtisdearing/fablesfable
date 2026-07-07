"use strict";
// fablesfable transparency site. Renders entirely from ./data/*.json exported
// by scripts/export_site.py. No framework, no build step.

const $ = (s, r = document) => r.querySelector(s);
const el = (t, cls, html) => { const e = document.createElement(t); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = x => (x == null || isNaN(x)) ? "—" : (x * 100).toFixed(0) + "%";
const pts = x => (x == null || isNaN(x)) ? "—" : ((x >= 0 ? "+" : "") + (x * 100).toFixed(1));

const DATA = {};
async function load(name) {
  try { const r = await fetch(`data/${name}.json`, { cache: "no-store" }); if (!r.ok) throw 0; return await r.json(); }
  catch { return null; }
}

async function boot() {
  const [picks, model, constants, evidence, meta] = await Promise.all(
    ["picks", "model", "constants", "evidence", "meta"].map(load));
  Object.assign(DATA, { picks, model, constants, evidence, meta });
  if (!picks && !model) {
    $("#banner").innerHTML = `<div class="notice warn">Could not load <code>data/*.json</code>.
      Serve this folder over HTTP (e.g. <code>python3 -m http.server</code> inside <code>site/</code>) —
      browsers block <code>fetch</code> of local files opened with <code>file://</code>.
      Then run <code>python3 scripts/export_site.py</code> to (re)generate the data.</div>`;
    return;
  }
  $("#disclaimer").textContent = (meta && meta.disclaimer) || "";
  renderBanner(); renderPicks(); renderHow(); renderModel(); renderConstants(); renderEvidence();
  wireTabs();
}

function wireTabs() {
  $("#tabs").addEventListener("click", e => {
    const b = e.target.closest("button[data-tab]"); if (!b) return;
    document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === b.dataset.tab));
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function renderBanner() {
  const p = DATA.picks || {}, box = $("#banner");
  let h = "";
  if (p.is_sample) h += `<div class="notice sample"><b>Illustrative sample.</b> ${esc(p.sample_note || "")}</div>`;
  if (p.publish === false) h += `<div class="notice warn"><b>NOT PUBLISHED</b> — data freshness gate failed: ${esc((p.publish_reasons || []).join("; "))}. These would persist as <code>blocked</code> and stay out of the record.</div>`;
  box.innerHTML = h;
}

/* ---------------- Picks & why ---------------- */
function tierRank(t) { return { STRONG: 0, PLAYABLE: 1, LEAN: 2, RESEARCH: 3, PASS: 4 }[t] ?? 5; }

function renderPicks() {
  const p = DATA.picks || {}, root = $("#picks");
  const head = el("div");
  head.appendChild(el("h2", null, `Best picks${p.week ? ` — ${p.season} Week ${p.week}` : ""}`));
  head.appendChild(el("p", "lede",
    `The selector runs <em>after</em> every candidate in a game is fully evaluated, then ranks by model-vs-market edge and tiers each pick. ` +
    `Click any pick to expand its full decision chain: projection → adjustments → market comparison → tier. Overs and unders both qualify; synthetic (no-market) lines are labeled RESEARCH and never called best bets.`));
  head.appendChild(legend());
  root.appendChild(head);

  const games = (p.games || []).filter(g => (g.picks || []).length || (g.picks || []).some);
  if (!p.games || !p.games.length) { root.appendChild(el("div", "notice", "No week generated yet. Run the weekly pipeline, then <code>scripts/export_site.py</code>.")); return; }

  p.games.forEach(g => {
    const picks = [...(g.picks || [])].sort((a, b) => tierRank(a.tier) - tierRank(b.tier) || (b.edge || 0) - (a.edge || 0));
    if (!picks.length) return;
    const gc = el("div", "game");
    const gh = el("div", "game-head");
    gh.innerHTML = `<b>${esc(g.matchup)}</b><span class="screened">${esc(g.screened || "")} candidates screened</span>`;
    gc.appendChild(gh);
    picks.forEach(pk => gc.appendChild(pickEl(pk)));
    if (g.notes && g.notes.length) {
      const n = el("div", "gnotes", `<b>Game notes (display only — never scored):</b> ` + g.notes.map(esc).join(" · "));
      gc.appendChild(n);
    }
    root.appendChild(gc);
  });
}

function legend() {
  const l = el("div", "legend");
  l.innerHTML = ["STRONG", "PLAYABLE", "LEAN", "RESEARCH"].map(t =>
    `<span><b class="t-${t}" style="border:none;padding:0">${t}</b></span>`).join("")
    + `<span>— tiers are market-specific &amp; configurable (higher bar for anytime-TD)</span>`;
  return l;
}

function pickEl(pk) {
  const wrap = el("div", "pick");
  const side = pk.market === "anytime_td" ? "YES" : (pk.side || "").toUpperCase();
  const line = pk.line != null ? pk.line + (pk.line_source === "odds_api" ? "" : "†") : "";
  const edgeCls = (pk.edge || 0) >= 0 ? "pos" : "neg";
  const row = el("div", "pick-row");
  row.innerHTML =
    `<span class="tierbadge t-${pk.tier}">${pk.tier}</span>
     <div class="pick-main">
       <div class="pick-title">${esc(pk.player)} <span class="pick-sub">· ${esc(String(pk.market).replace(/_/g, " "))} ${side} ${esc(line)}</span></div>
       <div class="pick-sub">${esc(pk.pos || "")} · ${esc(pk.team || "")} · projection ${esc(pk.mean)}</div>
     </div>
     <div class="pick-nums">
       <div><div class="k">Edge</div><div class="v ${edgeCls}">${pk.edge == null ? "—" : pts(pk.edge)}</div></div>
       <div><div class="k">Model→Mkt</div><div class="v">${pct(pk.model_prob)}→${pct(pk.market_prob)}</div></div>
       <div><div class="k">EV</div><div class="v ${(pk.ev || 0) >= 0 ? "pos" : "neg"}">${pk.ev == null ? "—" : pts(pk.ev) + "%"}</div></div>
     </div>
     <span class="chev">▶</span>`;
  const det = el("div", "pick-detail");
  det.appendChild(el("div", "writeup", esc(pk.writeup || "")));
  const ul = el("ul", "chain");
  (pk.decision_chain || []).forEach(s => {
    const li = el("li");
    li.innerHTML = `<div class="stage">${esc(s.stage)}</div><div class="detail">${esc(s.detail)}</div>`;
    ul.appendChild(li);
  });
  det.appendChild(ul);
  wrap.appendChild(row); wrap.appendChild(det);
  row.addEventListener("click", () => wrap.classList.toggle("open"));
  return wrap;
}

/* ---------------- How it works ---------------- */
function renderHow() {
  const m = DATA.model || {}, root = $("#how");
  root.appendChild(el("h2", null, "How a pick is made"));
  root.appendChild(el("p", "lede",
    "Seven stages, in order. The rule that keeps it honest: <em>selection happens last</em> — the model never decides a bet is good before it has projected and scored every candidate in the game."));
  (m.stages || []).forEach(s => {
    const c = el("div", "card stagecard");
    c.innerHTML = `<div class="num">${s.n}</div><div><h3>${esc(s.name)}</h3><p>${esc(s.what)}</p><div class="where">${esc(s.where)}</div></div>`;
    root.appendChild(c);
  });
  root.appendChild(el("p", "lede", "Two guardrails run through every stage: <em>walk-forward</em> features (a week only ever sees prior weeks — no leakage) and <em>measured constants</em> (see the Constants tab). Line movement is deliberately <em>not</em> a live input — a line drifting our way is after-the-fact CLV feedback, never a reason a currently-available bet is better."));
}

/* ---------------- The model ---------------- */
function renderModel() {
  const m = DATA.model || {}, root = $("#model");
  root.appendChild(el("h2", null, "What the model looks at"));
  root.appendChild(el("p", "lede", `Ranking probability comes from a <em>${esc(m.base_model || "calibrated classifier")}</em>. It reads ${(m.features || []).length} walk-forward features — every one listed below. The deterministic composite score is shown alongside so each pick stays explainable even when the ML layer orders the list.`));

  root.appendChild(el("h2", null, "Composite score weights"));
  const cw = m.composite_weights || {};
  root.appendChild(kvTable(Object.entries(cw).map(([k, v]) => [k, v]), ["Component", "Weight"]));
  if (m.matchup_subweights) {
    root.appendChild(el("p", "small", "The matchup component is itself a fixed weighted blend (no silent renormalization):"));
    root.appendChild(kvTable(Object.entries(m.matchup_subweights).map(([k, v]) => [k, v]), ["Matchup sub-score", "Weight"]));
  }

  if (m.selector) {
    root.appendChild(el("h2", null, "Selector tier thresholds (per market)"));
    root.appendChild(el("p", "lede", "The bar a pick must clear to earn each tier — a probability edge, an EV floor, and a model-confidence floor. Yardage markets get a lower bar; anytime-TD the highest (one-sided pricing + variance). All configurable."));
    const th = m.selector.thresholds || {};
    const t = el("table");
    t.innerHTML = `<thead><tr><th>Market</th><th>LEAN edge≥</th><th>PLAYABLE edge≥</th><th>STRONG edge≥</th><th>EV floor</th></tr></thead>`;
    const tb = el("tbody");
    Object.entries(th).forEach(([mk, b]) => {
      tb.appendChild(el("tr", null, `<td>${esc(mk)}</td><td class="num">${pts(b.lean)}</td><td class="num">${pts(b.playable)}</td><td class="num">${pts(b.strong)}</td><td class="num">${pts(b.ev_min)}%</td>`));
    });
    t.appendChild(tb); root.appendChild(t);
  }

  root.appendChild(el("h2", null, `All ${(m.features || []).length} model features`));
  const fw = el("div", "feat-wrap");
  (m.features || []).forEach(f => fw.appendChild(el("span", "pill", esc(f))));
  root.appendChild(fw);
  if ((m.retrain_pending_features || []).length) {
    root.appendChild(el("p", "small", `Retrain-gated (measured, riding the training frame, activated at the next model retrain): ${m.retrain_pending_features.map(esc).join(", ")}.`));
  }
}

function kvTable(rows, heads) {
  const t = el("table");
  t.innerHTML = `<thead><tr>${heads.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>`;
  const tb = el("tbody");
  rows.forEach(r => tb.appendChild(el("tr", null, `<td>${esc(r[0])}</td><td class="num">${esc(r[1])}</td>`)));
  t.appendChild(tb); return t;
}

/* ---------------- Measured constants ---------------- */
function renderConstants() {
  const c = DATA.constants || {}, root = $("#constants");
  root.appendChild(el("h2", null, "Every constant is measured, not guessed"));
  root.appendChild(el("p", "lede", esc(c.principle || "")));
  const t = el("table");
  t.innerHTML = `<thead><tr><th>Constant</th><th>Value</th><th>How it was measured (provenance)</th><th>Fit</th></tr></thead>`;
  const tb = el("tbody");
  (c.constants || []).forEach(k => {
    tb.appendChild(el("tr", null,
      `<td><b>${esc(k.name)}</b></td><td class="num">${esc(k.value)}<div class="small">${esc(k.unit)}</div></td>` +
      `<td class="prov">${esc(k.provenance)}</td><td><code>${esc((k.fit_script || "").replace("scripts/", ""))}</code></td>`));
  });
  t.appendChild(tb); root.appendChild(t);
  root.appendChild(el("p", "small", "Rejected-and-not-shipped effects (measured worse or insignificant): garbage-time filter, opponent-pace volume term, cold-weather passing term, O-line-out multiplier, opponent red-zone factor on the mean, drop-injury cleaning. Discipline cuts both ways."));
}

/* ---------------- Evidence ---------------- */
function renderEvidence() {
  const ev = DATA.evidence || {}, root = $("#evidence");
  root.appendChild(el("h2", null, "Is it actually good? The honest scoreboard"));
  root.appendChild(el("p", "lede", "Projection accuracy grades against free nflverse actuals. But the only accepted proof of a <em>betting</em> edge is forward CLV (closing-line value) — did our entry beat the market's close? Until that sample fills, the model is a research tool, and this page says so."));

  const kc = ev.killcheck || {};
  const box = el("div", "kpi");
  const v = kc.verdict || "INSUFFICIENT_SAMPLE";
  box.innerHTML =
    `<div class="box"><div class="k">Kill-check</div><div class="v"><span class="verdict ${v}">${esc(v.replace(/_/g, " "))}</span></div></div>` +
    `<div class="box"><div class="k">Resolved CLV leans</div><div class="v">${kc.n ?? 0}<span class="small"> / ${kc.min_sample ?? 150} needed</span></div></div>` +
    `<div class="box"><div class="k">Avg CLV</div><div class="v">${kc.lifetime_mean == null ? "—" : pts(kc.lifetime_mean)}</div></div>` +
    `<div class="box"><div class="k">Beat-the-close rate</div><div class="v">${kc.positive_rate == null ? "—" : pct(kc.positive_rate)}</div></div>`;
  root.appendChild(box);
  if (kc.detail) root.appendChild(el("p", "small", esc(kc.detail)));

  if (ev.picks_record && ev.picks_record.n) {
    root.appendChild(el("h2", null, "Graded picks by tier"));
    const rows = Object.entries(ev.picks_record.by_tier || {}).map(([t, r]) => [t, `${(r.hit_rate * 100).toFixed(1)}% (${r.n})`]);
    root.appendChild(kvTable(rows, ["Tier", "Hit rate (n)"]));
  }

  const rc = ev.recency_fit;
  if (rc) {
    root.appendChild(el("h2", null, "Measured upgrade: the recency-weight fit (Phase 8)"));
    root.appendChild(el("p", "lede", "The one core knob that shipped on reasoning, now fit. EWM span-8 with rest-game cleaning beat the old flat-8 average in <em>every market, every season</em> tested (walk-forward out-of-sample MAE):"));
    const sc = rc.season_consistency || {};
    const t = el("table");
    t.innerHTML = `<thead><tr><th>Market</th><th>Winning scheme</th><th>Seasons improved</th></tr></thead>`;
    const tb = el("tbody");
    Object.entries(sc).forEach(([mk, r]) => {
      const imp = r.seasons_improved ?? r.improved, tot = r.seasons_total ?? r.of;
      tb.appendChild(el("tr", null, `<td>${esc(mk)}</td><td>${esc(r.winner || "ewm8")}</td><td class="num">${imp}/${tot}</td>`));
    });
    t.appendChild(tb); root.appendChild(t);
    root.appendChild(el("p", "small", "Same rest-game cleaning was then extended to opponent-defense factors, team pace, the ML training frame, the player-learning ledger, and the calibration &amp; correlation fits (decisions_p8 §8.4)."));
  }

  const so = ev.situations;
  if (so && so.tags) {
    root.appendChild(el("h2", null, "Narrative tags: tested, gated, mostly rejected"));
    root.appendChild(el("p", "lede", "Revenge games, primetime, travel, birthdays — each measured against a significance bar (n≥100, BH-corrected q<0.05) before it can move a bet. On 2019–2025 history, <em>none cleared</em>. They stay visible in context panels but carry zero weight."));
    const rows = Object.entries(so.tags).sort((a, b) => (a[1].q_value ?? 1) - (b[1].q_value ?? 1)).slice(0, 8)
      .map(([t, r]) => `<tr><td>${esc(t)}</td><td class="num">${r.n}</td><td class="num">${(r.hit_rate * 100).toFixed(1)}%</td><td class="num">${r.q_value == null ? "—" : r.q_value}</td><td>${esc(r.verdict)}</td></tr>`);
    const t = el("table");
    t.innerHTML = `<thead><tr><th>Tag</th><th>n</th><th>Hit</th><th>q-value</th><th>Verdict</th></tr></thead><tbody>${rows.join("")}</tbody>`;
    root.appendChild(t);
  }
}

boot();
