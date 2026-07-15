"""Render the self-contained, auto-refreshing dashboard.

The pipeline inlines all data into the HTML and the page reloads itself on a
timer, so re-running ``run.py`` makes the open page show fresh numbers. No web
server, no external libraries, works offline by double-clicking the file.
"""

from __future__ import annotations

import json
from typing import Dict

from . import config

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Value Bets</title>
<style>
:root{
  --bg:#0b0f17; --panel:#141b29; --panel2:#1b2435; --line:#26304a;
  --text:#e8edf6; --muted:#8b97ad; --accent:#4ea1ff; --green:#2fd07a;
  --red:#ff5d6c; --yellow:#ffce54; --chip:#222d44;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 70% -10%,#16203a 0,#0b0f17 60%);
  color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--accent)}
.wrap{max-width:1180px;margin:0 auto;padding:22px 18px 60px}
header{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-bottom:6px}
h1{font-size:20px;margin:0;letter-spacing:.3px}
h1 .v{color:var(--accent)}
.badge{font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px;text-transform:uppercase;letter-spacing:.6px}
.badge.demo,.badge.offseason{background:#3a2d12;color:var(--yellow);border:1px solid #6b521f}
.badge.live,.badge.active{background:#10331f;color:var(--green);border:1px solid #1f6b40}
.badge.degraded{background:#3a1419;color:var(--red);border:1px solid #742c36}
.meta{margin-left:auto;color:var(--muted);font-size:12px;text-align:right}
.pipeline{margin:12px 0 4px;padding:10px 13px;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:#cdd7ea;font-size:12px}
.pipeline.offseason{border-color:#6b521f}.pipeline.degraded{border-color:#742c36}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0 8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.card .val{font-size:22px;font-weight:700;margin-top:3px}
.tabs{display:flex;gap:6px;margin:18px 0 12px;flex-wrap:wrap}
.tab{padding:8px 14px;background:var(--panel);border:1px solid var(--line);border-radius:9px;cursor:pointer;color:var(--muted);font-weight:600}
.tab.active{background:var(--accent);color:#03131f;border-color:var(--accent)}
.panel{display:none}
.panel.active{display:block}
table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);font-size:13px;vertical-align:middle}
th{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;background:var(--panel2)}
tr:last-child td{border-bottom:none}
tr:hover td{background:#172033}
.pick{font-weight:700}
.sub{color:var(--muted);font-size:11.5px}
.ev{font-weight:800}
.ev.pos{color:var(--green)} .ev.neg{color:var(--red)}
.pill{display:inline-block;background:var(--chip);border:1px solid var(--line);border-radius:20px;padding:2px 9px;font-size:11px;margin:2px 3px 0 0;color:#cdd7ea}
.pill.p{border-color:#1f5a39;color:#7be0a8} .pill.n{border-color:#6b2530;color:#ff97a1}
.book{font-size:11px;color:var(--muted);text-transform:capitalize}
.price{font-variant-numeric:tabular-nums;font-weight:700}
.wx{font-size:11px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.box{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.box h3{margin:0 0 12px;font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
.wbar{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:12px}
.wbar .name{width:120px;color:#cdd7ea}
.wbar .track{flex:1;height:8px;background:#0c1320;border-radius:6px;position:relative;overflow:hidden}
.wbar .fill{position:absolute;top:0;bottom:0;background:var(--accent);border-radius:6px}
.wbar .num{width:46px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.note{color:var(--muted);font-size:12px;margin-top:10px}
.empty{padding:26px;text-align:center;color:var(--muted)}
.disclaimer{margin-top:26px;padding:14px 16px;background:#160f12;border:1px solid #3a1f25;border-radius:10px;color:#d9a7af;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>NFL <span class="v">Value</span> Engine</h1>
    <span id="mode" class="badge"></span>
    <div class="meta">
      <div id="updated"></div>
      <div>auto-refresh in <span id="count"></span>s &middot; reloads when the pipeline reruns</div>
    </div>
  </header>

  <div class="pipeline" id="pipeline" hidden></div>

  <div class="cards" id="cards"></div>

  <div class="tabs">
    <div class="tab active" data-t="weekly">Weekly Projections</div>
    <div class="tab" data-t="bets">Value Bets</div>
    <div class="tab" data-t="props">Player Props</div>
    <div class="tab" data-t="leans">Weekly Leans</div>
    <div class="tab" data-t="mc">Monte Carlo</div>
    <div class="tab" data-t="games">All Games</div>
    <div class="tab" data-t="perf">Model &amp; Learning</div>
    <div class="tab" data-t="audit">Factor Audit</div>
    <div class="tab" data-t="backtest">Backtest</div>
  </div>

  <div class="panel active" id="weekly"></div>
  <div class="panel" id="bets"></div>
  <div class="panel" id="props"></div>
  <div class="panel" id="leans"></div>
  <div class="panel" id="mc"></div>
  <div class="panel" id="games"></div>
  <div class="panel" id="perf"></div>
  <div class="panel" id="audit"></div>
  <div class="panel" id="backtest"></div>

  <div class="disclaimer" id="disc"></div>
</div>

<script>
const DATA = __DATA_JSON__;

const fmtPct = x => (x>=0?"+":"") + (x*100).toFixed(1) + "%";
const fmtP = x => (x*100).toFixed(1) + "%";
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function evClass(ev){return ev>0?"ev pos":"ev neg";}

function whyPills(why){
  if(!why||!why.length) return '<span class="sub">market price edge</span>';
  return why.map(w=>`<span class="pill ${w.dir==='+'?'p':'n'}">${esc(w.factor)} ${w.dir}</span>`).join("");
}

function betRow(b){
  return `<tr>
    <td><div class="pick">${esc(b.outcome)}</div>
        <div class="sub">${esc(b.away_team)} @ ${esc(b.home_team)} &middot; ${esc(b.market)}</div></td>
    <td><div class="price">${esc(b.price_american)}</div><div class="book">${esc(b.best_book)}</div></td>
    <td><div class="sub">fair ${fmtP(b.p_consensus)}</div><div>proj ${fmtP(b.p_bet!=null?b.p_bet:b.p_model)}</div></td>
    <td class="${evClass(b.ev)}">${fmtPct(b.ev)}</td>
    <td>${b.stake_units.toFixed(2)}u</td>
    <td>${whyPills(b.why)}</td>
  </tr>`;
}

function renderBets(){
  const el=document.getElementById("bets");
  const b=DATA.value_bets||[];
  if(!b.length){el.innerHTML='<div class="box empty">No +EV game-line bets on the current slate. The market is efficient — this is normal. Check Props, or rerun when lines move.</div>';return;}
  el.innerHTML=`<table><thead><tr><th>Bet</th><th>Price</th><th>Probability</th><th>EV</th><th>Stake</th><th>Why (learned factors)</th></tr></thead>
    <tbody>${b.map(betRow).join("")}</tbody></table>
    <div class="note">Stake = fractional-Kelly suggestion in units (1u = 1% of bankroll). EV uses the best price across books vs the de-vigged market consensus, nudged by learned factors.</div>`;
}

function renderProps(){
  const el=document.getElementById("props");
  const b=DATA.value_props||[];
  if(!b.length){el.innerHTML='<div class="box empty">No +EV player props on the current slate.</div>';return;}
  el.innerHTML=`<table><thead><tr><th>Prop</th><th>Price</th><th>Probability</th><th>EV</th><th>Stake</th><th>Why</th></tr></thead>
    <tbody>${b.map(betRow).join("")}</tbody></table>`;
}

function renderLeans(){
  const el=document.getElementById("leans");
  const w=DATA.weekly_leans;
  if(!w||!w.games||!w.games.length){
    el.innerHTML='<div class="box empty">No weekly prop leans generated yet — run <code>python3 pipeline_weekly.py --season S --week W</code>.</div>';
    return;
  }
  const clv=DATA.leans_clv||{};
  const kc=DATA.leans_killcheck||{};
  const pub = w.publish===false
    ? `<div class="box" style="border-color:var(--red)"><b>NOT PUBLISHED</b> — data gate failed: ${esc((w.publish_reasons||[]).join("; "))}</div>` : "";
  const clvBox = `<div class="box"><b>Forward CLV</b> — resolved leans: ${clv.n||0}
      ${clv.lifetime_mean!=null?` · lifetime avg ${fmtPct(clv.lifetime_mean)} prob`:""}
      ${clv.rolling_mean!=null?` · rolling(${clv.window}) ${fmtPct(clv.rolling_mean)}`:""}
      ${kc.verdict?` · kill-check: <b>${esc(kc.verdict)}</b>`:""}
      <div class="sub">${esc(kc.detail||"CLV accrues only once real prop lines are pulled live (Phase 3).")}</div></div>`;
  const sideLabel = l => l.market==="anytime_td" ? "YES" : (l.side||"").toUpperCase();
  const games = w.games.map(g=>{
    const ctx=(w.contexts||{})[g.game_id];
    const rows=g.leans.map(l=>`<tr>
      <td><div class="pick">${esc(l.name)}</div><div class="sub">${esc(l.pos)} · ${esc(l.team)}</div></td>
      <td>${esc(String(l.market||"").replace(/_/g," "))}</td>
      <td class="price">${esc(l.line)}${l.line_source==="odds_api"?"":"†"}</td>
      <td><b>${sideLabel(l)}</b></td>
      <td>${esc(l.mean)}</td>
      <td>${l.edge!=null?fmtPct(l.edge):'<span class="sub">no_market</span>'}</td>
      <td class="price">${esc(l.composite)}</td></tr>`).join("");
    const ctxItems = ctx? ctx.entries.map(e=>e.items.map(i=>`<div class="sub">• <b>${esc(e.name)}</b> — ${esc(i)}</div>`).join("")).join("") : "";
    return `<div class="box"><b>${esc(g.matchup)}</b> <span class="sub">top ${g.leans.length} of ${g.screened_n} screened</span>
      <table><thead><tr><th>Player</th><th>Market</th><th>Line</th><th>Side</th><th>Proj</th><th>Edge</th><th>Score</th></tr></thead>
      <tbody>${rows}</tbody></table>
      ${ctx?`<div class="note"><b>Context — display only, never scored:</b>${ctxItems}</div>`:""}</div>`;
  }).join("");
  el.innerHTML = `<div class="note"><b>Leans, not locks.</b> ${esc(w.season)} week ${esc(w.week)} · clock ${esc(w.clock)} · as of ${esc(w.as_of)} · † = synthetic reference line (player's own trailing mean), not a market price — edge needs a real sportsbook line. If you or someone you know has a gambling problem: <b>1-800-GAMBLER</b>.</div>
    ${pub}${clvBox}${games}`;
}

function renderGames(){
  const el=document.getElementById("games");
  const g=DATA.all_games||[];
  if(!g.length){el.innerHTML='<div class="box empty">No games loaded.</div>';return;}
  el.innerHTML=`<table><thead><tr><th>Matchup</th><th>Kickoff</th><th>Spread</th><th>Total</th><th>Moneyline</th><th>Context</th></tr></thead>
    <tbody>${g.map(x=>`<tr>
      <td><div class="pick">${esc(x.away_team)} @ ${esc(x.home_team)}</div></td>
      <td class="sub">${esc(x.kickoff)}</td>
      <td>${esc(x.spread)}</td>
      <td>${esc(x.total)}</td>
      <td class="sub">${esc(x.ml)}</td>
      <td class="wx">${esc(x.context)}</td>
    </tr>`).join("")}</tbody></table>`;
}

function svgLine(points,w,h,color){
  if(!points.length) return "";
  const min=Math.min(...points), max=Math.max(...points), rng=(max-min)||1;
  const step=w/Math.max(points.length-1,1);
  const pts=points.map((v,i)=>`${(i*step).toFixed(1)},${(h-((v-min)/rng)*h).toFixed(1)}`).join(" ");
  return `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block">
    <polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}"/></svg>`;
}

function renderPerf(){
  const m=DATA.metrics||{}, w=DATA.weights||{};
  const cal=(m.calibration||[]);
  const calRows=cal.map(c=>{
    const diff=Math.abs(c.predicted-c.actual);
    const col=diff<0.06?"var(--green)":diff<0.12?"var(--yellow)":"var(--red)";
    return `<tr><td>${c.bucket}</td><td>${fmtP(c.predicted)}</td>
      <td style="color:${col}">${fmtP(c.actual)}</td><td class="sub">${c.n}</td></tr>`;}).join("");
  const eq=m.equity_curve||[];
  const wkeys=Object.keys(w).sort((a,b)=>Math.abs(w[b])-Math.abs(w[a]));
  const wmax=Math.max(0.01,...wkeys.map(k=>Math.abs(w[k])));
  const wbars=wkeys.map(k=>{
    const pos=w[k]>=0, pct=(Math.abs(w[k])/wmax)*100;
    return `<div class="wbar"><div class="name">${esc(k.replace(/^[gtp]_/,''))}</div>
      <div class="track"><div class="fill" style="width:${pct}%;background:${pos?'var(--green)':'var(--red)'}"></div></div>
      <div class="num">${w[k].toFixed(2)}</div></div>`;}).join("");

  document.getElementById("perf").innerHTML=`
    <div class="grid2">
      <div class="box"><h3>Betting performance (recommended picks)</h3>
        <div class="cards" style="margin:0">
          <div class="card"><div class="k">Bets settled</div><div class="val">${m.bets_settled||0}</div></div>
          <div class="card"><div class="k">Bet hit rate</div><div class="val">${fmtP(m.bets_win_rate||0)}</div></div>
          <div class="card"><div class="k">Bet ROI</div><div class="val ${(m.bets_roi||0)>=0?'ev pos':'ev neg'}">${fmtPct(m.bets_roi||0)}</div></div>
          <div class="card"><div class="k">Brier</div><div class="val">${(m.brier||0).toFixed(3)}</div></div>
        </div>
        <div class="note">Hit rate &amp; ROI are for the picks that cleared the EV threshold (1u flat). Brier is calibration over <b>all</b> ${m.settled_total||0} graded predictions (lower is better; 0.25 = a coin flip).</div>
      </div>
      <div class="box"><h3>Bankroll (recommended bets, fractional Kelly)</h3>
        ${eq.length>1?svgLine(eq,420,120,"var(--green)"):'<div class="empty">No settled bets yet — grade some games to populate.</div>'}
        <div class="note">Start ${ (m.start_bankroll||100).toFixed(0)}u &rarr; now <b>${(m.bankroll||100).toFixed(1)}u</b> over ${m.bets_settled||0} bets.</div>
      </div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div class="box"><h3>Calibration: predicted vs actual</h3>
        ${cal.length?`<table><thead><tr><th>Model prob</th><th>Predicted</th><th>Actual</th><th>n</th></tr></thead><tbody>${calRows}</tbody></table>`:'<div class="empty">Not enough graded predictions yet.</div>'}
      </div>
      <div class="box"><h3>Learned factor weights</h3>
        ${wbars}
        <div class="note">These start as small priors and move as games finish. Green = the factor pushes toward the bet; red = against. This is the model "learning what predictions were wrong and adjusting."</div>
      </div>
    </div>`;
}

let WEEKLY_WK = null;
function pickChip(txt, res){
  const c = res==='W' ? 'p' : res==='L' ? 'n' : '';
  const mark = res==='W' ? ' ✓' : res==='L' ? ' ✗' : res==='P' ? ' push' : '';
  return `<span class="pill ${c}">${esc(txt)}${mark}</span>`;
}
function weekRow(g){
  const margin = g.proj_margin>=0 ? `${g.home} by ${g.proj_margin}` : `${g.away} by ${(-g.proj_margin).toFixed(1)}`;
  const su = g.settled ? (g.su_correct===null ? 'P' : (g.su_correct ? 'W' : 'L')) : undefined;
  const winTeam = g.p_home_win>=g.p_away_win ? g.home : g.away;
  const winPct = Math.max(g.p_home_win,g.p_away_win)*100;
  const sp = g.market_spread_home>0 ? '+'+g.market_spread_home : g.market_spread_home;
  const apk = g.ats_pick, aln = apk.line>0?'+'+apk.line:apk.line;
  const final = g.settled ? `${esc(g.away)} ${g.away_score|0} – ${g.home_score|0} ${esc(g.home)}`
                          : '<span class="sub">upcoming</span>';
  return `<tr>
    <td><div class="pick">${esc(g.away)} @ ${esc(g.home)}</div>
        <div class="sub">${esc(winTeam)} ${winPct.toFixed(0)}% to win</div></td>
    <td><div><b>${esc(g.home)} ${g.proj_home}</b> – ${g.proj_away} ${esc(g.away)}</div>
        <div class="sub">proj ${esc(margin)}</div></td>
    <td><div>${esc(g.home)} ${sp}</div><div class="sub">O/U ${g.market_total}</div></td>
    <td>${pickChip('SU '+g.su_pick, su)} ${pickChip('ATS '+apk.team+' '+aln, g.ats_result)} ${pickChip(g.total_pick.side.toUpperCase()+' '+g.total_pick.line, g.total_result)}</td>
    <td class="sub">${final}</td>
  </tr>`;
}
function renderWeekly(){
  const el=document.getElementById("weekly");
  const wd=DATA.weekly;
  if(!wd){el.innerHTML='<div class="box empty">No weekly projections yet. Run <code>python3 weekly.py</code> (after build_ratings.py).</div>';return;}
  if(WEEKLY_WK===null) WEEKLY_WK=wd.default_week;
  let wk=wd.weeks.find(w=>w.week===WEEKLY_WK); if(!wk) wk=wd.weeks[wd.weeks.length-1];
  const r=wk.record_to_date;
  const opts=wd.weeks.map(w=>`<option value="${w.week}" ${w.week===wk.week?'selected':''}>${esc(w.label)}</option>`).join("");
  el.innerHTML=`
    <div class="box" style="margin-bottom:14px">
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px">
        <div><span class="sub">${wd.mode==='live'?'Live slate':'Season '+wd.season}</span>
          <select id="wksel" style="margin-left:8px;background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-weight:600">${opts}</select>
        </div>
        <div class="meta" style="margin-left:auto;text-align:right">Record through ${esc(wk.label)}:<br>
          <b style="color:var(--green)">SU ${r.su}</b> (${(r.su_pct*100).toFixed(0)}%) &middot;
          ATS ${r.ats} (${(r.ats_pct*100).toFixed(0)}%) &middot; Totals ${r.totals} &middot;
          avg proj error ${r.avg_margin_err} pts</div>
      </div>
    </div>
    <table><thead><tr><th>Matchup</th><th>Projection</th><th>Market line</th><th>Picks</th><th>Final</th></tr></thead>
      <tbody>${wk.games.map(weekRow).join("")}</tbody></table>
    <div class="note">Each week is projected with the Monte Carlo using only data through the prior week (walk-forward), then graded as results come in — like fantasy projections. SU = straight-up winner, ATS = against the spread. Step through the weeks with the dropdown to watch the record build. ${wd.mode!=='live'?'Change season with <code>python3 weekly.py --season 2022</code>.':''}</div>`;
  const sel=document.getElementById("wksel");
  if(sel) sel.onchange=()=>{WEEKLY_WK=parseInt(sel.value);renderWeekly();};
}

function miniBars(hist){
  if(!hist||!hist.counts) return "";
  const mx=Math.max(...hist.counts)||1;
  return '<div style="display:flex;align-items:flex-end;gap:1px;height:30px">'+
    hist.counts.map((c,i)=>{const h=Math.round(c/mx*30);const home=hist.bins[i]>=0;
      return `<div title="${hist.bins[i]} margin: ${(c*100).toFixed(1)}%" style="width:6px;height:${h}px;background:${home?'var(--accent)':'#556'}"></div>`;}).join("")+
    '</div>';
}

function renderMonteCarlo(){
  const el=document.getElementById("mc");
  const m=DATA.monte_carlo;
  if(!m){el.innerHTML='<div class="box empty">Monte Carlo is off. Run <code>python3 build_ratings.py</code> (needs pandas+numpy) to build team ratings from your historical data, then rerun.</div>';return;}
  const rows=m.games.map(g=>{
    const cover=g.p_home_cover!=null?`${g.market_home_point>0?'+':''}${g.market_home_point}: ${fmtP(g.p_home_cover)}`:'—';
    const over=g.p_over!=null?`${g.market_total}: O ${fmtP(g.p_over)}`:'—';
    return `<tr>
      <td><div class="pick">${esc(g.away_team)} @ ${esc(g.home_team)}</div>
          <div class="sub">proj ${g.proj_away}–${g.proj_home} &middot; margin ${g.margin_mean>0?'+':''}${g.margin_mean}±${g.margin_sd}</div></td>
      <td><div>${esc(g.home_team.split(' ').pop())} ${fmtP(g.p_home_win)}</div>
          <div class="sub">fair ${g.fair_home_ml>0?'+':''}${g.fair_home_ml}</div></td>
      <td class="sub">${cover}</td>
      <td class="sub">${over}</td>
      <td>${miniBars(g.margin_hist)}</td>
    </tr>`;}).join("");
  el.innerHTML=`<div class="box" style="margin-bottom:14px">
      <h3>Drive-by-drive Monte Carlo &middot; ${ (m.seasons||[]).join('–') } data &middot; ${m.n_sims.toLocaleString()} sims/game</h3>
      <div class="note">Each game is simulated possession-by-possession from team power ratings, scoring real integer points so the margin distribution spikes on 3 and 7. These are <b>calibrated projections</b> (see Backtest) — a fair-value second opinion, not an edge against the closing line.</div>
    </div>
    <table><thead><tr><th>Matchup (proj score)</th><th>Win prob</th><th>Cover vs mkt</th><th>Total vs mkt</th><th>Margin distribution</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function renderBacktest(){
  const el=document.getElementById("backtest");
  const b=DATA.backtest;
  if(!b){el.innerHTML='<div class="box empty">No backtest yet. Run <code>python3 backtest.py</code> after <code>build_ratings.py</code>.</div>';return;}
  const mk=(name,m)=>`<tr><td>${name}</td><td>${m.bets}</td><td>${fmtP(m.win_rate)}</td>
     <td class="${m.roi>=0?'ev pos':'ev neg'}">${fmtPct(m.roi)}</td><td class="sub">${m.units>=0?'+':''}${m.units}u</td></tr>`;
  const cal=(b.calibration||[]).filter(c=>c.n>=20).map(c=>{
    const d=Math.abs(c.predicted-c.actual);const col=d<0.05?'var(--green)':d<0.1?'var(--yellow)':'var(--red)';
    return `<tr><td>${c.bucket}</td><td>${fmtP(c.predicted)}</td><td style="color:${col}">${fmtP(c.actual)}</td><td class="sub">${c.n}</td></tr>`;}).join("");
  el.innerHTML=`
    <div class="cards" style="margin:0 0 14px">
      <div class="card"><div class="k">Games tested</div><div class="val">${b.n_games}</div></div>
      <div class="card"><div class="k">Win-prob Brier</div><div class="val">${(b.brier||0).toFixed(3)}</div></div>
      <div class="card"><div class="k">Model corr</div><div class="val">${(b.corr_model||0).toFixed(2)}</div></div>
      <div class="card"><div class="k">Market corr</div><div class="val">${(b.corr_market||0).toFixed(2)}</div></div>
    </div>
    <div class="box" style="margin-bottom:14px;border-color:#3a4d2a;background:#11180f">
      <h3 style="color:var(--green)">What the backtest says</h3>
      <div style="color:#cfe6c4">${esc(b.verdict||'')}</div>
    </div>
    <div class="grid2">
      <div class="box"><h3>Win-probability calibration (predicted vs actual)</h3>
        ${cal?`<table><thead><tr><th>Predicted win%</th><th>Predicted</th><th>Actual</th><th>n</th></tr></thead><tbody>${cal}</tbody></table>
        <div class="note">Close diagonal = trustworthy probabilities. This is the model working as intended.</div>`:'<div class="empty">n/a</div>'}
      </div>
      <div class="box"><h3>Betting INTO the closing line (honest result)</h3>
        <table><thead><tr><th>Market</th><th>Bets</th><th>Win%</th><th>ROI</th><th>P/L</th></tr></thead>
        <tbody>${mk('Spread (-110)',b.spread)}${mk('Total (-110)',b.total)}${mk('Moneyline',b.ml)}</tbody></table>
        <div class="note">ATS pick accuracy vs the close: <b>${fmtP(b.ats_pick_accuracy)}</b>. Negative ROI here is expected — closing lines are efficient. Real edges come from line-shopping softer prices (the Value Bets tab), not fighting the close.</div>
      </div>
    </div>`;
}

function renderAudit(){
  const el=document.getElementById("audit");
  const a=DATA.factor_audit, n=DATA.nested_factor_projection;
  if(!a){el.innerHTML='<div class="box empty">No all-data factor audit has been published.</div>';return;}
  const pp=x=>(x>=0?"+":"")+Number(x).toFixed(2)+"pp";
  const ci=x=>x&&x.length===2?`[${pp(x[0])}, ${pp(x[1])}]`:"—";
  const q=x=>x==null?"—":Number(x).toPrecision(3);
  const findings=(a.findings||[]).map(f=>`<tr>
    <td><b>${esc(f.name)}</b><div class="sub">${esc(f.cohort)}</div></td>
    <td>${f.exposed_hits}/${f.exposed_n} (${fmtP(f.exposed_rate)})</td>
    <td>${f.control_hits}/${f.control_n} (${fmtP(f.control_rate)})</td>
    <td class="${f.difference_pp>=0?'ev pos':'ev neg'}">${pp(f.difference_pp)}</td>
    <td>${ci(f.credible_interval_95_pp)}</td><td>${ci(f.cluster_interval_95_pp)}</td><td>${q(f.bh_q)}</td>
  </tr>`).join("");
  const nulls=(a.notable_nulls||[]).map(f=>`<tr><td>${esc(f.name)}</td><td>${f.exposed_n}</td>
    <td>${pp(f.difference_pp)}</td><td>${ci(f.credible_interval_95_pp)}</td><td>${q(f.bh_q)}</td></tr>`).join("");
  const segments=((n&&n.segments)||[]).map(s=>`<tr><td>${esc(s.name)}</td><td>${s.correct!=null?s.correct+'/':''}${s.outer_n}</td>
    <td>${fmtP(s.accuracy)}</td><td>${fmtP(s.segment_reference_accuracy)}</td><td>${pp(s.selective_lift_pp)}</td>
    <td>[${fmtP(s.posterior_95[0])}, ${fmtP(s.posterior_95[1])}]${s.excluded_from_best_reason?`<div class="sub">${esc(s.excluded_from_best_reason)}</div>`:""}</td></tr>`).join("");
  const best=n&&n.highest_eligible_accuracy;
  el.innerHTML=`
    <div class="box" style="margin-bottom:14px;border-color:#3a4d2a;background:#11180f">
      <h3 style="color:var(--green)">Honest all-data result</h3>
      <div><b>${a.frame_rows.toLocaleString()} player-market rows</b> · seasons ${(a.seasons||[]).join('–')} · ${a.method.family_size} predeclared tests.</div>
      <div class="note">${esc(a.target)}. ${esc(a.live_scoring_impact)}</div>
      ${best?`<div class="note"><b>No 100% system:</b> highest eligible nested accuracy was ${fmtP(best.accuracy)} on n=${best.outer_n}, but selective lift was ${pp(best.selective_lift_pp)} versus its segment reference.</div>`:""}
    </div>
    <div class="box"><h3>Signals surviving all-test correction</h3>
      <table><thead><tr><th>Factor</th><th>Exposed</th><th>Control</th><th>Difference</th><th>Bayes 95%</th><th>Cluster 95%</th><th>BH q</th></tr></thead><tbody>${findings}</tbody></table>
      <div class="note">Jeffreys beta-binomial Monte Carlo (${a.method.posterior}); dependence check: ${a.method.dependence}. Association is not automatic permission to score it live.</div>
    </div>
    <div class="grid2" style="margin-top:16px">
      <div class="box"><h3>Notable nulls</h3><table><thead><tr><th>Factor</th><th>n</th><th>Difference</th><th>Bayes 95%</th><th>BH q</th></tr></thead><tbody>${nulls}</tbody></table></div>
      <div class="box"><h3>Nested outer-season combinations</h3>${segments?`<table><thead><tr><th>Segment</th><th>Hits/n</th><th>Accuracy</th><th>Reference</th><th>Lift</th><th>95%</th></tr></thead><tbody>${segments}</tbody></table>`:'<div class="empty">No nested projection result.</div>'}
        <div class="note">${esc((n&&n.conclusion)||'')}</div></div>
    </div>`;
}

function renderCards(){
  const s=DATA.summary||{};
  document.getElementById("cards").innerHTML=`
    <div class="card"><div class="k">Games</div><div class="val">${s.n_games||0}</div></div>
    <div class="card"><div class="k">Value bets</div><div class="val">${s.n_value_bets||0}</div></div>
    <div class="card"><div class="k">Value props</div><div class="val">${s.n_value_props||0}</div></div>
    <div class="card"><div class="k">Best edge</div><div class="val ev pos">${s.best_ev!=null?fmtPct(s.best_ev):"—"}</div></div>
    <div class="card"><div class="k">Bankroll</div><div class="val">${(DATA.metrics&&DATA.metrics.bankroll||100).toFixed(1)}u</div></div>`;
}

function renderPipeline(){
  const p=DATA.pipeline||{};
  const status=(p.status||DATA.mode||"demo").toLowerCase();
  document.getElementById("mode").textContent=status;
  document.getElementById("mode").className="badge "+status;
  const checked=p.last_checked_at?`Pipeline checked ${p.last_checked_at}`:"";
  const snapshot=DATA.generated_at?`data snapshot ${DATA.generated_at}`:"";
  document.getElementById("updated").textContent=[checked,snapshot].filter(Boolean).join(" · ");
  if(!DATA.pipeline) return;
  const ints=p.integrations||{};
  const labels=Object.keys(ints).map(k=>`<span class="pill ${ints[k]==='configured'?'p':'n'}">${esc(k.replace(/_/g,' '))}: ${esc(ints[k])}</span>`).join("");
  const box=document.getElementById("pipeline");box.hidden=false;box.className="pipeline "+status;
  box.innerHTML=`<b>${esc(status.toUpperCase())}</b> · ${esc(p.detail||"")} ${labels}`;
}

// header + tabs + auto refresh
renderPipeline();
document.getElementById("disc").innerHTML=DATA.disclaimer||"";
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(x=>x.classList.remove("active"));
  t.classList.add("active");
  document.getElementById(t.dataset.t).classList.add("active");
});
renderWeekly();renderCards();renderBets();renderProps();renderLeans();renderMonteCarlo();renderGames();renderPerf();renderAudit();renderBacktest();

let secs=DATA.refresh_seconds||90;
const cd=document.getElementById("count");cd.textContent=secs;
setInterval(()=>{secs--;cd.textContent=Math.max(secs,0);if(secs<=0)location.reload();},1000);
</script>
</body>
</html>
"""


def write_dashboard(data: Dict, path: str = None) -> str:
    path = path or config.DASHBOARD_PATH
    payload = dict(data)
    payload.setdefault("factor_audit", config.load_json(
        config.ALL_DATA_FACTOR_AUDIT_PATH, None))
    payload.setdefault("nested_factor_projection", config.load_json(
        config.NESTED_FACTOR_PROJECTION_PATH, None))
    html = TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, default=str))
    with open(path, "w") as f:
        f.write(html)
    return path
