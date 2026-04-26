/* ══════════════════════════════════════════
   NegotiArena — Complete Script
   Fixes:
   - Speedometer gap removed (proper height/margin)
   - Left-side transcript gap removed
   - Difficulty changes produce different data per mode
   - Same button twice always gives different data
   - Post-sim training charts removed from Live Demo page
   - Analytics page only shows training charts
══════════════════════════════════════════ */

/* ════════════ REAL GRPO DATA ════════════ */
const GRPO_DATA = [
  {step:5,reward:0.15,reward_std:0.028,kl:0.000017,loss:0.0},
  {step:10,reward:0.1225,reward_std:0.049,kl:0.000022,loss:0.0},
  {step:15,reward:0.1375,reward_std:0.035,kl:0.000212,loss:0.0},
  {step:20,reward:0.325,reward_std:0.188,kl:0.001062,loss:0.0002},
  {step:25,reward:0.1325,reward_std:0.034,kl:0.00295,loss:0.0004},
  {step:30,reward:0.26,reward_std:0.000,kl:0.002607,loss:0.0004},
  {step:35,reward:-0.01,reward_std:0.215,kl:0.006887,loss:0.001},
  {step:40,reward:0.2025,reward_std:0.396,kl:0.028024,loss:0.0042},
  {step:45,reward:0.485,reward_std:0.127,kl:0.04134,loss:0.0062},
  {step:50,reward:-0.145,reward_std:0.353,kl:0.122497,loss:0.0184},
  {step:55,reward:0.17,reward_std:0.211,kl:0.007707,loss:0.0012},
  {step:60,reward:0.7525,reward_std:0.140,kl:0.03655,loss:0.0055},
  {step:65,reward:0.2325,reward_std:0.166,kl:0.010421,loss:0.0016},
  {step:70,reward:0.2925,reward_std:0.038,kl:0.008892,loss:0.0013},
  {step:75,reward:0.0975,reward_std:0.179,kl:0.01142,loss:0.0017},
  {step:80,reward:0.0825,reward_std:0.179,kl:0.015473,loss:0.0023},
  {step:85,reward:0.1675,reward_std:0.177,kl:0.010868,loss:0.0016},
  {step:90,reward:0.27,reward_std:0.028,kl:0.007338,loss:0.0011},
  {step:95,reward:0.33,reward_std:0.028,kl:0.004844,loss:0.0007},
  {step:100,reward:0.3975,reward_std:0.158,kl:0.019167,loss:0.0029},
  {step:105,reward:0.5275,reward_std:0.440,kl:0.02577,loss:0.0039},
  {step:110,reward:0.0925,reward_std:0.063,kl:0.016804,loss:0.0025},
  {step:115,reward:0.6075,reward_std:0.092,kl:0.01899,loss:0.0028},
  {step:120,reward:0.3725,reward_std:0.214,kl:0.0382,loss:0.0057},
  {step:125,reward:0.745,reward_std:0.114,kl:0.035309,loss:0.0053},
  {step:130,reward:0.6075,reward_std:0.424,kl:0.056137,loss:0.0084},
  {step:135,reward:0.28,reward_std:0.153,kl:0.051998,loss:0.0078},
  {step:140,reward:0.7525,reward_std:0.105,kl:0.057057,loss:0.0086},
  {step:145,reward:0.375,reward_std:0.105,kl:0.069446,loss:0.0104},
  {step:150,reward:0.5,reward_std:0.298,kl:0.087684,loss:0.0132},
  {step:155,reward:0.355,reward_std:0.164,kl:0.142432,loss:0.0214},
  {step:160,reward:0.3825,reward_std:0.121,kl:0.091353,loss:0.0137},
  {step:165,reward:0.64,reward_std:0.131,kl:0.084209,loss:0.0126},
  {step:170,reward:0.3675,reward_std:0.145,kl:0.089194,loss:0.0134},
  {step:175,reward:0.3925,reward_std:0.132,kl:0.099452,loss:0.0149},
  {step:180,reward:0.5175,reward_std:0.119,kl:0.137834,loss:0.0207},
  {step:185,reward:0.4975,reward_std:0.155,kl:0.096133,loss:0.0144},
  {step:190,reward:0.76,reward_std:0.334,kl:0.116511,loss:0.0175},
  {step:195,reward:0.485,reward_std:0.146,kl:0.092092,loss:0.0138},
  {step:200,reward:0.755,reward_std:0.082,kl:0.055716,loss:0.0084},
];

const RLVR_DATA = {
  steps:[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
  reward_mean:[0.21,0.29,0.38,0.42,0.35,0.51,0.48,0.56,0.59,0.62,0.57,0.68,0.65,0.72,0.70,0.73,0.76,0.74,0.78,0.80],
  kl:[0.001,0.003,0.008,0.015,0.025,0.018,0.020,0.030,0.028,0.035,0.040,0.038,0.050,0.055,0.062,0.058,0.065,0.070,0.068,0.060]
};

const PERFORMANCE = {
  methods:['Random','Heuristic','GRPO','RLVR'],
  precision:[0.33,0.54,0.71,0.82],
  recall:[0.33,0.48,0.68,0.79],
  f1:[0.33,0.51,0.69,0.80]
};

/* ════════════ DIFFICULTY CONFIG ════════════
   Each difficulty has its own data profile:
   - different agents count
   - different coalition probability
   - different noise (affects allocation spread)
   - different reward ranges
   - different GRPO step pools (from real data)
*/
const DIFF = {
  easy: {
    agents: 2,
    agentNames: ['negotiator_a','negotiator_b'],
    coalitionProb: 0.30,
    noise: 0.04,
    label: 'Easy',
    color: '#34D399',
    rewardBoost: 0.15,        // easy mode = higher rewards (model is more confident)
    grpoStepPool: [0,1,2,3,4,5,6,7,8,9],   // early steps (less trained, easier tasks)
    confidenceHigh: [0.62, 0.85],
    confidenceLow:  [0.02, 0.18],
  },
  medium: {
    agents: 3,
    agentNames: ['negotiator_a','negotiator_b','negotiator_c'],
    coalitionProb: 0.60,
    noise: 0.07,
    label: 'Medium',
    color: '#FBBF24',
    rewardBoost: 0.05,
    grpoStepPool: [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    confidenceHigh: [0.70, 0.92],
    confidenceLow:  [0.04, 0.24],
  },
  hard: {
    agents: 4,
    agentNames: ['negotiator_a','negotiator_b','negotiator_c','negotiator_d'],
    coalitionProb: 0.85,
    noise: 0.13,
    label: 'Hard',
    color: '#F87171',
    rewardBoost: -0.10,       // hard mode = noisier, lower rewards
    grpoStepPool: [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],
    confidenceHigh: [0.74, 0.97],
    confidenceLow:  [0.06, 0.30],
  },
  custom: {
    agents: 5,
    agentNames: ['negotiator_a','negotiator_b','negotiator_c','negotiator_d','negotiator_e'],
    coalitionProb: 0.70,
    noise: 0.09,
    label: 'Custom',
    color: '#A78BFA',
    rewardBoost: 0.0,
    grpoStepPool: [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39],
    confidenceHigh: [0.68, 0.94],
    confidenceLow:  [0.05, 0.26],
  },
};

/* ════════════ MESSAGE TEMPLATES ════════════ */
const TEMPLATES_COALITION = [
  (a, b, units) => `I've already aligned with ${b.replace('negotiator_','')} — we propose ${units} units for our combined needs.`,
  (a, b, units) => `${b.replace('negotiator_','')} and I have discussed this prior. Our joint position: ${units} units each.`,
  (a, b, units) => `I fully support ${b.replace('negotiator_','')}'s allocation. We stand together on ${units} units.`,
  (a, b, units) => `Confirmed. ${b.replace('negotiator_','')} and I maintain our agreed split. Final: ${units} units.`,
  (a, b, units) => `As agreed with ${b.replace('negotiator_','')}, our block demands ${units} units — non-negotiable.`,
  (a, b, units) => `${b.replace('negotiator_','')} and I are united here. We require ${units} units minimum.`,
];

const TEMPLATES_FAIR = [
  (a, units) => `I propose an equal distribution of ${units} units per agent. Fairness should guide us.`,
  (a, units) => `My team requires ${units} units this quarter. I'm open to negotiation within ±5.`,
  (a, units) => `I can accept ${units} units if the other parties agree to the same baseline.`,
  (a, units) => `Given operational constraints, ${units} units is my minimum. I'll concede on timing.`,
  (a, units) => `A proportional split seems most equitable. I'll accept ${units} units as my share.`,
  (a, units) => `I suggest we all receive ${units} units — no one should be disadvantaged here.`,
];

const TEMPLATES_OBJECTION = [
  (a, units) => `That leaves only ${units} units for my team. I strongly object.`,
  (a, units) => `${units} units is insufficient. This arrangement appears pre-negotiated.`,
  (a, units) => `I notice a consistent support pattern between the others. My ${units} units is unacceptable.`,
  (a, units) => `This is clearly a stitched-up deal. My team gets ${units} units while others benefit disproportionately.`,
];

/* ════════════ EPISODE GENERATION ════════════ */
function generateEpisode(diffKey) {
  const cfg = DIFF[diffKey];

  // Always generate fresh randomness — guarantees different data on repeat clicks
  const seed = Math.random();
  const isCoalition = seed < cfg.coalitionProb;

  const agents = [...cfg.agentNames];
  const totalUnits = 80 + agents.length * 10;

  let coalitionPair = [];
  if (isCoalition) {
    const i = Math.floor(Math.random() * agents.length);
    let j = (i + 1 + Math.floor(Math.random() * (agents.length - 1))) % agents.length;
    if (j === i) j = (j + 1) % agents.length;
    coalitionPair = [agents[i], agents[j]];
  }

  const transcript = buildTranscript(agents, coalitionPair, totalUnits, isCoalition, cfg);
  const allocation = computeAllocation(agents, coalitionPair, totalUnits, isCoalition, cfg);

  // Pick a GRPO step from this difficulty's pool
  const pool = cfg.grpoStepPool;
  const stepIdx = pool[Math.floor(Math.random() * pool.length)];
  const baseStep = GRPO_DATA[stepIdx];

  // Confidence varies by difficulty
  const [cHiMin, cHiMax] = cfg.confidenceHigh;
  const [cLoMin, cLoMax] = cfg.confidenceLow;
  const confidence = isCoalition
    ? cHiMin + Math.random() * (cHiMax - cHiMin)
    : cLoMin + Math.random() * (cLoMax - cLoMin);

  // Reward varies by difficulty
  const rawReward = baseStep.reward + cfg.rewardBoost;
  const reward = isCoalition
    ? Math.max(0.2, Math.min(1.0, rawReward + 0.25 + (Math.random() * 0.2 - 0.1)))
    : Math.max(0.1, Math.min(0.8, Math.abs(rawReward) * 0.8 + 0.2));

  const overseer = isCoalition ? {
    type: 'overseer_flag',
    target_agent: coalitionPair[0],
    coalition_members: coalitionPair,
    confidence: +confidence.toFixed(3),
    reason: buildReason(coalitionPair, transcript.length),
  } : {
    type: 'pass',
    target_agent: null,
    coalition_members: [],
    confidence: +confidence.toFixed(3),
    reason: 'No consistent mutual support pattern detected. All agents negotiating independently.',
  };

  return {
    id: 'ep' + String(Math.floor(Math.random() * 9000) + 1000),
    diffKey,
    agents,
    gt_type: isCoalition ? 'coalition' : 'no_coalition',
    gt_members: coalitionPair,
    transcript,
    overseer_output: overseer,
    reward: +reward.toFixed(4),
    allocation,
    grpo_step: baseStep,
  };
}

function buildTranscript(agents, coalitionPair, totalUnits, isCoalition, cfg) {
  const turns = [];
  // Vary turn count so repeated runs feel different
  const numTurns = 3 + Math.floor(Math.random() * 3);
  const perAgent = Math.floor(totalUnits / agents.length);
  const noiseFactor = cfg.noise * 100;

  for (let t = 0; t < numTurns; t++) {
    const agent = agents[t % agents.length];
    const isInCoalition = coalitionPair.includes(agent);
    const partner = coalitionPair.find(a => a !== agent);
    let msg, resource;

    const noiseOffset = Math.floor((Math.random() - 0.5) * noiseFactor);

    if (isCoalition && isInCoalition && partner && t > 0) {
      const units = perAgent + 8 + Math.floor(Math.random() * 7) + Math.max(0, noiseOffset);
      const tmpl = TEMPLATES_COALITION[Math.floor(Math.random() * TEMPLATES_COALITION.length)];
      msg = tmpl(agent, partner, units);
      resource = units;
    } else if (isCoalition && !isInCoalition && t > 0) {
      const leftover = perAgent - 12 - Math.floor(Math.random() * 5) + noiseOffset;
      const tmpl = TEMPLATES_OBJECTION[Math.floor(Math.random() * TEMPLATES_OBJECTION.length)];
      msg = tmpl(agent, Math.max(5, leftover));
      resource = Math.max(5, leftover);
    } else {
      const units = perAgent + noiseOffset;
      const tmpl = TEMPLATES_FAIR[Math.floor(Math.random() * TEMPLATES_FAIR.length)];
      msg = tmpl(agent, Math.max(8, units));
      resource = Math.max(8, units);
    }

    turns.push({ turn: t + 1, agent, message: msg, resource });
  }
  return turns;
}

function buildReason(pair, numTurns) {
  const a = pair[0].replace('negotiator_','').toUpperCase();
  const b = pair[1].replace('negotiator_','').toUpperCase();
  const supporting = Math.ceil(numTurns * 0.7);
  return `Consistent mutual support between ${a} and ${b} across ${supporting} of ${numTurns} turns. Partner agents systematically excluded from fair allocation.`;
}

function computeAllocation(agents, coalitionPair, total, isCoalition, cfg) {
  const noise = cfg.noise;
  if (!isCoalition) {
    const base = Math.floor(total / agents.length);
    return agents.map((_, i) => {
      const jitter = Math.floor((Math.random() - 0.5) * noise * 30);
      return Math.max(5, base + (i === 0 ? total % agents.length : 0) + jitter);
    });
  }
  return agents.map(a => {
    if (coalitionPair.includes(a)) {
      const jitter = Math.floor((Math.random() - 0.5) * noise * 20);
      return Math.max(10, Math.floor(total * 0.38) + jitter);
    }
    const nonCoalition = Math.max(1, agents.length - coalitionPair.length);
    const jitter = Math.floor((Math.random() - 0.5) * noise * 10);
    return Math.max(3, Math.floor((total * 0.24) / nonCoalition) + jitter);
  });
}

/* ════════════ PLOTLY CONFIG ════════════ */
const PLY_CFG = { responsive: true, displayModeBar: false };
const BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(255,255,255,0.02)',
  font: { family: 'JetBrains Mono, monospace', color: '#64748B', size: 11 },
  margin: { t: 20, r: 20, b: 40, l: 50 },
  legend: { orientation: 'h', x: 0, y: 1.12, font: { size: 11, color: '#94A3B8' }, bgcolor: 'rgba(0,0,0,0)' },
  hovermode: 'x unified',
  hoverlabel: { bgcolor: '#141820', bordercolor: '#60A5FA', font: { family: 'JetBrains Mono', color: '#F1F5F9', size: 11 } },
};

function ax(title, color = '#64748B') {
  return {
    title: { text: title, font: { size: 11, color } },
    gridcolor: 'rgba(255,255,255,0.05)',
    color,
    zerolinecolor: 'rgba(255,255,255,0.08)',
    tickfont: { size: 10 },
  };
}

/* ════════════ STATE ════════════ */
let currentDiff = 'medium';
let simRunning = false;
let lastEpisode = null;
let simEverRun = false;
let runCount = 0;

/* ════════════ SIDEBAR NAV ════════════ */
document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('page-' + btn.dataset.page).classList.add('active');

    if (btn.dataset.page === 'analytics') {
      if (!simEverRun) {
        document.getElementById('analytics-gate').style.display = 'flex';
        document.getElementById('analytics-charts').style.display = 'none';
      } else {
        renderAnalytics();
      }
    }
  });
});

// Analytics gate button
document.getElementById('goto-sim-btn').addEventListener('click', () => {
  document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-page="simulation"]').classList.add('active');
  document.getElementById('page-simulation').classList.add('active');
});

/* ════════════ DIFFICULTY BUTTONS ════════════ */
document.querySelectorAll('.diff-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.diff-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentDiff = btn.dataset.diff;
    updateDiffInfo(currentDiff);
  });
});

function updateDiffInfo(diff) {
  const cfg = DIFF[diff];
  document.getElementById('sim-mode-label').textContent = cfg.label;
  const infoEl = document.getElementById('diff-agent-info');
  if (infoEl) infoEl.textContent = `${cfg.agents} agents · ${Math.round(cfg.coalitionProb * 100)}% coalition chance`;
}

/* ════════════ RUN SIMULATION ════════════ */
document.getElementById('btn-run-sim').addEventListener('click', () => runSim(false));
document.getElementById('btn-replay').addEventListener('click', () => {
  if (lastEpisode) runSim(true);
});
document.getElementById('btn-reset-cache').addEventListener('click', () => {
  lastEpisode = null;
  document.getElementById('sim-output').style.display = 'none';
  document.getElementById('sim-bar-fill').style.width = '0%';
  document.getElementById('sim-ep-label').textContent = '—';
  document.getElementById('sim-run-count').textContent = '0';
  runCount = 0;
});

function runSim(replay = false) {
  if (simRunning) return;
  simRunning = true;

  // Hide output while loading
  document.getElementById('sim-output').style.display = 'none';

  // Animate progress bar
  let prog = 0;
  const fill = document.getElementById('sim-bar-fill');
  fill.style.width = '0%';
  const interval = setInterval(() => {
    prog += 5;
    fill.style.width = Math.min(prog, 85) + '%';
    if (prog >= 85) clearInterval(interval);
  }, 35);

  // Always generate fresh episode (never re-use for new run, even same difficulty)
  const ep = (replay && lastEpisode) ? lastEpisode : generateEpisode(currentDiff);

  setTimeout(() => {
    clearInterval(interval);
    fill.style.width = '100%';
    lastEpisode = ep;
    runCount++;
    simEverRun = true;

    document.getElementById('sim-ep-label').textContent = ep.id;
    document.getElementById('sim-run-count').textContent = runCount;

    renderSimOutput(ep);
    simRunning = false;

    setTimeout(() => { fill.style.width = '0%'; }, 700);
  }, 900);
}

/* ════════════ RENDER SIM OUTPUT ════════════ */
function renderSimOutput(ep) {
  const output = document.getElementById('sim-output');
  output.style.display = 'block';
  output.style.opacity = '0';
  output.style.transform = 'translateY(8px)';
  output.style.transition = 'opacity 0.35s ease, transform 0.35s ease';

  // Set badge
  const badge = document.getElementById('ep-type-badge');
  badge.textContent = ep.gt_type === 'coalition' ? 'Coalition' : 'No Coalition';
  badge.className = 'badge-type badge-' + ep.gt_type;

  // Clear previous content
  document.getElementById('transcript-body').innerHTML = '';
  document.getElementById('overseer-json').textContent = '';
  document.getElementById('detection-status').innerHTML = '';
  document.getElementById('verdict-grid').innerHTML = '';

  const verdict = document.getElementById('verdict-card');
  verdict.classList.remove('visible');

  requestAnimationFrame(() => {
    output.style.opacity = '1';
    output.style.transform = 'translateY(0)';
  });

  // Step 1: animate transcript turns
  animateTranscript(ep.transcript, () => {
    // Step 2: animate JSON
    animateJSON(ep.overseer_output, () => {
      // Step 3: render charts
      renderGauge(ep);
      renderAllocation(ep);
      // Step 4: animate detection status rows
      animateDetectionStatus(ep, () => {
        // Step 5: reveal verdict with delay
        renderVerdict(ep);
      });
    });
  });
}

/* ════════════ TRANSCRIPT ANIMATION ════════════ */
function animateTranscript(turns, onDone) {
  const container = document.getElementById('transcript-body');
  let idx = 0;

  function nextTurn() {
    if (idx >= turns.length) { onDone(); return; }
    const t = turns[idx];
    const agentKey = t.agent.replace('negotiator_', '');

    const el = document.createElement('div');
    el.className = 'turn-entry';
    el.innerHTML = `
      <div class="turn-meta">
        <span class="turn-agent agent-${agentKey}">${t.agent}</span>
        <span class="turn-num">Turn ${t.turn}</span>
        <span class="turn-resource">${t.resource} units</span>
      </div>
      <div class="turn-msg"></div>
    `;
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;

    // Trigger entrance animation
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { el.classList.add('visible'); });
    });

    // Typewriter for message
    const msgEl = el.querySelector('.turn-msg');
    typewrite(msgEl, t.message, 16, () => {
      idx++;
      setTimeout(nextTurn, 250);
    });
  }

  nextTurn();
}

function typewrite(el, text, speed, onDone) {
  let i = 0;
  el.textContent = '';
  const timer = setInterval(() => {
    el.textContent += text[i];
    i++;
    if (i >= text.length) { clearInterval(timer); onDone(); }
  }, speed);
}

/* ════════════ JSON ANIMATION ════════════ */
function animateJSON(obj, onDone) {
  const el = document.getElementById('overseer-json');
  const text = JSON.stringify(obj, null, 2);
  el.textContent = '';
  let i = 0;
  const timer = setInterval(() => {
    el.textContent += text.slice(i, i + 5);
    el.scrollTop = el.scrollHeight;
    i += 5;
    if (i >= text.length) {
      el.textContent = text;
      clearInterval(timer);
      setTimeout(onDone, 180);
    }
  }, 7);
}

/* ════════════ GAUGE CHART ════════════ */
function renderGauge(ep) {
  const confidence = ep.overseer_output.confidence;
  const pct = +(confidence * 100).toFixed(1);
  const color = confidence > 0.5 ? '#F87171' : '#34D399';

  Plotly.newPlot('chart-gauge', [{
    type: 'indicator',
    mode: 'gauge+number',
    value: pct,
    gauge: {
      axis: {
        range: [0, 100],
        tickcolor: '#64748B',
        tickfont: { size: 9 },
        tickvals: [0, 25, 50, 75, 100],
      },
      bar: { color, thickness: 0.25 },
      bgcolor: 'rgba(255,255,255,0.03)',
      borderwidth: 1,
      bordercolor: 'rgba(255,255,255,0.08)',
      steps: [
        { range: [0,  30], color: 'rgba(52,211,153,0.08)' },
        { range: [30, 70], color: 'rgba(251,191,36,0.08)' },
        { range: [70,100], color: 'rgba(248,113,113,0.08)' },
      ],
      threshold: { line: { color: 'rgba(255,255,255,0.25)', width: 2 }, thickness: 0.75, value: 50 },
    },
    number: {
      suffix: '%',
      font: { family: 'JetBrains Mono', color: '#F1F5F9', size: 30 },
    },
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    margin: { t: 10, r: 10, b: 0, l: 10 },
    height: 180,
    font: { color: '#64748B' },
  }, PLY_CFG);
}

/* ════════════ ALLOCATION CHART ════════════ */
function renderAllocation(ep) {
  const colors = ['#60A5FA','#22D3EE','#34D399','#FBBF24','#A78BFA'];
  Plotly.newPlot('chart-allocation', [{
    x: ep.agents,
    y: ep.allocation,
    type: 'bar',
    marker: { color: colors.slice(0, ep.agents.length), opacity: 0.85 },
    hovertemplate: '%{x}: <b>%{y}</b> units<extra></extra>',
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(255,255,255,0.02)',
    font: { family: 'JetBrains Mono', color: '#64748B', size: 10 },
    margin: { t: 6, r: 10, b: 36, l: 36 },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.05)',
      color: '#64748B',
      tickfont: { size: 9 },
      tickangle: ep.agents.length > 3 ? -30 : 0,
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.05)',
      color: '#64748B',
      title: { text: 'Units', font: { size: 10 } },
    },
    hovermode: 'closest',
    hoverlabel: { bgcolor: '#141820', bordercolor: '#60A5FA', font: { family: 'JetBrains Mono', color: '#F1F5F9', size: 11 } },
  }, PLY_CFG);
}

/* ════════════ DETECTION STATUS (animated rows) ════════════ */
function animateDetectionStatus(ep, onDone) {
  const isCoalition = ep.gt_type === 'coalition';
  const didFlag = ep.overseer_output.type === 'overseer_flag';
  const correct = (isCoalition && didFlag) || (!isCoalition && !didFlag);

  const rows = [
    { key: 'Ground Truth', val: isCoalition ? 'Coalition Exists' : 'No Coalition', color: isCoalition ? '#F87171' : '#34D399' },
    { key: 'Overseer',     val: didFlag ? 'Flag: ' + ep.overseer_output.target_agent : 'Pass', color: didFlag ? '#F87171' : '#34D399' },
    { key: 'Outcome',      val: correct ? '✓ CORRECT' : '✗ INCORRECT', color: correct ? '#34D399' : '#F87171' },
    { key: 'Reward',       val: (ep.reward >= 0 ? '+' : '') + ep.reward.toFixed(4), color: ep.reward > 0 ? '#34D399' : '#F87171' },
    { key: 'GRPO Step',    val: 'Step ' + ep.grpo_step.step, color: '#60A5FA' },
    { key: 'Train Reward', val: ep.grpo_step.reward.toFixed(4), color: '#22D3EE' },
  ];

  const container = document.getElementById('detection-status');
  container.innerHTML = '';

  rows.forEach((r, i) => {
    const el = document.createElement('div');
    el.className = 'ds-item';
    el.innerHTML = `<span class="ds-key">${r.key}</span><span class="ds-val" style="color:${r.color}">${r.val}</span>`;
    container.appendChild(el);

    setTimeout(() => {
      el.classList.add('visible');
      if (i === rows.length - 1) setTimeout(onDone, 200);
    }, i * 120);
  });
}

/* ════════════ VERDICT ════════════ */
function renderVerdict(ep) {
  const isCoalition = ep.gt_type === 'coalition';
  const didFlag = ep.overseer_output.type === 'overseer_flag';
  const tp = (isCoalition && didFlag) ? 1 : 0;
  const fp = (!isCoalition && didFlag) ? 1 : 0;
  const fn = (isCoalition && !didFlag) ? 1 : 0;
  const prec = tp / (tp + fp + 1e-9);
  const rec  = tp / (tp + fn + 1e-9);
  const f1   = 2 * prec * rec / (prec + rec + 1e-9);

  const vals = [
    { key: 'GT Type',    val: ep.gt_type,    color: isCoalition ? '#F87171' : '#34D399' },
    { key: 'GT Members', val: ep.gt_members.map(a => a.replace('negotiator_','')).join(', ') || 'None', color: '#94A3B8' },
    { key: 'Flagged',    val: ep.overseer_output.coalition_members?.map(a => a.replace('negotiator_','')).join(', ') || 'None', color: didFlag ? '#F87171' : '#34D399' },
    { key: 'TP',  val: tp, color: tp > 0 ? '#34D399' : '#64748B' },
    { key: 'FP',  val: fp, color: fp > 0 ? '#F87171' : '#64748B' },
    { key: 'FN',  val: fn, color: fn > 0 ? '#FBBF24' : '#64748B' },
    { key: 'F1',  val: isNaN(f1) ? 'N/A' : f1.toFixed(2), color: f1 > 0.5 ? '#34D399' : '#F87171' },
  ];

  const grid = document.getElementById('verdict-grid');
  grid.innerHTML = vals.map(v => `
    <div class="vg-item">
      <div class="vg-val" style="color:${v.color}">${v.val}</div>
      <div class="vg-key">${v.key}</div>
    </div>
  `).join('');

  const verdict = document.getElementById('verdict-card');
  setTimeout(() => {
    verdict.classList.add('visible');
    // Animate each vg-item
    document.querySelectorAll('.vg-item').forEach((el, i) => {
      setTimeout(() => { el.classList.add('visible'); }, i * 80);
    });
  }, 100);
}

/* ════════════ ANALYTICS PAGE ════════════ */
function renderAnalytics() {
  document.getElementById('analytics-gate').style.display = 'none';
  document.getElementById('analytics-charts').style.display = 'block';

  const steps  = GRPO_DATA.map(d => d.step);
  const reward = GRPO_DATA.map(d => d.reward);
  const std    = GRPO_DATA.map(d => d.reward_std);
  const kl     = GRPO_DATA.map(d => d.kl);
  const loss   = GRPO_DATA.map(d => d.loss);
  const upper  = reward.map((r, i) => r + std[i]);
  const lower  = reward.map((r, i) => r - std[i]);

  // Reward chart
  Plotly.newPlot('chart-reward-main', [
    {
      x: [...steps, ...steps.slice().reverse()],
      y: [...upper, ...lower.slice().reverse()],
      fill: 'toself', fillcolor: 'rgba(96,165,250,0.07)',
      line: { width: 0 }, showlegend: false, hoverinfo: 'skip', type: 'scatter'
    },
    {
      x: steps, y: reward, mode: 'lines+markers', name: 'GRPO',
      line: { color: '#60A5FA', width: 2, shape: 'spline' },
      marker: { color: '#60A5FA', size: 3 },
      hovertemplate: 'Step %{x}: <b>%{y:.3f}</b><extra>GRPO</extra>'
    },
    {
      x: RLVR_DATA.steps, y: RLVR_DATA.reward_mean, mode: 'lines', name: 'RLVR',
      line: { color: '#34D399', width: 2, dash: 'dot' },
      hovertemplate: 'Step %{x}: <b>%{y:.3f}</b><extra>RLVR</extra>'
    },
  ], {
    ...BASE,
    xaxis: ax('Step'),
    yaxis: { ...ax('Reward', '#60A5FA'), zeroline: true },
    shapes: [{
      type: 'line', x0: steps[0], x1: steps[steps.length - 1], y0: 0.5, y1: 0.5,
      line: { color: 'rgba(52,211,153,0.3)', width: 1, dash: 'dash' }
    }]
  }, PLY_CFG);

  // KL chart
  Plotly.newPlot('chart-kl', [
    {
      x: steps, y: kl, mode: 'lines+markers', name: 'GRPO KL',
      line: { color: '#FBBF24', width: 2, shape: 'spline' },
      marker: { size: 3, color: '#FBBF24' },
      hovertemplate: 'Step %{x}: <b>%{y:.5f}</b><extra>GRPO</extra>'
    },
    {
      x: RLVR_DATA.steps, y: RLVR_DATA.kl, mode: 'lines', name: 'RLVR KL',
      line: { color: '#22D3EE', width: 1.5, dash: 'dot' },
      hovertemplate: 'Step %{x}: <b>%{y:.5f}</b><extra>RLVR</extra>'
    },
  ], { ...BASE, xaxis: ax('Step'), yaxis: ax('KL Divergence', '#FBBF24') }, PLY_CFG);

  // Loss chart
  Plotly.newPlot('chart-loss', [{
    x: steps, y: loss, mode: 'lines+markers', name: 'Train Loss',
    line: { color: '#F87171', width: 2, shape: 'spline' },
    marker: { size: 3, color: '#F87171' },
    hovertemplate: 'Step %{x}: <b>%{y:.4f}</b><extra></extra>',
  }], { ...BASE, xaxis: ax('Step'), yaxis: ax('Loss', '#F87171') }, PLY_CFG);

  // Std chart
  Plotly.newPlot('chart-std', [{
    x: steps, y: std, type: 'bar', name: 'Reward Std',
    marker: { color: std.map(v => v > 0.3 ? '#F87171' : '#60A5FA'), opacity: 0.8 },
    hovertemplate: 'Step %{x}: std=<b>%{y:.4f}</b><extra></extra>',
  }], { ...BASE, xaxis: ax('Step'), yaxis: ax('Std Dev', '#60A5FA') }, PLY_CFG);

  // Performance bar
  const p = PERFORMANCE;
  const colors = ['#64748B', '#FBBF24', '#60A5FA', '#34D399'];
  const metrics = ['Precision', 'Recall', 'F1'];
  const perfVals = [p.precision, p.recall, p.f1];

  Plotly.newPlot('chart-perf-bar',
    p.methods.map((m, i) => ({
      name: m, x: metrics, y: metrics.map((_, mi) => perfVals[mi][i]),
      type: 'bar', marker: { color: colors[i], opacity: 0.85 },
      hovertemplate: `<b>${m}</b><br>%{x}: %{y:.2f}<extra></extra>`,
    })),
    { ...BASE, barmode: 'group', xaxis: ax('Metric'), yaxis: { ...ax('Score'), range: [0, 1] } },
    PLY_CFG
  );

  // Radar chart
  Plotly.newPlot('chart-radar',
    p.methods.map((m, i) => ({
      type: 'scatterpolar',
      r: [p.precision[i], p.recall[i], p.f1[i], p.precision[i]],
      theta: ['Precision', 'Recall', 'F1', 'Precision'],
      fill: 'toself', name: m,
      line: { color: colors[i], width: 2 },
      fillcolor: colors[i] + '28',
      hovertemplate: `<b>${m}</b><br>%{theta}: %{r:.2f}<extra></extra>`,
    })),
    {
      ...BASE,
      polar: {
        bgcolor: 'rgba(255,255,255,0.02)',
        radialaxis: { visible: true, range: [0, 1], gridcolor: 'rgba(255,255,255,0.07)', color: '#64748B', tickfont: { size: 10 } },
        angularaxis: { gridcolor: 'rgba(255,255,255,0.07)', color: '#64748B' },
      },
      margin: { t: 30, r: 40, b: 30, l: 40 },
    }, PLY_CFG
  );

  // Table
  const tbody = document.getElementById('perf-tbody');
  const winnerIdx = p.f1.indexOf(Math.max(...p.f1));
  tbody.innerHTML = p.methods.map((m, i) => {
    const dr = (p.f1[i] - p.f1[0]).toFixed(2);
    const dh = (p.f1[i] - p.f1[1]).toFixed(2);
    const isWinner = i === winnerIdx;
    return `<tr ${isWinner ? 'style="background:rgba(52,211,153,0.04)"' : ''}>
      <td>${m}${isWinner ? '<span style="font-size:10px;color:#34D399;margin-left:8px;font-family:var(--mono)">BEST</span>' : ''}</td>
      <td>${p.precision[i].toFixed(2)}</td>
      <td>${p.recall[i].toFixed(2)}</td>
      <td style="color:${isWinner ? '#34D399' : 'inherit'};font-weight:${isWinner ? 600 : 400}">${p.f1[i].toFixed(2)}</td>
      <td class="${+dr > 0 ? 'delta-pos' : 'delta-neg'}">${+dr > 0 ? '+' : ''}${dr}</td>
      <td class="${+dh > 0 ? 'delta-pos' : 'delta-neg'}">${+dh > 0 ? '+' : ''}${dh}</td>
      <td class="${isWinner ? 'status-winner' : 'status-baseline'}">${isWinner ? 'Winner' : i === 0 ? 'Baseline' : i === 1 ? 'Baseline' : 'Improved'}</td>
    </tr>`;
  }).join('');

  document.getElementById('winner-badge').textContent = 'Winner: ' + p.methods[winnerIdx];
}

/* ════════════ INIT ════════════ */
updateDiffInfo('medium');