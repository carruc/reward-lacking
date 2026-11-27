"use strict";

(() => {
  const CLUSTER_PALETTE = [
    "#4f9cf9", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
    "#fb7185", "#22d3ee", "#a3e635", "#f97316", "#60a5fa",
    "#e879f9", "#2dd4bf", "#facc15", "#818cf8", "#f87171",
    "#38bdf8", "#4ade80", "#fb923c", "#c084fc", "#f43f5e",
    "#10b981", "#eab308", "#6366f1", "#ec4899", "#14b8a6",
    "#84cc16", "#ef4444", "#06b6d4", "#d946ef", "#3b82f6",
  ];
  const NOISE_COLOR = "#64748b";
  const OUT_SCOPE_COLOR = "#2c3340";

  const STATUS_STYLE = [
    { name: "clean", color: "#38bdf8" },
    { name: "alien", color: "#f43f5e" },
    { name: "overexposed_duplicate", color: "#f59e0b" },
    { name: "blank", color: "#94a3b8" },
    { name: "test", color: "#a78bfa" },
  ];

  const ERROR_COLORS = {
    correct: "#22c55e",
    wrong: "#ef4444",
    test: "#475569",
  };

  const ALIEN_COLORS = {
    alien: "#f43f5e",
    other: "#475569",
  };

  const SPLIT_OPTIONS = [
    ["all", "All"],
    ["train", "Train"],
    ["clean", "Clean"],
    ["dropped", "Dropped"],
    ["test", "Test"],
  ];

  const $ = (s) => document.querySelector(s);

  const App = () => window.GalleryApp;

  const state = {
    initialized: false,
    loading: false,
    meta: null,
    samples: null,
    idToIndex: new Map(),
    spaceOrder: [],
    spaceCache: new Map(),      // spaceId -> view payload
    activeSpaceId: null,
    activeView: null,
    colorBy: "cluster",
    splitFilter: "all",
    classFilter: "all",
    tool: "pan",                // "pan" | "lasso"
    showLabels: true,
    showThumbs: true,
    ptSize: 4,
    clusterSort: "size",
    legendCollapsed: false,

    // Camera (world coords in [-1, 1])
    zoom: 1.0,
    panX: 0,
    panY: 0,
    camAnim: null,

    // Render coords (interpolated for smooth space morph)
    renderX: null,
    renderY: null,
    morphAnim: null,

    // Spatial hash
    gridN: 64,
    gridCells: null,

    // Hover & Selection
    hoverIndex: -1,
    pinnedIndex: -1,
    pinnedProbs: null,
    pulseIndex: -1,
    inspectorMode: null,        // null | "cluster" | "lasso" | "sample"
    selectedClusterId: null,
    selectedIndices: [],        // ordered indices shown in inspector
    selectedSet: new Set(),
    neighborSims: new Map(),    // idx -> sim
    inspRendered: 0,
    inspPage: 60,
    inspScrollBound: false,

    // Cross-model overlap
    overlap: null,
    ovCkaTab: "linear_all",
    ovKnnTab: "raw:20",
  };

  const spriteCache = new Map();
  const thumbCache = new Map();
  const MAX_THUMBS = 256;
  let rafDrawPending = false;
  let pointer = {
    down: false,
    moved: false,
    mode: "pan",
    startX: 0,
    startY: 0,
    lastX: 0,
    lastY: 0,
    lassoScreen: [],
  };

  function rampColor(t, stops) {
    const clamped = Math.max(0, Math.min(1, t));
    const scaled = clamped * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(scaled));
    const frac = scaled - i;
    const a = stops[i];
    const b = stops[i + 1];
    const r = Math.round(a[0] + (b[0] - a[0]) * frac);
    const g = Math.round(a[1] + (b[1] - a[1]) * frac);
    const bl = Math.round(a[2] + (b[2] - a[2]) * frac);
    return `rgb(${r},${g},${bl})`;
  }

  const CONF_STOPS = [
    [225, 29, 72],
    [245, 158, 11],
    [34, 197, 94],
    [56, 189, 248],
  ];
  const SKY_STOPS = [
    [37, 99, 235],
    [56, 189, 248],
    [250, 204, 21],
    [239, 68, 68],
  ];

  const CONF_PALETTE = Array.from({ length: 64 }, (_, i) => rampColor(i / 63, CONF_STOPS));
  const SKY_PALETTE = Array.from({ length: 64 }, (_, i) => rampColor(i / 63, SKY_STOPS));
  const OV_PALETTE = CONF_PALETTE;

  function getSprite(color) {
    let sp = spriteCache.get(color);
    if (sp) return sp;
    const c = document.createElement("canvas");
    c.width = 32;
    c.height = 32;
    const g = c.getContext("2d");
    g.fillStyle = color;
    g.beginPath();
    g.arc(16, 16, 13.5, 0, Math.PI * 2);
    g.fill();
    spriteCache.set(color, c);
    return c;
  }

  function clusterColor(cid) {
    if (cid === -1) return NOISE_COLOR;
    if (cid < -1) return OUT_SCOPE_COLOR;
    return CLUSTER_PALETTE[cid % CLUSTER_PALETTE.length];
  }

  function pointColor(i) {
    const view = state.activeView;
    const samples = state.samples;
    if (!view || !samples) return NOISE_COLOR;

    switch (state.colorBy) {
      case "cluster":
        return clusterColor(view.cluster[i]);
      case "true_class": {
        const lbl = samples.label[i];
        return lbl === null ? "#3b4352" : App().color(lbl);
      }
      case "pred_class":
        return App().color(view.pred[i]);
      case "status":
        return (STATUS_STYLE[samples.status[i]] || STATUS_STYLE[4]).color;
      case "conf": {
        const idx = Math.min(63, Math.max(0, Math.floor(view.conf[i] * 63.99)));
        return CONF_PALETTE[idx];
      }
      case "error": {
        const lbl = samples.label[i];
        if (lbl === null) return ERROR_COLORS.test;
        return view.pred[i] === lbl ? ERROR_COLORS.correct : ERROR_COLORS.wrong;
      }
      case "sky_med": {
        const t = (samples.sky_med[i] - 12) / 100;
        const idx = Math.min(63, Math.max(0, Math.floor(Math.max(0, Math.min(1, t)) * 63.99)));
        return SKY_PALETTE[idx];
      }
      case "tag_alien": {
        const id = samples.ids[i];
        return App().hasTag(id, "alien") ? ALIEN_COLORS.alien : ALIEN_COLORS.other;
      }
      case "overlap": {
        const arr = state.overlap && state.overlap.per_sample_k20 && state.overlap.per_sample_k20.mean_pairwise;
        if (!arr) return NOISE_COLOR;
        const t = Math.max(0, Math.min(1, arr[i] / 0.75));
        const idx = Math.min(63, Math.max(0, Math.floor(t * 63.99)));
        return OV_PALETTE[idx];
      }
      default:
        return NOISE_COLOR;
    }
  }

  function isForeground(i) {
    const samples = state.samples;
    const view = state.activeView;
    if (!samples || !view) return false;
    const st = samples.status[i];
    if (state.splitFilter === "train" && st === 4) return false;
    if (state.splitFilter === "clean" && st !== 0) return false;
    if (state.splitFilter === "dropped" && (st === 0 || st === 4)) return false;
    if (state.splitFilter === "test" && st !== 4) return false;

    if (state.classFilter !== "all") {
      const target = Number(state.classFilter);
      const lbl = samples.label[i];
      if (lbl !== null) {
        if (lbl !== target) return false;
      } else if (view.pred[i] !== target) {
        return false;
      }
    }
    return true;
  }

  function sampleObj(i) {
    return {
      id: state.samples.ids[i],
      split: state.samples.split[i],
      label: state.samples.label[i],
    };
  }

  function sampleThumbUrl(i) {
    return App().imgSrc(sampleObj(i));
  }

  function getCachedThumb(i) {
    const id = state.samples.ids[i];
    const hit = thumbCache.get(id);
    if (hit) {
      thumbCache.delete(id);
      thumbCache.set(id, hit);
      return hit;
    }
    if (thumbCache.size >= MAX_THUMBS) {
      const oldest = thumbCache.keys().next().value;
      thumbCache.delete(oldest);
    }
    const img = new Image();
    const rec = { img, status: "loading" };
    thumbCache.set(id, rec);
    img.onload = () => {
      rec.status = "ready";
      scheduleRender();
    };
    img.onerror = () => {
      rec.status = "err";
    };
    img.src = sampleThumbUrl(i);
    return rec;
  }

  /* ---------- geometry & spatial hash ---------- */
  function stageSize() {
    const stage = $("#cl-stage");
    return {
      w: Math.max(100, stage.clientWidth),
      h: Math.max(100, stage.clientHeight),
    };
  }

  function worldToScreen(wx, wy, w, h) {
    const base = 0.44 * Math.min(w, h);
    return [
      w * 0.5 + state.panX + wx * base * state.zoom,
      h * 0.5 + state.panY - wy * base * state.zoom,
    ];
  }

  function screenToWorld(sx, sy, w, h) {
    const base = 0.44 * Math.min(w, h);
    const s = base * state.zoom;
    return [
      (sx - w * 0.5 - state.panX) / s,
      -(sy - h * 0.5 - state.panY) / s,
    ];
  }

  function rebuildSpatialGrid() {
    const n = state.samples ? state.samples.ids.length : 0;
    const g = state.gridN;
    const cells = Array.from({ length: g * g }, () => []);
    if (state.renderX && state.renderY) {
      for (let i = 0; i < n; i++) {
        const gx = Math.min(g - 1, Math.max(0, Math.floor(((state.renderX[i] + 1.05) / 2.1) * g)));
        const gy = Math.min(g - 1, Math.max(0, Math.floor(((state.renderY[i] + 1.05) / 2.1) * g)));
        cells[gy * g + gx].push(i);
      }
    }
    state.gridCells = cells;
  }

  function hitTest(sx, sy) {
    if (!state.gridCells || !state.renderX || !state.renderY) return -1;
    const { w, h } = stageSize();
    const base = 0.44 * Math.min(w, h);
    const pxPerWorld = base * state.zoom;
    const [wx, wy] = screenToWorld(sx, sy, w, h);
    const maxPx = Math.max(11, state.ptSize + 7);
    const rWorld = maxPx / pxPerWorld;
    const g = state.gridN;

    const minGx = Math.max(0, Math.floor(((wx - rWorld + 1.05) / 2.1) * g));
    const maxGx = Math.min(g - 1, Math.floor(((wx + rWorld + 1.05) / 2.1) * g));
    const minGy = Math.max(0, Math.floor(((wy - rWorld + 1.05) / 2.1) * g));
    const maxGy = Math.min(g - 1, Math.floor(((wy + rWorld + 1.05) / 2.1) * g));

    let bestFg = -1;
    let bestFgDist = Infinity;
    let bestBg = -1;
    let bestBgDist = Infinity;

    for (let gy = minGy; gy <= maxGy; gy++) {
      for (let gx = minGx; gx <= maxGx; gx++) {
        const bucket = state.gridCells[gy * g + gx];
        for (let k = 0; k < bucket.length; k++) {
          const idx = bucket[k];
          const dx = (state.renderX[idx] - wx) * pxPerWorld;
          const dy = (state.renderY[idx] - wy) * pxPerWorld;
          const d2 = dx * dx + dy * dy;
          if (d2 > maxPx * maxPx) continue;
          if (isForeground(idx)) {
            if (d2 < bestFgDist) {
              bestFgDist = d2;
              bestFg = idx;
            }
          } else if (d2 < bestBgDist) {
            bestBgDist = d2;
            bestBg = idx;
          }
        }
      }
    }
    return bestFg >= 0 ? bestFg : bestBg;
  }

  /* ---------- canvas rendering ---------- */
  function scheduleRender() {
    if (rafDrawPending) return;
    rafDrawPending = true;
    requestAnimationFrame(() => {
      rafDrawPending = false;
      drawCanvas();
    });
  }

  function drawCanvas() {
    const view = state.activeView;
    const samples = state.samples;
    const canvas = $("#cl-canvas");
    if (!canvas || !view || !samples || !state.renderX) return;

    const { w, h } = stageSize();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const targetW = Math.round(w * dpr);
    const targetH = Math.round(h * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const n = samples.ids.length;
    const rx = state.renderX;
    const ry = state.renderY;
    const base = 0.44 * Math.min(w, h);
    const s = base * state.zoom;
    const ox = w * 0.5 + state.panX;
    const oy = h * 0.5 + state.panY;
    const r = state.ptSize * Math.min(1.85, 0.9 + Math.log2(Math.max(1, state.zoom)) * 0.22);
    const hasSel = state.selectedSet.size > 0;
    const margin = 48;

    const fgList = [];
    const selList = [];
    const inViewportFg = [];
    let fgTotal = 0;

    // 1. Dimmed background points
    ctx.globalAlpha = hasSel ? 0.14 : 0.16;
    for (let i = 0; i < n; i++) {
      const sx = ox + rx[i] * s;
      const sy = oy - ry[i] * s;
      if (sx < -margin || sx > w + margin || sy < -margin || sy > h + margin) continue;
      const fg = isForeground(i);
      const isSel = hasSel && state.selectedSet.has(i);
      if (fg) fgTotal++;
      if (!fg || (hasSel && !isSel)) {
        const sp = getSprite(pointColor(i));
        ctx.drawImage(sp, sx - r * 0.85, sy - r * 0.85, r * 1.7, r * 1.7);
      } else {
        fgList.push(i, sx, sy);
        if (isSel) selList.push(i, sx, sy);
        if (sx >= 0 && sx <= w && sy >= 0 && sy <= h) inViewportFg.push(i, sx, sy);
      }
    }

    // 2. Foreground points
    const useThumbs = state.showThumbs && state.zoom >= 2.2 && inViewportFg.length > 0 && inViewportFg.length <= 3 * 120;
    ctx.globalAlpha = 0.9;
    if (!useThumbs) {
      for (let k = 0; k < fgList.length; k += 3) {
        const idx = fgList[k];
        const sx = fgList[k + 1];
        const sy = fgList[k + 2];
        const sp = getSprite(pointColor(idx));
        ctx.drawImage(sp, sx - r, sy - r, r * 2, r * 2);
      }
    } else {
      // Draw dots as fallback first, then thumbnails cleanly on top
      for (let k = 0; k < fgList.length; k += 3) {
        const idx = fgList[k];
        const sx = fgList[k + 1];
        const sy = fgList[k + 2];
        const sp = getSprite(pointColor(idx));
        ctx.drawImage(sp, sx - r, sy - r, r * 2, r * 2);
      }
      ctx.globalAlpha = 1.0;
      const box = Math.min(64, Math.max(24, Math.round(state.zoom * 5.2)));
      const half = box * 0.5;
      for (let k = 0; k < inViewportFg.length; k += 3) {
        const idx = inViewportFg[k];
        const sx = inViewportFg[k + 1];
        const sy = inViewportFg[k + 2];
        const rec = getCachedThumb(idx);
        if (rec.status === "ready") {
          ctx.save();
          ctx.beginPath();
          ctx.roundRect(sx - half, sy - half, box, box, 4);
          ctx.clip();
          ctx.drawImage(rec.img, sx - half, sy - half, box, box);
          ctx.restore();
          ctx.strokeStyle = pointColor(idx);
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.roundRect(sx - half, sy - half, box, box, 4);
          ctx.stroke();
        }
      }
    }

    // 3. Selection rings (if small selection or pinned)
    if (selList.length > 0 && selList.length <= 3 * 400 && !useThumbs) {
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.1;
      for (let k = 0; k < selList.length; k += 3) {
        ctx.beginPath();
        ctx.arc(selList[k + 1], selList[k + 2], r + 1.4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // 4. Cluster centroid labels
    if (state.showLabels && !state.morphAnim && view.clusters) {
      ctx.globalAlpha = 1.0;
      ctx.font = "600 11px system-ui, -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const cl of view.clusters) {
        if (cl.id < 0) continue;
        const [sx, sy] = worldToScreen(cl.cx, cl.cy, w, h);
        if (sx < -40 || sx > w + 40 || sy < -20 || sy > h + 20) continue;
        const text = `#${cl.id}`;
        const tw = ctx.measureText(text).width + 10;
        const isAct = state.selectedClusterId === cl.id;
        ctx.fillStyle = isAct ? "rgba(79,156,249,0.92)" : "rgba(11,13,16,0.78)";
        ctx.strokeStyle = isAct ? "#ffffff" : clusterColor(cl.id);
        ctx.lineWidth = isAct ? 1.5 : 1.0;
        ctx.beginPath();
        ctx.roundRect(sx - tw * 0.5, sy - 9, tw, 18, 9);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#f8fafc";
        ctx.fillText(text, sx, sy + 0.5);
      }
    }

    // 5. Pinned, pulsed, and hovered point reticles
    ctx.globalAlpha = 1.0;
    const drawReticle = (idx, ringColor, radiusAdd, lineW) => {
      if (idx < 0 || idx >= n) return;
      const sx = ox + rx[idx] * s;
      const sy = oy - ry[idx] * s;
      ctx.strokeStyle = ringColor;
      ctx.lineWidth = lineW;
      ctx.beginPath();
      ctx.arc(sx, sy, r + radiusAdd, 0, Math.PI * 2);
      ctx.stroke();
    };
    drawReticle(state.pinnedIndex, "#ffffff", 4.5, 2.2);
    drawReticle(state.pulseIndex, "#facc15", 6.5, 2.2);
    drawReticle(state.hoverIndex, "#4f9cf9", 4.0, 2.0);

    // 6. Active Lasso polygon
    if (pointer.lassoScreen.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(pointer.lassoScreen[0][0], pointer.lassoScreen[0][1]);
      for (let k = 1; k < pointer.lassoScreen.length; k++) {
        ctx.lineTo(pointer.lassoScreen[k][0], pointer.lassoScreen[k][1]);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(79, 156, 249, 0.14)";
      ctx.fill();
      ctx.strokeStyle = "#4f9cf9";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.restore();
    }

    updateHud(fgTotal, inViewportFg.length / 3);
  }

  function updateHud(fgInBounds, inViewCount) {
    const hud = $("#cl-hud");
    if (!hud || !state.activeView || !state.samples) return;
    const sp = state.meta.spaces.find((x) => x.id === state.activeSpaceId);
    const total = state.samples.ids.length;
    const selMsg = state.selectedSet.size > 0 ? `<span><strong>${state.selectedSet.size.toLocaleString()}</strong> selected</span>` : "";
    hud.innerHTML =
      `<span><strong>${App().esc(sp ? sp.name : state.activeSpaceId)}</strong></span>` +
      `<span><strong>${Math.round(inViewCount).toLocaleString()}</strong> in view / ${total.toLocaleString()}</span>` +
      `<span>zoom <strong>${state.zoom.toFixed(1)}×</strong></span>` +
      selMsg +
      `<span>Drag: pan · Wheel: zoom · Shift+Drag: lasso · Click: inspect</span>`;
  }

  /* ---------- animations ---------- */
  function startMorph(targetX, targetY) {
    const n = targetX.length;
    if (!state.renderX || state.renderX.length !== n) {
      state.renderX = Float32Array.from(targetX);
      state.renderY = Float32Array.from(targetY);
      rebuildSpatialGrid();
      scheduleRender();
      return;
    }
    const fromX = state.renderX.slice();
    const fromY = state.renderY.slice();
    const t0 = performance.now();
    const dur = 350;

    if (state.morphAnim) cancelAnimationFrame(state.morphAnim);
    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      for (let i = 0; i < n; i++) {
        state.renderX[i] = fromX[i] + (targetX[i] - fromX[i]) * e;
        state.renderY[i] = fromY[i] + (targetY[i] - fromY[i]) * e;
      }
      drawCanvas();
      if (u < 1) {
        state.morphAnim = requestAnimationFrame(step);
      } else {
        state.morphAnim = null;
        rebuildSpatialGrid();
        drawCanvas();
      }
    };
    state.morphAnim = requestAnimationFrame(step);
  }

  function animateCamera(targetZoom, targetPanX, targetPanY) {
    const z0 = state.zoom;
    const px0 = state.panX;
    const py0 = state.panY;
    const t0 = performance.now();
    const dur = 260;
    if (state.camAnim) cancelAnimationFrame(state.camAnim);

    const step = (now) => {
      const u = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - u, 3);
      state.zoom = z0 + (targetZoom - z0) * e;
      state.panX = px0 + (targetPanX - px0) * e;
      state.panY = py0 + (targetPanY - py0) * e;
      drawCanvas();
      if (u < 1) state.camAnim = requestAnimationFrame(step);
      else state.camAnim = null;
    };
    state.camAnim = requestAnimationFrame(step);
  }

  function zoomToIndices(indices) {
    if (!indices || !indices.length || !state.renderX) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const idx of indices) {
      const x = state.renderX[idx];
      const y = state.renderY[idx];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const { w, h } = stageSize();
    const base = 0.44 * Math.min(w, h);
    const spanX = Math.max(0.06, maxX - minX);
    const spanY = Math.max(0.06, maxY - minY);
    const targetZoom = Math.max(1.0, Math.min(14.0, 0.72 * Math.min(w / (spanX * base), h / (spanY * base))));
    const cx = 0.5 * (minX + maxX);
    const cy = 0.5 * (minY + maxY);
    animateCamera(targetZoom, -cx * base * targetZoom, cy * base * targetZoom);
  }

  /* ---------- UI builders ---------- */
  function buildToolbarStatic() {
    const spDiv = $("#cl-space-pills");
    spDiv.innerHTML = "";
    for (const sp of state.meta.spaces) {
      const b = document.createElement("button");
      b.className = "cl-chip";
      b.dataset.space = sp.id;
      const wf1 = sp.val_wf1 !== null && sp.val_wf1 !== undefined ? `<small>${sp.val_wf1.toFixed(3)} WF1</small>` : "";
      b.innerHTML = `${App().esc(sp.name)} ${wf1}`;
      b.title = sp.subtitle || sp.id;
      b.onclick = () => selectSpace(sp.id);
      spDiv.appendChild(b);
    }

    const splitDiv = $("#cl-split-pills");
    splitDiv.innerHTML = "";
    for (const [val, label] of SPLIT_OPTIONS) {
      const b = document.createElement("button");
      b.className = "cl-chip" + (state.splitFilter === val ? " on" : "");
      b.dataset.split = val;
      b.textContent = label;
      b.onclick = () => {
        state.splitFilter = val;
        splitDiv.querySelectorAll(".cl-chip").forEach((c) => c.classList.toggle("on", c.dataset.split === val));
        scheduleRender();
      };
      splitDiv.appendChild(b);
    }

    const clsSel = $("#cl-class-filter");
    clsSel.innerHTML = `<option value="all">All classes</option>`;
    App().CLASSES.forEach((name, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = `${idx} · ${name}`;
      clsSel.appendChild(opt);
    });
  }

  function renderLegend() {
    const leg = $("#cl-legend");
    if (!leg || !state.activeView) return;
    const labels = {
      cluster: "HDBSCAN Cluster",
      true_class: "True Class",
      pred_class: "Predicted Class",
      status: "Split / Status",
      conf: "Confidence",
      error: "Prediction Errors",
      sky_med: "Sky Median",
      tag_alien: "Tag: alien",
      overlap: "k-NN@20 Overlap",
    };
    let bodyHtml = "";
    if (!state.legendCollapsed) {
      if (state.colorBy === "cluster") {
        const cls = state.activeView.clusters || [];
        const top = cls.slice(0, 14);
        bodyHtml = `<div class="cl-legend-items">` +
          top.map((c) => {
            const name = c.id === -1 ? "Noise (-1)" : `#${c.id} · ${App().CLASSES[c.dominant_pred] ?? ""}`;
            return `<div class="cl-legend-item"><span class="cl-swatch" style="background:${clusterColor(c.id)}"></span><span>${App().esc(name)} (${c.count})</span></div>`;
          }).join("") +
          (cls.length > 14 ? `<div class="cl-legend-item" style="color:var(--dim)">+${cls.length - 14} more in sidebar</div>` : "") +
          `</div>`;
      } else if (state.colorBy === "true_class" || state.colorBy === "pred_class") {
        bodyHtml = `<div class="cl-legend-items">` +
          App().CLASSES.map((n, idx) =>
            `<div class="cl-legend-item"><span class="cl-swatch" style="background:${App().color(idx)}"></span><span>${idx} ${App().esc(n)}</span></div>`
          ).join("") + `</div>`;
      } else if (state.colorBy === "status") {
        bodyHtml = `<div class="cl-legend-items">` +
          STATUS_STYLE.map((s) =>
            `<div class="cl-legend-item"><span class="cl-swatch" style="background:${s.color}"></span><span>${App().esc(s.name)}</span></div>`
          ).join("") + `</div>`;
      } else if (state.colorBy === "error") {
        bodyHtml = `<div class="cl-legend-items">
          <div class="cl-legend-item"><span class="cl-swatch" style="background:${ERROR_COLORS.correct}"></span><span>Train correct</span></div>
          <div class="cl-legend-item"><span class="cl-swatch" style="background:${ERROR_COLORS.wrong}"></span><span>Train misclassified</span></div>
          <div class="cl-legend-item"><span class="cl-swatch" style="background:${ERROR_COLORS.test}"></span><span>Test (unlabelled)</span></div>
        </div>`;
      } else if (state.colorBy === "tag_alien") {
        bodyHtml = `<div class="cl-legend-items">
          <div class="cl-legend-item"><span class="cl-swatch" style="background:${ALIEN_COLORS.alien}"></span><span>Tagged: alien</span></div>
          <div class="cl-legend-item"><span class="cl-swatch" style="background:${ALIEN_COLORS.other}"></span><span>Not tagged alien</span></div>
        </div>`;
      } else if (state.colorBy === "conf") {
        bodyHtml = `<div class="cl-legend-ramp">
          <div class="cl-ramp-bar" style="background:linear-gradient(90deg, #e11d48, #f59e0b, #22c55e, #38bdf8)"></div>
          <div class="cl-ramp-labels"><span>0.1</span><span>0.5</span><span>1.0</span></div>
        </div>`;
      } else if (state.colorBy === "sky_med") {
        bodyHtml = `<div class="cl-legend-ramp">
          <div class="cl-ramp-bar" style="background:linear-gradient(90deg, #2563eb, #38bdf8, #facc15, #ef4444)"></div>
          <div class="cl-ramp-labels"><span>dark (12)</span><span>33</span><span>bright (112+)</span></div>
        </div>`;
      } else if (state.colorBy === "overlap") {
        bodyHtml = `<div class="cl-legend-ramp">
          <div class="cl-ramp-bar" style="background:linear-gradient(90deg, #e11d48, #f59e0b, #22c55e, #38bdf8)"></div>
          <div class="cl-ramp-labels"><span>0%</span><span>35%</span><span>75%+</span></div>
        </div>`;
      }
    }
    leg.innerHTML =
      `<div class="cl-legend-head" id="cl-legend-toggle">
         <span>${App().esc(labels[state.colorBy] || "Legend")}</span>
         <span>${state.legendCollapsed ? "+" : "−"}</span>
       </div>${bodyHtml}`;
    $("#cl-legend-toggle").onclick = () => {
      state.legendCollapsed = !state.legendCollapsed;
      renderLegend();
    };
  }

  function renderClusterSidebar() {
    const view = state.activeView;
    if (!view) return;
    const clusters = [...(view.clusters || [])];
    const nPos = clusters.filter((c) => c.id >= 0).length;
    const noiseObj = clusters.find((c) => c.id === -1);
    const noiseCount = noiseObj ? noiseObj.count : 0;
    $("#cl-side-stats").innerHTML = `<strong>${nPos}</strong> clusters · ${noiseCount.toLocaleString()} noise`;

    const sortKey = state.clusterSort;
    clusters.sort((a, b) => {
      if (a.id === -1 && b.id !== -1) return 1;
      if (b.id === -1 && a.id !== -1) return -1;
      if (sortKey === "dropped") return (b.dropped_count - a.dropped_count) || (b.count - a.count);
      if (sortKey === "test_frac") return (b.test_count / b.count - a.test_count / a.count) || (b.count - a.count);
      if (sortKey === "purity_asc") return ((a.purity ?? 2) - (b.purity ?? 2)) || (b.count - a.count);
      if (sortKey === "conf_asc") return (a.mean_conf - b.mean_conf) || (b.count - a.count);
      return b.count - a.count;
    });

    const list = $("#cl-cluster-list");
    list.innerHTML = "";
    const esc = App().esc;
    for (const c of clusters) {
      const card = document.createElement("div");
      card.className = "cl-card" + (state.inspectorMode === "cluster" && state.selectedClusterId === c.id ? " active" : "");
      card.dataset.cid = String(c.id);

      const title = c.id === -1 ? "Noise (-1)" : `Cluster #${c.id}`;
      const domCls = c.dominant_class !== null ? c.dominant_class : c.dominant_pred;
      const domName = App().CLASSES[domCls] ?? `Class ${domCls}`;
      const purityStr = c.purity !== null ? `${Math.round(c.purity * 100)}%` : "test-only";
      const testPct = Math.round((100 * c.test_count) / Math.max(1, c.count));

      const badges = [];
      if (c.alien_count > 0) badges.push(`<span class="cl-badge-warn">${c.alien_count} alien</span>`);
      if (c.overexposed_count > 0) badges.push(`<span class="cl-badge-amber">${c.overexposed_count} overexp</span>`);
      if (c.blank_count > 0) badges.push(`<span class="cl-badge-amber">${c.blank_count} blank</span>`);

      const cleanW = (100 * c.clean_count) / Math.max(1, c.count);
      const dropW = (100 * c.dropped_count) / Math.max(1, c.count);
      const testW = Math.max(0, 100 - cleanW - dropW);

      const exImgs = (c.exemplars || []).slice(0, 4).map((idx) =>
        `<img loading="lazy" decoding="async" src="${sampleThumbUrl(idx)}" alt="${esc(state.samples.ids[idx])}">`
      ).join("");

      card.innerHTML =
        `<div class="cl-card-top">
           <span class="cl-swatch" style="background:${clusterColor(c.id)}"></span>
           <span>${esc(title)}</span>
           <span class="cl-card-count">${c.count.toLocaleString()}</span>
         </div>
         <div class="cl-card-meta">
           <span class="dom">${c.dominant_class !== null ? `C${domCls}` : `Pred C${domCls}`} ${esc(domName)}</span>
           <span>${purityStr}</span>
           <span>${testPct}% test</span>
           <span>conf ${(c.mean_conf * 100).toFixed(0)}%</span>
           ${badges.join("")}
         </div>
         <div class="cl-mix-bar" title="clean: ${c.clean_count}, dropped: ${c.dropped_count}, test: ${c.test_count}">
           <div class="cl-mix-clean" style="width:${cleanW.toFixed(1)}%"></div>
           <div class="cl-mix-dropped" style="width:${dropW.toFixed(1)}%"></div>
           <div class="cl-mix-test" style="width:${testW.toFixed(1)}%"></div>
         </div>
         <div class="cl-exemplars">${exImgs}</div>`;

      card.onclick = () => selectCluster(c.id);
      card.ondblclick = () => {
        selectCluster(c.id);
        zoomToIndices(state.selectedIndices);
      };
      list.appendChild(card);
    }
  }

  /* ---------- hover card ---------- */
  function updateHoverCard(idx, sx, sy) {
    const card = $("#cl-hover");
    if (idx < 0 || !state.activeView || !state.samples) {
      card.hidden = true;
      return;
    }
    const { w, h } = stageSize();
    const cw = 244;
    const ch = 205;
    const left = sx + 18 + cw > w ? Math.max(8, sx - cw - 14) : sx + 16;
    const top = sy + 16 + ch > h ? Math.max(8, sy - ch - 10) : sy + 14;
    card.style.left = left + "px";
    card.style.top = top + "px";

    const id = state.samples.ids[idx];
    const st = STATUS_STYLE[state.samples.status[idx]] || STATUS_STYLE[4];
    const lbl = state.samples.label[idx];
    const pred = state.activeView.pred[idx];
    const conf = state.activeView.conf[idx];
    const cid = state.activeView.cluster[idx];
    const cprob = state.activeView.cluster_prob[idx];
    const sky = state.samples.sky_med[idx];
    const esc = App().esc;

    $("#cl-hover-img").src = sampleThumbUrl(idx);
    $("#cl-hover-id").textContent = id;
    const badge = $("#cl-hover-status");
    badge.textContent = st.name;
    badge.style.color = st.color;
    badge.style.borderColor = st.color;

    const trueLine = lbl !== null
      ? `True: <strong style="color:${App().color(lbl)}">${lbl} · ${esc(App().CLASSES[lbl])}</strong><br>`
      : "";
    const clText = cid === -1 ? "Noise (-1)" : cid < -1 ? "Out of scope" : `#${cid} (p=${cprob.toFixed(2)})`;
    const ov = state.overlap && state.overlap.per_sample_k20;
    const ovLine = ov
      ? `<br>kNN@20 overlap: <strong>${(ov.mean_pairwise[idx] * 100).toFixed(1)}%</strong> ` +
        `(E↔C ${(ov.eff_conv[idx] * 100).toFixed(0)}%, E↔D ${(ov.eff_dino[idx] * 100).toFixed(0)}%, C↔D ${(ov.conv_dino[idx] * 100).toFixed(0)}%)`
      : "";
    $("#cl-hover-rows").innerHTML =
      `${trueLine}` +
      `Pred: <strong style="color:${App().color(pred)}">${pred} · ${esc(App().CLASSES[pred])}</strong> (${(conf * 100).toFixed(1)}%)<br>` +
      `Cluster: <strong>${esc(clText)}</strong> · sky ${sky}${ovLine}`;

    const tags = App().annotations[id] || [];
    $("#cl-hover-tags").innerHTML = tags.map((t) => `<span class="tile-tag">${esc(t)}</span>`).join("");
    card.hidden = false;
  }

  /* ---------- selection & inspector ---------- */
  function clearSelection() {
    state.inspectorMode = null;
    state.selectedClusterId = null;
    state.selectedIndices = [];
    state.selectedSet.clear();
    state.pinnedIndex = -1;
    state.pinnedProbs = null;
    state.pulseIndex = -1;
    state.neighborSims.clear();
    $("#cl-inspector").hidden = true;
    renderClusterSidebar();
    scheduleRender();
  }

  function selectCluster(cid) {
    if (!state.activeView) return;
    const indices = [];
    const clArr = state.activeView.cluster;
    for (let i = 0; i < clArr.length; i++) {
      if (clArr[i] === cid) indices.push(i);
    }
    state.inspectorMode = "cluster";
    state.selectedClusterId = cid;
    state.pinnedIndex = -1;
    state.neighborSims.clear();
    state.selectedIndices = indices;
    state.selectedSet = new Set(indices);
    const title = cid === -1 ? "Noise (-1)" : `Cluster #${cid}`;
    openInspector(title, `${indices.length.toLocaleString()} samples in ${activeSpaceName()}`);
    renderClusterSidebar();
    scheduleRender();
  }

  function selectLasso(indices) {
    state.inspectorMode = "lasso";
    state.selectedClusterId = null;
    state.pinnedIndex = -1;
    state.neighborSims.clear();
    state.selectedIndices = indices;
    state.selectedSet = new Set(indices);
    openInspector(`Lasso Selection`, `${indices.length.toLocaleString()} samples in ${activeSpaceName()}`);
    renderClusterSidebar();
    scheduleRender();
  }

  async function selectSample(idx) {
    if (!state.activeView || idx < 0) return;
    state.inspectorMode = "sample";
    state.selectedClusterId = null;
    state.pinnedIndex = idx;
    state.pinnedProbs = null;
    state.neighborSims.clear();
    state.selectedIndices = [idx];
    state.selectedSet = new Set([idx]);
    openInspector(`Sample ${state.samples.ids[idx]}`, `Fetching PCA-50 nearest neighbors…`);
    renderClusterSidebar();
    scheduleRender();

    try {
      const spaceAtReq = state.activeSpaceId;
      const url = `/api/clusters/neighbors?space=${encodeURIComponent(spaceAtReq)}&index=${idx}&k=48`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (state.pinnedIndex !== idx || state.inspectorMode !== "sample" || state.activeSpaceId !== spaceAtReq) return;

      const neighIdx = [];
      for (const item of data.neighbors) {
        neighIdx.push(item.index);
        state.neighborSims.set(item.index, item.sim);
      }
      state.pinnedProbs = data.probs;
      state.selectedIndices = neighIdx;
      state.selectedSet = new Set([idx, ...neighIdx]);
      renderSampleFocusHeader(idx, data.probs);
      $("#cl-insp-sub").textContent = `48 nearest neighbors in ${activeSpaceName()} (PCA-50 cosine)`;
      sortAndRenderInspectorGrid();
      scheduleRender();
    } catch (err) {
      console.error(err);
      $("#cl-insp-sub").textContent = `Could not load nearest neighbors (${err.message})`;
    }
  }

  function activeSpaceName() {
    const sp = state.meta && state.meta.spaces.find((x) => x.id === state.activeSpaceId);
    return sp ? sp.name : state.activeSpaceId;
  }

  function renderSegBar(hist, total) {
    if (!total) return `<div class="cl-seg-bar"></div>`;
    let html = `<div class="cl-seg-bar">`;
    for (let c = 0; c < 10; c++) {
      if (!hist[c]) continue;
      const pct = (100 * hist[c]) / total;
      html += `<div style="width:${pct.toFixed(2)}%;background:${App().color(c)}" title="Class ${c} ${App().esc(App().CLASSES[c])}: ${hist[c]} (${pct.toFixed(1)}%)"></div>`;
    }
    html += `</div>`;
    return html;
  }

  function renderSampleFocusHeader(idx, probs) {
    const esc = App().esc;
    const id = state.samples.ids[idx];
    const st = STATUS_STYLE[state.samples.status[idx]] || STATUS_STYLE[4];
    const lbl = state.samples.label[idx];
    const pred = state.activeView.pred[idx];
    const conf = state.activeView.conf[idx];
    const cid = state.activeView.cluster[idx];
    const sky = state.samples.sky_med[idx];
    const tags = App().annotations[id] || [];
    const isAlien = App().hasTag(id, "alien");
    const ov = state.overlap && state.overlap.per_sample_k20;
    const ovRow = ov
      ? `<div>kNN@20 overlap: <strong>${(ov.mean_pairwise[idx] * 100).toFixed(1)}%</strong> ` +
        `(3-way ${(ov.three_way[idx] * 100).toFixed(1)}%)</div>`
      : "";

    const topProbs = probs
      ? probs.map((p, c) => ({ c, p })).sort((a, b) => b.p - a.p).slice(0, 5)
      : [];

    const summary = $("#cl-insp-summary");
    summary.innerHTML =
      `<div class="cl-sample-focus">
         <div class="cl-sample-hero">
           <img id="cl-focus-img" src="${sampleThumbUrl(idx)}" alt="${esc(id)}" title="Click to open in Lightbox">
           <div class="cl-sample-meta">
             <div><strong>${esc(id)}</strong> · <span style="color:${st.color}">${esc(st.name)}</span></div>
             ${lbl !== null ? `<div>True: <strong style="color:${App().color(lbl)}">${lbl} ${esc(App().CLASSES[lbl])}</strong></div>` : ""}
             <div>Pred: <strong style="color:${App().color(pred)}">${pred} ${esc(App().CLASSES[pred])}</strong> (${(conf * 100).toFixed(1)}%)</div>
             <div>Cluster: <strong>${cid === -1 ? "Noise (-1)" : `#${cid}`}</strong> · sky ${sky}</div>
             ${ovRow}
             <div>Tags: <strong>${tags.length ? tags.map(esc).join(", ") : "none"}</strong></div>
           </div>
         </div>
         <div class="cl-prob-bars">
           ${topProbs.map(({ c, p }) =>
             `<div class="cl-prob-row">
                <span class="cl-prob-name">${c} ${esc(App().CLASSES[c])}</span>
                <div class="cl-prob-track"><div class="cl-prob-fill" style="width:${(p * 100).toFixed(1)}%;background:${App().color(c)}"></div></div>
                <span class="cl-prob-val">${(p * 100).toFixed(1)}%</span>
              </div>`
           ).join("")}
         </div>
         <div class="cl-insp-actions">
           <button class="cl-chip ${isAlien ? "on" : ""}" id="cl-sample-alien">${isAlien ? "✓ alien" : "+ alien"}</button>
           <button class="cl-chip" id="cl-sample-edit">Edit tags</button>
           <button class="cl-chip" id="cl-insp-gallery">Neighbors → Gallery</button>
         </div>
       </div>`;

    $("#cl-focus-img").onclick = () => {
      const items = [sampleObj(idx), ...state.selectedIndices.map(sampleObj)];
      App().openLightbox(0, items, inspectorLightboxMeta);
    };
    $("#cl-sample-alien").onclick = () => {
      App().toggleQuickTag(id);
      renderSampleFocusHeader(idx, probs);
    };
    $("#cl-sample-edit").onclick = () => App().openTagEditor(id);
    $("#cl-insp-gallery").onclick = () => {
      const ids = [id, ...state.selectedIndices.map((k) => state.samples.ids[k])];
      App().openSubsetInGallery(ids, `Neighbors of ${id} (${ids.length})`);
    };
  }

  function renderGroupSummary() {
    const indices = state.selectedIndices;
    const samples = state.samples;
    const view = state.activeView;
    const trueHist = new Array(10).fill(0);
    const predHist = new Array(10).fill(0);
    let clean = 0, alien = 0, over = 0, blank = 0, test = 0, train = 0, confSum = 0;

    for (const i of indices) {
      const st = samples.status[i];
      if (st === 0) clean++;
      else if (st === 1) alien++;
      else if (st === 2) over++;
      else if (st === 3) blank++;
      else test++;
      if (st !== 4) train++;
      if (samples.label[i] !== null) trueHist[samples.label[i]]++;
      predHist[view.pred[i]]++;
      confSum += view.conf[i];
    }
    const meanConf = indices.length ? (confSum / indices.length) * 100 : 0;

    const pills = [
      `<span><strong>${clean}</strong> clean</span>`,
      `<span><strong>${test}</strong> test</span>`,
      alien ? `<span style="color:#ff9f9f"><strong>${alien}</strong> alien</span>` : "",
      over ? `<span style="color:#fcd34d"><strong>${over}</strong> overexp</span>` : "",
      blank ? `<span style="color:#fcd34d"><strong>${blank}</strong> blank</span>` : "",
      `<span>conf <strong>${meanConf.toFixed(0)}%</strong></span>`,
    ].filter(Boolean).map((x) => `<span class="cl-stat-pill">${x}</span>`).join("");

    $("#cl-insp-summary").innerHTML =
      `<div class="cl-stat-pills">${pills}</div>
       ${train > 0 ? `<div class="cl-dist-block"><div class="cl-dist-title"><span>True classes (train)</span><span>${train}</span></div>${renderSegBar(trueHist, train)}</div>` : ""}
       <div class="cl-dist-block"><div class="cl-dist-title"><span>Predicted classes (all)</span><span>${indices.length}</span></div>${renderSegBar(predHist, indices.length)}</div>
       <div class="cl-insp-actions">
         <button class="cl-chip on" id="cl-insp-gallery">Open in Gallery</button>
         <button class="cl-chip" id="cl-insp-zoom">Zoom to</button>
         <input type="text" id="cl-insp-tag" value="alien" placeholder="tag" spellcheck="false">
         <button class="cl-chip" id="cl-insp-apply-tag">Tag all</button>
       </div>`;

    $("#cl-insp-gallery").onclick = () => {
      const ids = state.selectedIndices.map((k) => state.samples.ids[k]);
      const lbl = state.inspectorMode === "cluster"
        ? `Cluster #${state.selectedClusterId} (${ids.length})`
        : `Lasso (${ids.length})`;
      App().openSubsetInGallery(ids, lbl);
    };
    $("#cl-insp-zoom").onclick = () => zoomToIndices(state.selectedIndices);
    $("#cl-insp-apply-tag").onclick = () => {
      const tag = ($("#cl-insp-tag").value || "").trim().toLowerCase();
      if (!tag) return;
      let added = 0;
      for (const i of state.selectedIndices) {
        const id = state.samples.ids[i];
        const tags = App().tagsOf(id);
        if (!tags.includes(tag)) {
          tags.push(tag);
          added++;
        }
      }
      App().scheduleSave();
      App().setStatus(added > 0 ? `applied '${tag}' to ${added} samples` : `all already tagged '${tag}'`);
      refreshInspectorTilesOnly();
      scheduleRender();
    };
  }

  function openInspector(title, subtitle) {
    $("#cl-insp-title").textContent = title;
    $("#cl-insp-sub").textContent = subtitle;
    $("#cl-inspector").hidden = false;
    if (state.inspectorMode === "sample") {
      renderSampleFocusHeader(state.pinnedIndex, null);
    } else {
      renderGroupSummary();
    }
    sortAndRenderInspectorGrid();
  }

  function sortAndRenderInspectorGrid() {
    const mode = $("#cl-insp-sort").value;
    const view = state.activeView;
    const samples = state.samples;
    const arr = [...state.selectedIndices];

    if (state.inspectorMode === "sample" && mode === "rep") {
      arr.sort((a, b) => (state.neighborSims.get(b) || 0) - (state.neighborSims.get(a) || 0));
    } else if (mode === "rep" && state.inspectorMode === "cluster") {
      const cl = (view.clusters || []).find((c) => c.id === state.selectedClusterId);
      const cx = cl ? cl.cx : 0;
      const cy = cl ? cl.cy : 0;
      arr.sort((a, b) => {
        const da = (view.x[a] - cx) ** 2 + (view.y[a] - cy) ** 2;
        const db = (view.x[b] - cx) ** 2 + (view.y[b] - cy) ** 2;
        return da - db;
      });
    } else if (mode === "conf_asc") {
      arr.sort((a, b) => view.conf[a] - view.conf[b]);
    } else if (mode === "conf_desc") {
      arr.sort((a, b) => view.conf[b] - view.conf[a]);
    } else if (mode === "errors") {
      arr.sort((a, b) => {
        const ea = samples.label[a] !== null && view.pred[a] !== samples.label[a] ? 0 : samples.label[a] !== null ? 1 : 2;
        const eb = samples.label[b] !== null && view.pred[b] !== samples.label[b] ? 0 : samples.label[b] !== null ? 1 : 2;
        return ea - eb || view.conf[a] - view.conf[b];
      });
    } else if (mode === "test_first") {
      arr.sort((a, b) => (samples.status[a] === 4 ? 0 : 1) - (samples.status[b] === 4 ? 0 : 1) || view.conf[a] - view.conf[b]);
    } else if (mode === "train_first") {
      arr.sort((a, b) => (samples.status[a] === 4 ? 1 : 0) - (samples.status[b] === 4 ? 1 : 0) || a - b);
    } else {
      arr.sort((a, b) => samples.ids[a].localeCompare(samples.ids[b]));
    }

    state.selectedIndices = arr;
    state.inspRendered = 0;
    $("#cl-insp-grid").innerHTML = "";
    $("#cl-insp-grid-count").textContent = `${arr.length.toLocaleString()} items`;
    renderInspectorChunk();
  }

  function renderInspectorChunk() {
    const grid = $("#cl-insp-grid");
    const arr = state.selectedIndices;
    const end = Math.min(state.inspRendered + state.inspPage, arr.length);
    const esc = App().esc;
    let html = "";

    for (; state.inspRendered < end; state.inspRendered++) {
      const idx = arr[state.inspRendered];
      const id = state.samples.ids[idx];
      const sp = state.samples.split[idx];
      const lbl = state.samples.label[idx];
      const pred = state.activeView.pred[idx];
      const conf = state.activeView.conf[idx];
      const isAlien = App().hasTag(id, "alien");
      const sim = state.inspectorMode === "sample" ? state.neighborSims.get(idx) : undefined;

      const topLeft = sim !== undefined
        ? `<span class="cl-mini-badge">${sim.toFixed(2)}</span>`
        : `<span class="cl-mini-badge">${Math.round(conf * 100)}%</span>`;
      const topRight = isAlien ? `<span class="cl-mini-badge alien">alien</span>` : "";
      const botLeft = lbl !== null
        ? `<span style="color:${App().color(lbl)};font-weight:700">${lbl}→${pred}</span>`
        : `<span style="color:${App().color(pred)};font-weight:700">p:${pred}</span>`;

      html += `<div class="cl-mini-tile ${sp === "test" ? "test" : ""}" data-idx="${idx}" data-pos="${state.inspRendered}" title="${esc(id)}">
                 <img loading="lazy" decoding="async" src="${sampleThumbUrl(idx)}" alt="${esc(id)}">
                 <div class="cl-mini-top">${topLeft}${topRight}</div>
                 <div class="cl-mini-bot">${botLeft}<span style="color:var(--dim)">${sp === "test" ? "test" : "tr"}</span></div>
               </div>`;
    }
    grid.insertAdjacentHTML("beforeend", html);
  }

  function refreshInspectorTilesOnly() {
    if ($("#cl-inspector").hidden) return;
    if (state.inspectorMode === "sample" && state.pinnedIndex >= 0) {
      // keep existing grid intact if just refreshing badges; re-render visible chunk cleanly
    }
    const prevRendered = state.inspRendered;
    state.inspRendered = 0;
    $("#cl-insp-grid").innerHTML = "";
    const oldPage = state.inspPage;
    state.inspPage = Math.max(oldPage, prevRendered);
    renderInspectorChunk();
    state.inspPage = oldPage;
  }

  function inspectorLightboxMeta(m) {
    const idx = state.idToIndex.get(m.id);
    if (idx === undefined || !state.activeView || !state.samples) return "";
    const pred = state.activeView.pred[idx];
    const conf = state.activeView.conf[idx];
    const cid = state.activeView.cluster[idx];
    const st = STATUS_STYLE[state.samples.status[idx]] || STATUS_STYLE[4];
    const sim = state.inspectorMode === "sample" ? state.neighborSims.get(idx) : undefined;
    const simStr = sim !== undefined ? ` · cos ${sim.toFixed(3)}` : "";
    return `Status: ${App().esc(st.name)} · Cluster: ${cid === -1 ? "Noise" : `#${cid}`} · ` +
           `Pred: ${pred} ${App().esc(App().CLASSES[pred])} (${(conf * 100).toFixed(1)}%)${simStr}`;
  }

  /* ---------- data loading & space switching ---------- */
  async function loadMeta() {
    const overlay = $("#cl-overlay");
    const card = $("#cl-overlay-card");
    overlay.hidden = false;
    card.innerHTML = "Loading cluster metadata…";

    const res = await fetch("/api/clusters/meta");
    if (!res.ok) throw new Error(`HTTP ${res.status} loading /api/clusters/meta`);
    const meta = await res.json();
    if (!meta.available) {
      card.innerHTML =
        `<strong>Cluster embeddings not built yet.</strong><br><br>` +
        `Run <code>uv run build-clusters</code> in the repository root, then reload or click retry.` +
        `<br><br><button class="cl-chip on" id="cl-retry-boot">Retry</button>`;
      $("#cl-retry-boot").onclick = () => bootClusters(true);
      return false;
    }

    state.meta = meta;
    state.samples = meta.samples;
    state.idToIndex = new Map(meta.samples.ids.map((id, i) => [id, i]));
    state.spaceOrder = meta.spaces.map((s) => s.id);
    buildToolbarStatic();
    return true;
  }

  async function selectSpace(spaceId) {
    if (!state.meta) return;
    state.activeSpaceId = spaceId;
    document.querySelectorAll("#cl-space-pills .cl-chip").forEach((b) =>
      b.classList.toggle("on", b.dataset.space === spaceId));

    const overlay = $("#cl-overlay");
    let view = state.spaceCache.get(spaceId);
    if (!view) {
      overlay.hidden = false;
      $("#cl-overlay-card").innerHTML = `Loading <strong>${App().esc(activeSpaceName())}</strong> embeddings…`;
      const res = await fetch(`/api/clusters/space?name=${encodeURIComponent(spaceId)}`);
      if (state.activeSpaceId !== spaceId) return;
      if (!res.ok) {
        $("#cl-overlay-card").innerHTML = `Failed to load space '${App().esc(spaceId)}' (HTTP ${res.status}).`;
        return;
      }
      view = await res.json();
      if (state.activeSpaceId !== spaceId) return;
      state.spaceCache.set(spaceId, view);
    }
    if (state.activeSpaceId !== spaceId) return;
    overlay.hidden = true;
    state.activeView = view;
    syncHdbPopoverFromView(view);
    renderLegend();
    renderClusterSidebar();
    startMorph(view.x, view.y);

    if (state.inspectorMode === "sample" && state.pinnedIndex >= 0) {
      selectSample(state.pinnedIndex);
    } else if (state.inspectorMode === "lasso" && state.selectedIndices.length > 0) {
      openInspector("Lasso Selection", `${state.selectedIndices.length.toLocaleString()} samples in ${activeSpaceName()}`);
    } else if (state.inspectorMode === "cluster") {
      clearSelection();
    }
  }

  function syncHdbPopoverFromView(view) {
    const p = view.hdbscan_params || {};
    $("#cl-hdb-on").value = p.cluster_on || "2d";
    $("#cl-hdb-scope").value = p.scope || "all";
    $("#cl-hdb-mcs").value = p.min_cluster_size ?? 80;
    $("#cl-hdb-ms").value = p.min_samples ?? 15;
    $("#cl-hdb-maxcs").value = p.max_cluster_size ?? 0;
    $("#cl-hdb-method").value = p.method || "eom";
    $("#cl-hdb-status").textContent = "";
  }

  async function runLiveRecluster() {
    if (!state.activeSpaceId || !state.activeView) return;
    const status = $("#cl-hdb-status");
    const btn = $("#cl-hdb-run");
    status.textContent = "Running HDBSCAN…";
    btn.disabled = true;
    try {
      const rawMax = Number($("#cl-hdb-maxcs").value);
      const payload = {
        space: state.activeSpaceId,
        cluster_on: $("#cl-hdb-on").value,
        scope: $("#cl-hdb-scope").value,
        min_cluster_size: Number($("#cl-hdb-mcs").value) || 80,
        min_samples: Number($("#cl-hdb-ms").value) || 15,
        max_cluster_size: Number.isFinite(rawMax) && rawMax > 0 ? rawMax : 0,
        method: $("#cl-hdb-method").value,
        pca_dims: 15,
      };
      const res = await fetch("/api/clusters/recluster", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);

      state.activeView.cluster = data.cluster;
      state.activeView.cluster_prob = data.cluster_prob;
      state.activeView.clusters = data.clusters;
      state.activeView.hdbscan_params = data.hdbscan_params;

      const nPos = data.clusters.filter((c) => c.id >= 0).length;
      status.textContent = `${nPos} clusters in ${data.elapsed_ms} ms`;
      renderLegend();
      renderClusterSidebar();
      if (state.inspectorMode === "cluster") clearSelection();
      scheduleRender();
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    } finally {
      btn.disabled = false;
    }
  }

  /* ---------- interactions ---------- */
  function pointInPoly(px, py, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1];
      const xj = poly[j][0], yj = poly[j][1];
      const intersect = ((yi > py) !== (yj > py)) &&
        (px < ((xj - xi) * (py - yi)) / ((yj - yi) || 1e-12) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function bindStageEvents() {
    const stage = $("#cl-stage");
    const canvas = $("#cl-canvas");

    canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || !state.activeView) return;
      canvas.setPointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      pointer.down = true;
      pointer.moved = false;
      pointer.startX = pointer.lastX = sx;
      pointer.startY = pointer.lastY = sy;
      pointer.mode = (state.tool === "lasso" || e.shiftKey) ? "lasso" : "pan";
      pointer.lassoScreen = pointer.mode === "lasso" ? [[sx, sy]] : [];
      if (pointer.mode === "pan") stage.classList.add("panning");
      $("#cl-hover").hidden = true;
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!state.activeView) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;

      if (!pointer.down) {
        const hit = hitTest(sx, sy);
        if (hit !== state.hoverIndex) {
          state.hoverIndex = hit;
          scheduleRender();
        }
        updateHoverCard(hit, sx, sy);
        return;
      }

      const dx = sx - pointer.lastX;
      const dy = sy - pointer.lastY;
      if (Math.hypot(sx - pointer.startX, sy - pointer.startY) > 5) {
        pointer.moved = true;
      }

      if (pointer.mode === "pan") {
        state.panX += dx;
        state.panY += dy;
        pointer.lastX = sx;
        pointer.lastY = sy;
        scheduleRender();
      } else {
        const prev = pointer.lassoScreen[pointer.lassoScreen.length - 1];
        if (!prev || Math.hypot(sx - prev[0], sy - prev[1]) >= 4) {
          pointer.lassoScreen.push([sx, sy]);
          scheduleRender();
        }
      }
    });

    canvas.addEventListener("pointerup", (e) => {
      if (!pointer.down) return;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const wasLasso = pointer.mode === "lasso";
      const moved = pointer.moved;
      const polyScreen = pointer.lassoScreen;

      pointer.down = false;
      pointer.lassoScreen = [];
      stage.classList.remove("panning");

      if (wasLasso && moved && polyScreen.length >= 3) {
        const { w, h } = stageSize();
        const polyWorld = polyScreen.map(([px, py]) => screenToWorld(px, py, w, h));
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const [wx, wy] of polyWorld) {
          if (wx < minX) minX = wx;
          if (wx > maxX) maxX = wx;
          if (wy < minY) minY = wy;
          if (wy > maxY) maxY = wy;
        }
        const picked = [];
        const rx = state.renderX;
        const ry = state.renderY;
        const n = state.samples.ids.length;
        for (let i = 0; i < n; i++) {
          if (!isForeground(i)) continue;
          const wx = rx[i];
          const wy = ry[i];
          if (wx < minX || wx > maxX || wy < minY || wy > maxY) continue;
          if (pointInPoly(wx, wy, polyWorld)) picked.push(i);
        }
        if (picked.length > 0) selectLasso(picked);
        else scheduleRender();
        return;
      }

      if (!moved) {
        const hit = hitTest(sx, sy);
        if (hit >= 0) selectSample(hit);
        else clearSelection();
      }
    });

    canvas.addEventListener("pointerleave", () => {
      if (state.hoverIndex >= 0) {
        state.hoverIndex = -1;
        $("#cl-hover").hidden = true;
        scheduleRender();
      }
    });

    canvas.addEventListener("wheel", (e) => {
      if (!state.activeView) return;
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const { w, h } = stageSize();
      const base = 0.44 * Math.min(w, h);
      const [wx, wy] = screenToWorld(mx, my, w, h);
      const factor = Math.exp(-e.deltaY * 0.0014);
      const nextZoom = Math.max(0.5, Math.min(45.0, state.zoom * factor));
      state.zoom = nextZoom;
      state.panX = mx - w * 0.5 - wx * base * nextZoom;
      state.panY = my - h * 0.5 + wy * base * nextZoom;
      scheduleRender();
    }, { passive: false });

    canvas.addEventListener("dblclick", (e) => {
      if (!state.activeView) return;
      const hit = hitTest(e.offsetX, e.offsetY);
      if (hit < 0) animateCamera(1.0, 0, 0);
    });

    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(() => scheduleRender()).observe(stage);
    }
  }

  function bindUiControls() {
    $("#cl-color-by").addEventListener("change", (e) => {
      state.colorBy = e.target.value;
      renderLegend();
      scheduleRender();
    });

    $("#cl-class-filter").addEventListener("change", (e) => {
      state.classFilter = e.target.value;
      scheduleRender();
    });

    const setTool = (tool) => {
      state.tool = tool;
      $("#cl-tool-pan").classList.toggle("on", tool === "pan");
      $("#cl-tool-lasso").classList.toggle("on", tool === "lasso");
      $("#cl-stage").classList.toggle("lasso-mode", tool === "lasso");
    };
    $("#cl-tool-pan").onclick = () => setTool("pan");
    $("#cl-tool-lasso").onclick = () => setTool("lasso");

    $("#cl-toggle-labels").onclick = () => {
      state.showLabels = !state.showLabels;
      $("#cl-toggle-labels").classList.toggle("on", state.showLabels);
      scheduleRender();
    };
    $("#cl-toggle-thumbs").onclick = () => {
      state.showThumbs = !state.showThumbs;
      $("#cl-toggle-thumbs").classList.toggle("on", state.showThumbs);
      scheduleRender();
    };
    $("#cl-pt-size").addEventListener("input", (e) => {
      state.ptSize = Number(e.target.value);
      scheduleRender();
    });
    $("#cl-reset-view").onclick = () => animateCamera(1.0, 0, 0);

    $("#cl-toggle-sidebar").onclick = () => {
      const sb = $("#cl-sidebar");
      sb.hidden = !sb.hidden;
      $("#cl-toggle-sidebar").classList.toggle("on", !sb.hidden);
      scheduleRender();
    };
    $("#cl-toggle-sidebar").classList.add("on");

    $("#cl-cluster-sort").addEventListener("change", (e) => {
      state.clusterSort = e.target.value;
      renderClusterSidebar();
    });

    // HDBSCAN popover
    const pop = $("#cl-hdb-popover");
    $("#cl-hdb-btn").onclick = () => {
      pop.hidden = !pop.hidden;
      $("#cl-hdb-btn").classList.toggle("on", !pop.hidden);
    };
    $("#cl-hdb-close").onclick = () => {
      pop.hidden = true;
      $("#cl-hdb-btn").classList.remove("on");
    };
    $("#cl-hdb-default").onclick = () => {
      $("#cl-hdb-on").value = "2d";
      $("#cl-hdb-scope").value = "all";
      $("#cl-hdb-mcs").value = 80;
      $("#cl-hdb-ms").value = 15;
      $("#cl-hdb-maxcs").value = 3000;
      $("#cl-hdb-method").value = "eom";
    };
    $("#cl-hdb-run").onclick = runLiveRecluster;

    // Cross-model overlap modal
    $("#cl-overlap-btn").onclick = openOverlapModal;
    $("#cl-ov-close").onclick = closeOverlapModal;
    $("#cl-ov-done").onclick = closeOverlapModal;
    $("#cl-overlap-modal").addEventListener("click", (e) => {
      if (e.target === $("#cl-overlap-modal")) closeOverlapModal();
    });
    $("#cl-ov-color-btn").onclick = () => {
      closeOverlapModal();
      state.colorBy = "overlap";
      $("#cl-color-by").value = "overlap";
      renderLegend();
      scheduleRender();
    };
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("#cl-overlap-modal").hidden) closeOverlapModal();
    });

    // Inspector
    $("#cl-insp-close").onclick = clearSelection;
    $("#cl-insp-sort").addEventListener("change", sortAndRenderInspectorGrid);

    const grid = $("#cl-insp-grid");
    grid.addEventListener("mouseover", (e) => {
      const tile = e.target.closest(".cl-mini-tile");
      const next = tile ? Number(tile.dataset.idx) : -1;
      if (next !== state.pulseIndex) {
        state.pulseIndex = next;
        scheduleRender();
      }
    });
    grid.addEventListener("mouseleave", () => {
      if (state.pulseIndex >= 0) {
        state.pulseIndex = -1;
        scheduleRender();
      }
    });
    grid.addEventListener("click", (e) => {
      const tile = e.target.closest(".cl-mini-tile");
      if (!tile) return;
      const pos = Number(tile.dataset.pos);
      const items = state.selectedIndices.map(sampleObj);
      App().openLightbox(pos, items, inspectorLightboxMeta);
    });
    grid.addEventListener("contextmenu", (e) => {
      const tile = e.target.closest(".cl-mini-tile");
      if (!tile) return;
      e.preventDefault();
      const idx = Number(tile.dataset.idx);
      const id = state.samples.ids[idx];
      if (e.shiftKey) App().openTagEditor(id);
      else App().toggleQuickTag(id);
    });

    const inspScroll = $("#cl-insp-scroll");
    inspScroll.addEventListener("scroll", () => {
      if (state.inspRendered >= state.selectedIndices.length) return;
      if (inspScroll.scrollTop + inspScroll.clientHeight + 220 >= inspScroll.scrollHeight) {
        renderInspectorChunk();
      }
    });

    App().onAnnotationsChanged(() => {
      if (!state.initialized) return;
      if (state.colorBy === "tag_alien") scheduleRender();
      if (state.inspectorMode === "sample" && state.pinnedIndex >= 0 && !$("#cl-inspector").hidden) {
        renderSampleFocusHeader(state.pinnedIndex, state.pinnedProbs);
      }
      refreshInspectorTilesOnly();
    });
  }

  /* ---------- cross-model overlap modal ---------- */
  async function loadOverlap() {
    try {
      const res = await fetch("/api/clusters/overlap");
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.available) {
        state.overlap = data;
      }
    } catch (err) {
      console.error("Could not load overlap metrics:", err);
    }
  }

  function heatCellStyle(val, isDiag) {
    if (isDiag) return "";
    const alpha = Math.max(0.05, Math.min(0.42, val * 0.45));
    return `background:rgba(79,156,249,${alpha.toFixed(3)});color:#fff;font-weight:600`;
  }

  function renderMatrixTable(names, mat, isPct = false) {
    const esc = App().esc;
    const shortNames = names.map((n) => n.replace(" (reg)", ""));
    let html = `<table class="cl-ov-table"><thead><tr><th class="row-hdr">Model</th>` +
      shortNames.map((n) => `<th>${esc(n)}</th>`).join("") + `</tr></thead><tbody>`;
    for (let i = 0; i < mat.length; i++) {
      html += `<tr><td class="row-hdr">${esc(shortNames[i])}</td>`;
      for (let j = 0; j < mat[i].length; j++) {
        const v = mat[i][j];
        const diag = i === j;
        const txt = isPct ? `${(v * 100).toFixed(1)}%` : v.toFixed(4);
        html += `<td class="${diag ? "diag" : ""}" style="${heatCellStyle(v, diag)}">${txt}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table>`;
    return html;
  }

  function openOverlapModal() {
    renderOverlapModal();
    $("#cl-overlap-modal").hidden = false;
  }

  function closeOverlapModal() {
    $("#cl-overlap-modal").hidden = true;
  }

  function renderOverlapModal() {
    const body = $("#cl-ov-body");
    const ov = state.overlap;
    if (!ov) {
      body.innerHTML =
        `<div class="cl-overlay-card" style="margin:auto">
           <strong>Overlap metrics not generated yet.</strong><br><br>
           Run <code>uv run eval-overlap</code> in the repository root, then reload.
         </div>`;
      $("#cl-ov-color-btn").disabled = true;
      return;
    }
    $("#cl-ov-color-btn").disabled = false;
    const esc = App().esc;
    const names = ov.spaces.map((s) => s.name);

    const ckaTabs = [
      ["linear_all", "All"],
      ["linear_clean", "Clean"],
      ["linear_test", "Test"],
      ["linear_dropped", "Dropped"],
      ["linear_alien", "Alien"],
      ["linear_overexposed", "Overexp"],
      ["linear_pca50_all", "PCA-50"],
      ["rbf_clean", "RBF Clean"],
    ].filter(([k]) => ov.cka && ov.cka[k]);
    if (!ov.cka[state.ovCkaTab]) state.ovCkaTab = ckaTabs[0][0];
    const ckaEntry = ov.cka[state.ovCkaTab];

    const knnTabs = [
      ["raw:10", "k=10", ov.knn_raw.by_k["10"]],
      ["raw:20", "k=20", ov.knn_raw.by_k["20"]],
      ["raw:50", "k=50", ov.knn_raw.by_k["50"]],
      ["pca:20", "PCA-50 k=20", ov.knn_pca50.by_k["20"]],
    ];
    const activeKnn = (knnTabs.find(([k]) => k === state.ovKnnTab) || knnTabs[1])[2];

    const ariMat = ov.hdbscan_agreement.ari_non_noise;
    const ariAll = ov.hdbscan_agreement.ari_all;
    const nmiAll = ov.hdbscan_agreement.nmi_all;

    const statusRows = Object.entries(ov.knn_raw.by_k["20"].by_status || {}).map(([stName, r]) =>
      `<tr>
         <td class="row-hdr">${esc(stName)}</td>
         <td>${r.count.toLocaleString()}</td>
         <td>${(r.eff_conv * 100).toFixed(1)}%</td>
         <td>${(r.eff_dino * 100).toFixed(1)}%</td>
         <td>${(r.conv_dino * 100).toFixed(1)}%</td>
         <td><strong>${(r.mean_pairwise * 100).toFixed(1)}%</strong></td>
         <td>${(r.three_way * 100).toFixed(1)}%</td>
       </tr>`
    ).join("");

    const classRows = (ov.knn_raw.by_k["20"].by_class_clean || []).map((r) =>
      `<tr>
         <td class="row-hdr">${r.class} · ${esc(r.name)}</td>
         <td>${r.count}</td>
         <td>${(r.eff_conv * 100).toFixed(1)}%</td>
         <td>${(r.eff_dino * 100).toFixed(1)}%</td>
         <td>${(r.conv_dino * 100).toFixed(1)}%</td>
         <td><strong>${(r.mean_pairwise * 100).toFixed(1)}%</strong></td>
       </tr>`
    ).join("");

    body.innerHTML =
      `<div class="cl-ov-grid">
         <div class="cl-ov-card">
           <h3><span>CKA Relational Similarity</span><small style="color:var(--dim)">${ckaEntry.count.toLocaleString()} pts</small></h3>
           <div class="cl-ov-tabs" id="cl-ov-cka-tabs">
             ${ckaTabs.map(([k, lbl]) => `<button class="cl-ov-tab ${state.ovCkaTab === k ? "on" : ""}" data-tab="${k}">${esc(lbl)}</button>`).join("")}
           </div>
           ${renderMatrixTable(names, ckaEntry.matrix, false)}
           <div class="cl-ov-badges">
             <span class="cl-stat-pill">HDBSCAN NMI (all): <strong>${nmiAll[0][1].toFixed(2)}</strong> / <strong>${nmiAll[0][2].toFixed(2)}</strong> / <strong>${nmiAll[1][2].toFixed(2)}</strong></span>
             <span class="cl-stat-pill">Non-noise ARI: <strong>${ariMat[0][1].toFixed(2)}</strong> / <strong>${ariMat[0][2].toFixed(2)}</strong> / <strong>${ariMat[1][2].toFixed(2)}</strong></span>
             <span class="cl-stat-pill">All ARI: <strong>${ariAll[0][1].toFixed(2)}</strong> / <strong>${ariAll[0][2].toFixed(2)}</strong> / <strong>${ariAll[1][2].toFixed(2)}</strong></span>
           </div>
         </div>
         <div class="cl-ov-card">
           <h3><span>Exact k-NN Shared Neighbors</span><small style="color:var(--dim)">rand ${(activeKnn.random_baseline * 100).toFixed(3)}%</small></h3>
           <div class="cl-ov-tabs" id="cl-ov-knn-tabs">
             ${knnTabs.map(([k, lbl]) => `<button class="cl-ov-tab ${state.ovKnnTab === k ? "on" : ""}" data-tab="${k}">${esc(lbl)}</button>`).join("")}
           </div>
           ${renderMatrixTable(names, activeKnn.matrix, true)}
           <div class="cl-ov-badges">
             <span class="cl-stat-pill">Mean pairwise: <strong>${(activeKnn.mean_pairwise.mean * 100).toFixed(1)}%</strong> (med ${(activeKnn.mean_pairwise.median * 100).toFixed(1)}%)</span>
             <span class="cl-stat-pill">3-way consensus: <strong>${(activeKnn.three_way.mean * 100).toFixed(1)}%</strong></span>
           </div>
         </div>
       </div>
       <div class="cl-ov-grid">
         <div class="cl-ov-card">
           <h3><span>k=20 Overlap by Data Status</span></h3>
           <table class="cl-ov-table">
             <thead><tr><th class="row-hdr">Status</th><th>n</th><th>E↔C</th><th>E↔D</th><th>C↔D</th><th>Mean</th><th>3-way</th></tr></thead>
             <tbody>${statusRows}</tbody>
           </table>
         </div>
         <div class="cl-ov-card">
           <h3><span>k=20 Overlap by Clean Class</span></h3>
           <table class="cl-ov-table">
             <thead><tr><th class="row-hdr">Class</th><th>n</th><th>E↔C</th><th>E↔D</th><th>C↔D</th><th>Mean</th></tr></thead>
             <tbody>${classRows}</tbody>
           </table>
         </div>
       </div>`;

    body.querySelectorAll("#cl-ov-cka-tabs .cl-ov-tab").forEach((b) => {
      b.onclick = () => {
        state.ovCkaTab = b.dataset.tab;
        renderOverlapModal();
      };
    });
    body.querySelectorAll("#cl-ov-knn-tabs .cl-ov-tab").forEach((b) => {
      b.onclick = () => {
        state.ovKnnTab = b.dataset.tab;
        renderOverlapModal();
      };
    });
  }

  async function bootClusters(forceRetry = false) {
    if (state.loading) return;
    if (state.initialized && !forceRetry) return;
    state.loading = true;
    try {
      const ok = await loadMeta();
      if (!ok) return;
      state.initialized = true;
      loadOverlap().then(() => {
        if (state.colorBy === "overlap") scheduleRender();
      });
      const defaultSpace = state.spaceOrder.includes("dinov2_vitl14_reg")
        ? "dinov2_vitl14_reg"
        : state.spaceOrder[0];
      if (defaultSpace) await selectSpace(defaultSpace);
    } catch (err) {
      console.error(err);
      $("#cl-overlay").hidden = false;
      $("#cl-overlay-card").innerHTML = `<strong>Error loading clusters:</strong> ${App().esc(err.message)}`;
    } finally {
      state.loading = false;
    }
  }

  window.ClustersView = {
    activate() {
      App().onBootReady(() => {
        bootClusters().then(() => scheduleRender());
      });
    },
  };

  bindStageEvents();
  bindUiControls();
  if (location.hash === "#clusters") {
    window.ClustersView.activate();
  }
})();
