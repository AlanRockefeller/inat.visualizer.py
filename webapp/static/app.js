/* iNat Seasonal Visualizer - web frontend.
 *
 * Taxon and place names are looked up directly against api.inaturalist.org from
 * the browser; our own server only answers numeric queries under api/.
 */
"use strict";

const INAT_API = "https://api.inaturalist.org/v1";
const OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const LOCAL_TILES = "tiles/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const RADIUS_MIN_KM = 1;
const RADIUS_MAX_KM = 1000;
const TOP_CELLS = 25;
const DEFAULTS = { lat: 37.7749, lon: -122.4194, radius: 10, dateFrom: "2000-01-01" };

const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];
const MONTH_ABBR = MONTHS.map((m) => m.slice(0, 3));
// Feb 29 stays selectable; observations on it only exist in leap years.
const DAYS_IN_MONTH = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const CUSTOM_SEASON = "Custom range";
const AROUND_TODAY = "Around today (± 2 weeks)";
const SEASON_PRESETS = [
  ["Spring (Mar – May)", [3, 1], [5, 31]],
  ["Summer (Jun – Aug)", [6, 1], [8, 31]],
  ["Fall (Sep – Nov)", [9, 1], [11, 30]],
  ["Winter (Dec – Feb)", [12, 1], [2, 29]],
  ...MONTHS.map((name, i) => [name, [i + 1, 1], [i + 1, DAYS_IN_MONTH[i]]]),
];
// ColorBrewer YlOrRd, the same ramp as the desktop heatmap.
const HEAT_RAMP = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
  "#fc4e2a", "#e31a1c", "#bd0026", "#800026"];

const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString();

const state = {
  mode: "seasonal",
  taxa: { organism: null, exclude: null }, // {id, label}
  seasonal: null, // last /api/seasonal response + its request
  abundance: null, // last /api/abundance response + its request
  ranked: [],
  selectedCell: null,
  placeNames: new Map(),
  controller: null,
};

/* ---------- small helpers ---------- */

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setStatus(message, kind = "") {
  const el = $("status");
  el.textContent = message;
  el.className = "status " + kind;
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function normalizeLon(lon) {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

function monthDayString([m, d]) {
  return `${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

function formatMonthDay([m, d]) {
  return `${MONTH_ABBR[m - 1]} ${d}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) {
        detail = Array.isArray(body.detail)
          ? body.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ")
          : body.detail;
      }
    } catch (_) { /* not JSON */ }
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/* ---------- autocomplete (taxa and places) ---------- */

function attachAutocomplete(input, list, { search, render, onPick }) {
  let items = [];
  let active = -1;
  let requestId = 0;

  const close = () => { list.hidden = true; active = -1; };
  const highlight = (index) => {
    active = index;
    [...list.children].forEach((li, i) => li.setAttribute("aria-selected", i === index));
  };
  const pick = (index) => {
    if (items[index]) onPick(items[index]);
    close();
  };

  const run = debounce(async () => {
    const query = input.value.trim();
    if (query.length < 2) return;
    const id = ++requestId;
    try {
      const results = await search(query);
      if (id !== requestId) return;
      items = results;
      list.innerHTML = results.map((item, i) =>
        `<li role="option" data-index="${i}">${render(item)}</li>`).join("");
      list.hidden = results.length === 0;
      active = -1;
    } catch (_) {
      if (id === requestId) close();
    }
  }, 250);

  input.addEventListener("input", () => {
    // Any edit hides the current suggestions and invalidates in-flight
    // results, so nothing from an earlier query can be picked or reopen the
    // list under text the user has since changed.
    requestId++;
    close();
    run();
  });
  input.addEventListener("keydown", (event) => {
    if (list.hidden) return;
    if (event.key === "ArrowDown") { event.preventDefault(); highlight(Math.min(items.length - 1, active + 1)); }
    else if (event.key === "ArrowUp") { event.preventDefault(); highlight(Math.max(0, active - 1)); }
    else if (event.key === "Enter" && active >= 0) { event.preventDefault(); pick(active); }
    else if (event.key === "Escape") { close(); }
  });
  list.addEventListener("mousedown", (event) => {
    const li = event.target.closest("li");
    if (li) { event.preventDefault(); pick(Number(li.dataset.index)); }
  });
  input.addEventListener("blur", () => setTimeout(close, 150));
}

async function searchTaxa(query, signal) {
  const url = `${INAT_API}/taxa/autocomplete?per_page=8&q=${encodeURIComponent(query)}`;
  return (await fetchJson(url, { signal })).results || [];
}

function taxonLabel(taxon) {
  const common = taxon.preferred_common_name ? ` (${taxon.preferred_common_name})` : "";
  return `${taxon.name}${common}`;
}

function renderTaxon(taxon) {
  const photo = taxon.default_photo?.square_url
    ? `<img src="${escapeHtml(taxon.default_photo.square_url)}" alt="" loading="lazy">` : "";
  const italic = ["genus", "species", "subspecies", "variety", "form"].includes(taxon.rank);
  const name = italic ? `<span class="sci">${escapeHtml(taxon.name)}</span>` : escapeHtml(taxon.name);
  const common = taxon.preferred_common_name ? ` · ${escapeHtml(taxon.preferred_common_name)}` : "";
  return `${photo}<span>${name}<br><span class="meta">${escapeHtml(taxon.rank)}${common}` +
    ` · ${fmt(taxon.observations_count || 0)} obs</span></span>`;
}

function setupTaxonPicker(field) {
  const input = $(field);
  const list = input.parentElement.querySelector(".suggestions");
  attachAutocomplete(input, list, {
    search: searchTaxa,
    render: renderTaxon,
    onPick: (taxon) => {
      state.taxa[field] = { id: taxon.id, label: taxonLabel(taxon) };
      input.value = state.taxa[field].label;
    },
  });
  input.addEventListener("input", () => {
    if (state.taxa[field] && input.value !== state.taxa[field].label) state.taxa[field] = null;
  });
}

// Like the desktop app: typed text that was never picked resolves to iNat's best match.
// The lookup shares the search's abort signal, and a result is dropped if the
// field was edited meanwhile, so a stale reply cannot restore the old taxon.
async function resolveTaxon(field, signal) {
  const text = $(field).value.trim();
  if (!text) { state.taxa[field] = null; return null; }
  if (state.taxa[field]) return state.taxa[field];
  const [best] = await searchTaxa(text, signal);
  if ($(field).value.trim() !== text) {
    throw new Error(`The ${field} field changed during the lookup. Run the search again.`);
  }
  if (!best) throw new Error(`No iNaturalist taxon matches "${text}".`);
  state.taxa[field] = { id: best.id, label: taxonLabel(best) };
  $(field).value = state.taxa[field].label;
  return state.taxa[field];
}

function setupPlacePicker() {
  const input = $("place");
  const list = input.parentElement.querySelector(".suggestions");
  attachAutocomplete(input, list, {
    search: async (query) => {
      const url = `${INAT_API}/search?sources=places&per_page=8&q=${encodeURIComponent(query)}`;
      const results = (await fetchJson(url)).results || [];
      return results.filter((r) => r.type === "Place" && r.record?.location).map((r) => r.record);
    },
    render: (place) => escapeHtml(place.display_name || place.name),
    onPick: (place) => {
      const [lat, lon] = place.location.split(",").map(Number);
      input.value = place.display_name || place.name;
      setCenter(lat, lon);
      const coords = place.bounding_box_geojson?.coordinates?.flat(3);
      if (coords && coords.length >= 4) {
        const lons = coords.filter((_, i) => i % 2 === 0);
        const lats = coords.filter((_, i) => i % 2 === 1);
        map.fitBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]],
          { padding: [20, 20], maxZoom: 12 });
      } else {
        map.setView([lat, lon], 10);
      }
    },
  });
}

/* ---------- map ---------- */

const map = L.map("map", { worldCopyJump: true, zoomControl: true }).setView([DEFAULTS.lat, DEFAULTS.lon], 9);

// Tiles come through the server's nginx cache; fall back to OSM directly when
// that path is unavailable (for example when running uvicorn locally).
const tiles = L.tileLayer(LOCAL_TILES, { maxZoom: 18, attribution: OSM_ATTRIBUTION }).addTo(map);
let tilesLoaded = false;
let tileErrors = 0;
tiles.on("tileload", () => { tilesLoaded = true; });
tiles.on("tileerror", () => {
  if (!tilesLoaded && ++tileErrors >= 2) {
    tilesLoaded = true;
    tiles.setUrl(OSM_TILES);
  }
});

const selectionColor = () => cssVar("--selection");
const searchCircle = L.circle([DEFAULTS.lat, DEFAULTS.lon], {
  radius: DEFAULTS.radius * 1000, color: selectionColor(), weight: 2, fillOpacity: 0.08, interactive: false,
});
const centerMarker = L.marker([DEFAULTS.lat, DEFAULTS.lon], {
  draggable: true, keyboard: true, title: "Search center (drag to move)",
  icon: L.divIcon({ className: "radius-handle", iconSize: [14, 14] }),
});
const radiusHandle = L.marker([DEFAULTS.lat, DEFAULTS.lon], {
  draggable: true, title: "Drag to change the radius",
  icon: L.divIcon({ className: "radius-handle", iconSize: [12, 12] }),
});
const seasonalLayer = L.layerGroup([searchCircle, centerMarker, radiusHandle]).addTo(map);
const heatRenderer = L.canvas({ padding: 0.5 });
const heatLayer = L.layerGroup();
let highlightRect = null;

function eastPoint(lat, lon, radiusKm) {
  const dLon = (radiusKm / (111.32 * Math.max(0.01, Math.cos(lat * Math.PI / 180))));
  return [lat, lon + dLon];
}

function updateSelectionOverlay() {
  const lat = Number($("lat").value);
  const lon = Number($("lon").value);
  const radius = Number($("radius").value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(radius)) return;
  // Web Mercator stops at 85.05°; only the drawing is clamped, the search keeps
  // its true latitude so a polar abundance cell still graphs its own area.
  const shownLat = clamp(lat, -85.0511, 85.0511);
  searchCircle.setLatLng([shownLat, lon]).setRadius(radius * 1000);
  centerMarker.setLatLng([shownLat, lon]);
  radiusHandle.setLatLng(eastPoint(shownLat, lon, radius));
}

function setCenter(lat, lon) {
  $("lat").value = clamp(lat, -90, 90).toFixed(4);
  $("lon").value = normalizeLon(lon).toFixed(4);
  updateSelectionOverlay();
}

function setRadius(km) {
  $("radius").value = Math.round(clamp(km, RADIUS_MIN_KM, RADIUS_MAX_KM));
  updateSelectionOverlay();
}

function fitSelection() {
  map.fitBounds(searchCircle.getBounds(), { padding: [30, 30], maxZoom: 13 });
}

map.on("click", (event) => {
  if (state.mode === "seasonal") {
    setCenter(event.latlng.lat, event.latlng.lng);
  } else {
    const cell = cellAt(event.latlng.lat, normalizeLon(event.latlng.lng));
    if (cell) selectCell(cell, { pan: false });
  }
});
centerMarker.on("drag", (e) => setCenter(e.latlng.lat, e.latlng.lng));
radiusHandle.on("drag", (e) => setRadius(map.distance(centerMarker.getLatLng(), e.latlng) / 1000));
radiusHandle.on("dragend", updateSelectionOverlay);
["lat", "lon", "radius"].forEach((id) => $(id).addEventListener("change", () => {
  updateSelectionOverlay();
  fitSelection();
}));

/* ---------- mode switching ---------- */

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".segmented button").forEach((b) =>
    b.setAttribute("aria-selected", b.dataset.mode === mode));
  document.querySelectorAll("[data-show]").forEach((el) => {
    el.hidden = el.dataset.show !== mode;
    // A disabled fieldset keeps its hidden controls out of form validation.
    if (el instanceof HTMLFieldSetElement) el.disabled = el.hidden;
  });
  $("run").textContent = mode === "seasonal" ? "Graph" : "Map abundance";
  $("organism-hint").textContent = mode === "seasonal"
    ? "Leave empty for all organisms." : "Required for the abundance map.";
  if (mode === "seasonal") {
    map.removeLayer(heatLayer);
    if (highlightRect) map.removeLayer(highlightRect);
    seasonalLayer.addTo(map);
    fitSelection();
  } else {
    map.removeLayer(seasonalLayer);
    heatLayer.addTo(map);
    if (highlightRect) highlightRect.addTo(map);
    if (!state.abundance) map.setView([20, 0], 2);
  }
  setTimeout(() => map.invalidateSize(), 0);
}

document.querySelectorAll(".segmented button").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));

/* ---------- season range ---------- */

function fillDays(daySelect, month, selectedDay) {
  daySelect.innerHTML = "";
  for (let d = 1; d <= DAYS_IN_MONTH[month - 1]; d++) daySelect.add(new Option(d, d));
  daySelect.value = Math.min(selectedDay, DAYS_IN_MONTH[month - 1]);
}

function setupSeasonPicker() {
  const preset = $("season-preset");
  preset.add(new Option(CUSTOM_SEASON, CUSTOM_SEASON));
  preset.add(new Option(AROUND_TODAY, AROUND_TODAY));
  SEASON_PRESETS.forEach(([label]) => preset.add(new Option(label, label)));
  for (const which of ["start", "end"]) {
    const monthSelect = $(`season-${which}-month`);
    MONTH_ABBR.forEach((m, i) => monthSelect.add(new Option(m, i + 1)));
    monthSelect.addEventListener("change", () => {
      fillDays($(`season-${which}-day`), Number(monthSelect.value), Number($(`season-${which}-day`).value));
      syncPresetToRange();
    });
    $(`season-${which}-day`).addEventListener("change", syncPresetToRange);
  }
  preset.addEventListener("change", () => applyPreset(preset.value));
  applyPreset(AROUND_TODAY);
}

function aroundToday() {
  const now = new Date();
  const start = new Date(now.getTime() - 14 * 864e5);
  const end = new Date(now.getTime() + 14 * 864e5);
  return [[start.getMonth() + 1, start.getDate()], [end.getMonth() + 1, end.getDate()]];
}

function applyPreset(label) {
  let range = null;
  if (label === AROUND_TODAY) range = aroundToday();
  else {
    const match = SEASON_PRESETS.find(([name]) => name === label);
    if (match) range = [match[1], match[2]];
  }
  if (range) setSeasonRange(range[0], range[1]);
  $("season-preset").value = label;
}

function setSeasonRange(start, end) {
  $("season-start-month").value = start[0];
  fillDays($("season-start-day"), start[0], start[1]);
  $("season-end-month").value = end[0];
  fillDays($("season-end-day"), end[0], end[1]);
  syncPresetToRange();
}

function seasonRange() {
  return [
    [Number($("season-start-month").value), Number($("season-start-day").value)],
    [Number($("season-end-month").value), Number($("season-end-day").value)],
  ];
}

function syncPresetToRange() {
  const [start, end] = seasonRange();
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];
  const match = SEASON_PRESETS.find(([, s, e]) => same(s, start) && same(e, end));
  const [ts, te] = aroundToday();
  $("season-preset").value = match ? match[0]
    : (same(ts, start) && same(te, end) ? AROUND_TODAY : CUSTOM_SEASON);
}

function seasonLabel() {
  const [start, end] = seasonRange();
  return `${formatMonthDay(start)} – ${formatMonthDay(end)}`;
}

/* ---------- seasonal graph ---------- */

const LEAP_YEAR = 2024;

function dayOfYearLabel(dayIndex) {
  const date = new Date(Date.UTC(LEAP_YEAR, 0, dayIndex + 1));
  return `${MONTH_ABBR[date.getUTCMonth()]} ${date.getUTCDate()}`;
}

function binLabels(view) {
  if (view === "monthly") return { ticks: MONTH_ABBR.slice(), tooltip: MONTHS.slice() };
  if (view === "weekly") {
    const ticks = [];
    const tooltip = [];
    for (let w = 0; w < 52; w++) {
      const mid = dayOfYearLabel(w * 7 + 3);
      ticks.push(MONTH_ABBR[new Date(Date.UTC(LEAP_YEAR, 0, w * 7 + 4)).getUTCMonth()]);
      tooltip.push(`Week ${w + 1} (around ${mid})`);
    }
    return { ticks, tooltip };
  }
  const ticks = [];
  const tooltip = [];
  for (let d = 0; d < 366; d++) {
    const date = new Date(Date.UTC(LEAP_YEAR, 0, d + 1));
    ticks.push(MONTH_ABBR[date.getUTCMonth()]);
    tooltip.push(`${dayOfYearLabel(d)} (day ${d + 1})`);
  }
  return { ticks, tooltip };
}

let chart = null;

function renderChart() {
  const result = state.seasonal;
  if (!result) return;
  const view = $("view").value;
  const data = result.data.bins[view];
  const { ticks, tooltip } = binLabels(view);
  const barColor = cssVar("--bar");
  const hoverColor = cssVar("--bar-hover");
  const inkColor = cssVar("--text-2");
  const gridColor = cssVar("--grid");
  const bodyFont = getComputedStyle(document.body).fontFamily;

  if (chart) chart.destroy();
  chart = new Chart($("chart"), {
    type: "bar",
    data: {
      labels: ticks,
      datasets: [{
        label: "Observations",
        data,
        backgroundColor: barColor,
        hoverBackgroundColor: hoverColor,
        borderRadius: view === "daily" ? 1 : 4,
        borderSkipped: "start",
        categoryPercentage: view === "daily" ? 1 : 0.9,
        barPercentage: view === "daily" ? 1 : 0.9,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          displayColors: false,
          callbacks: {
            title: (items) => tooltip[items[0].dataIndex],
            label: (item) => `${fmt(item.raw)} observations`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: inkColor,
            font: { family: bodyFont },
            autoSkip: false,
            maxRotation: 0,
            // One label per month, at the first bin of that month.
            callback: (_value, index) =>
              (index === 0 || ticks[index] !== ticks[index - 1] ? ticks[index] : null),
          },
        },
        y: {
          beginAtZero: true,
          grid: { color: gridColor },
          border: { display: false },
          ticks: { color: inkColor, font: { family: bodyFont }, precision: 0 },
          title: { display: true, text: "Observations", color: inkColor },
        },
      },
    },
  });
}

function seasonalTitle(request) {
  const who = request.organismLabel || "All organisms";
  const minus = request.excludeLabel ? `, excluding ${request.excludeLabel}` : "";
  return `${who}${minus} within ${fmt(request.radius)} km of ${request.lat.toFixed(3)}, ${request.lon.toFixed(3)}`;
}

function seasonalQuery(request, extra = {}) {
  const params = new URLSearchParams({
    lat: request.lat, lon: request.lon, radius: request.radius,
    date_from: request.dateFrom, date_to: request.dateTo, ...extra,
  });
  if (request.taxonId) params.append("taxon_id", request.taxonId);
  if (request.excludeId) params.append("exclude_taxon_id", request.excludeId);
  return `api/seasonal?${params}`;
}

async function runSeasonal(signal) {
  const organism = await resolveTaxon("organism", signal);
  const exclude = await resolveTaxon("exclude", signal);
  const request = {
    lat: Number($("lat").value),
    lon: Number($("lon").value),
    radius: Number($("radius").value),
    dateFrom: $("date-from").value,
    dateTo: $("date-to").value,
    taxonId: organism?.id,
    organismLabel: organism?.label,
    excludeId: exclude?.id,
    excludeLabel: exclude?.label,
  };
  setStatus("Counting observations…", "busy");
  const started = performance.now();
  const data = await fetchJson(seasonalQuery(request), { signal });
  state.seasonal = { request, data };
  $("chart-title").textContent = seasonalTitle(request);
  const span = data.first_date ? `, ${data.first_date} to ${data.last_date}` : "";
  const taxa = data.taxa_included ? ` · ${fmt(data.taxa_included)} taxa included` : "";
  $("chart-summary").textContent = `${fmt(data.total)} observations${span}${taxa}.`;
  $("download-png").disabled = data.total === 0;
  $("download-csv").disabled = data.total === 0;
  renderChart();
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  setStatus(data.total ? `Done in ${seconds}s.` : "No observations match this search.");
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadPng() {
  if (!chart) return;
  const source = $("chart");
  const pad = 20;
  const titleHeight = 48;
  const out = document.createElement("canvas");
  const scale = source.width / source.clientWidth;
  out.width = source.width + pad * 2 * scale;
  out.height = source.height + (titleHeight + pad) * scale;
  const ctx = out.getContext("2d");
  ctx.fillStyle = cssVar("--surface");
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.fillStyle = cssVar("--text");
  ctx.font = `600 ${15 * scale}px ${getComputedStyle(document.body).fontFamily}`;
  ctx.fillText($("chart-title").textContent, pad * scale, 24 * scale);
  ctx.fillStyle = cssVar("--text-2");
  ctx.font = `${12 * scale}px ${getComputedStyle(document.body).fontFamily}`;
  ctx.fillText(`${$("chart-summary").textContent} Source: iNaturalist via dikarya.us/visualizer`,
    pad * scale, 42 * scale);
  ctx.drawImage(source, pad * scale, titleHeight * scale);
  out.toBlob((blob) => downloadBlob(blob, "inat-seasonal.png"), "image/png");
}

async function downloadCsv() {
  if (!state.seasonal) return;
  setStatus("Preparing CSV…", "busy");
  try {
    const response = await fetch(seasonalQuery(state.seasonal.request, { format: "csv" }));
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || response.statusText);
    }
    downloadBlob(await response.blob(), "inat-observations.csv");
    setStatus("CSV downloaded.");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

/* ---------- abundance map ---------- */

function rankCells(cells, metric, minCount, limit) {
  let ranked;
  if (metric === "share") {
    ranked = cells.filter((c) => c.count >= minCount);
    ranked.sort((a, b) => (b.share - a.share) || (b.count - a.count));
  } else {
    ranked = cells.slice().sort((a, b) => b.count - a.count);
  }
  return limit ? ranked.slice(0, limit) : ranked;
}

function rampColor(t) {
  const x = clamp(t, 0, 1) * (HEAT_RAMP.length - 1);
  const i = Math.min(HEAT_RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = HEAT_RAMP[i];
  const b = HEAT_RAMP[i + 1];
  const mix = (k) => Math.round(parseInt(a.slice(k, k + 2), 16) * (1 - f) + parseInt(b.slice(k, k + 2), 16) * f);
  return `rgb(${mix(1)}, ${mix(3)}, ${mix(5)})`;
}

function cellKey(cell) {
  return `${cell.cell_lat.toFixed(4)},${cell.cell_lon.toFixed(4)},${cell.cell_deg}`;
}

function cellBounds(cell) {
  return [[cell.cell_lat, cell.cell_lon], [cell.cell_lat + cell.cell_deg, cell.cell_lon + cell.cell_deg]];
}

function cellAt(lat, lon) {
  if (!state.abundance) return null;
  return state.abundance.data.cells.find((c) =>
    c.cell_lat <= lat && lat < c.cell_lat + c.cell_deg && c.cell_lon <= lon && lon < c.cell_lon + c.cell_deg) || null;
}

function locationLabel(cell) {
  return state.placeNames.get(cellKey(cell))
    || `${cell.centroid_lat.toFixed(2)}°, ${cell.centroid_lon.toFixed(2)}°`;
}

function drawHeatmap() {
  heatLayer.clearLayers();
  const result = state.abundance;
  if (!result) return;
  const metric = $("metric").value;
  const minCount = Number($("min-obs").value) || 1;
  const visible = metric === "share"
    ? result.data.cells.filter((c) => c.count >= minCount) : result.data.cells;
  const values = visible.map((c) => (metric === "share" ? c.share * 100 : c.count));
  // Not Math.max(...values): a broad taxon at 0.25° can exceed the argument limit.
  const max = values.length ? values.reduce((a, b) => (b > a ? b : a), -Infinity) : 1;
  const logMax = Math.log(Math.max(max, 2));
  visible.forEach((cell, i) => {
    const t = metric === "share" ? values[i] / Math.max(max, 1e-9) : Math.log(Math.max(values[i], 1)) / logMax;
    const rect = L.rectangle(cellBounds(cell), {
      renderer: heatRenderer, stroke: false, fillColor: rampColor(t), fillOpacity: 0.72,
    });
    rect.bindTooltip(() =>
      `<strong>${escapeHtml(locationLabel(cell))}</strong><br>${fmt(cell.count)} of ${fmt(cell.total)} observations` +
      ` (${(cell.share * 100).toFixed(1)}%)`, { sticky: true });
    rect.on("click", (e) => { L.DomEvent.stop(e); selectCell(cell, { pan: false }); });
    heatLayer.addLayer(rect);
  });
  $("legend").hidden = visible.length === 0;
  $("legend-min").textContent = metric === "share" ? "0%" : "1";
  $("legend-max").textContent = metric === "share" ? `${max.toFixed(1)}%` : fmt(max);
  $("legend-label").textContent = metric === "share" ? "% of all observations (linear)" : "observations (log scale)";
}

function renderRanking() {
  const result = state.abundance;
  if (!result) return;
  const metric = $("metric").value;
  const minCount = Number($("min-obs").value) || 1;
  state.ranked = rankCells(result.data.cells, metric, minCount, TOP_CELLS);
  const tbody = $("ranking").querySelector("tbody");
  tbody.innerHTML = state.ranked.map((cell, i) => `
    <tr data-index="${i}" tabindex="0" aria-selected="${state.selectedCell === cell}"
        title="${fmt(cell.count)} of ${fmt(cell.total)} observations in this ${cell.cell_deg}° cell during the season">
      <td class="num">${i + 1}</td>
      <td data-key="${cellKey(cell)}">${escapeHtml(locationLabel(cell))}</td>
      <td class="num">${fmt(cell.count)}</td>
      <td class="num">${(cell.share * 100).toFixed(1)}%</td>
    </tr>`).join("");
  const metricText = metric === "share"
    ? `share of all observations (cells with ≥ ${minCount} obs)` : "observation count";
  $("abundance-summary").textContent =
    `Top ${state.ranked.length} of ${fmt(result.data.cells.length)} ${result.data.cell_deg}° cells by ${metricText}. ` +
    `${fmt(result.data.total)} observations in total. Click a cell or row, then "Graph this area".`;
  requestPlaceNames();
}

$("ranking").addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-index]");
  if (row) selectCell(state.ranked[Number(row.dataset.index)], { pan: true });
});
$("ranking").addEventListener("dblclick", (event) => {
  if (event.target.closest("tr[data-index]")) graphSelectedCell();
});
$("ranking").addEventListener("keydown", (event) => {
  // Rows are focusable so keyboard users can pick a cell: Enter or Space
  // selects it, and Enter on the selected row opens its seasonal graph.
  const row = event.target.closest("tr[data-index]");
  if (!row || (event.key !== "Enter" && event.key !== " ")) return;
  event.preventDefault();
  const cell = state.ranked[Number(row.dataset.index)];
  if (event.key === "Enter" && state.selectedCell === cell) graphSelectedCell();
  else selectCell(cell, { pan: true });
});

function selectCell(cell, { pan }) {
  state.selectedCell = cell;
  $("graph-cell").disabled = !cell;
  document.querySelectorAll("#ranking tbody tr").forEach((tr) =>
    tr.setAttribute("aria-selected", state.ranked[Number(tr.dataset.index)] === cell));
  if (highlightRect) map.removeLayer(highlightRect);
  highlightRect = null;
  if (!cell) return;
  highlightRect = L.rectangle(cellBounds(cell), {
    color: selectionColor(), weight: 3, fill: false, interactive: false,
  }).addTo(map);
  if (pan) {
    const span = Math.max(cell.cell_deg * 5, 2);
    const lat = cell.cell_lat + cell.cell_deg / 2;
    const lon = cell.cell_lon + cell.cell_deg / 2;
    map.fitBounds([[lat - span, lon - span], [lat + span, lon + span]]);
  }
}

function graphSelectedCell() {
  const cell = state.selectedCell;
  if (!cell) return;
  setMode("seasonal");
  setCenter(cell.cell_lat + cell.cell_deg / 2, cell.cell_lon + cell.cell_deg / 2);
  setRadius(cell.radius_km);
  fitSelection();
  $("search-form").requestSubmit();
}
$("graph-cell").addEventListener("click", graphSelectedCell);

// Port of most_specific_standard_place: the finest administrative place.
function mostSpecificPlace(response) {
  const standard = response?.results?.standard || [];
  const candidates = standard.filter((p) => p && p.admin_level != null && (p.display_name || p.name));
  if (!candidates.length) return null;
  candidates.sort((a, b) => (b.admin_level - a.admin_level) || ((a.bbox_area || 0) - (b.bbox_area || 0)));
  return candidates[0].display_name || candidates[0].name;
}

let placeNameRun = 0;
async function requestPlaceNames() {
  const run = ++placeNameRun;
  const pending = state.ranked.filter((c) => !state.placeNames.has(cellKey(c)));
  for (const cell of pending) {
    if (run !== placeNameRun) return;
    const d = 0.005;
    const params = new URLSearchParams({
      nelat: cell.centroid_lat + d, nelng: cell.centroid_lon + d,
      swlat: cell.centroid_lat - d, swlng: cell.centroid_lon - d, per_page: 30,
    });
    try {
      const name = mostSpecificPlace(await fetchJson(`${INAT_API}/places/nearby?${params}`));
      if (!name) continue;
      state.placeNames.set(cellKey(cell), name);
      const td = document.querySelector(`#ranking td[data-key="${cellKey(cell)}"]`);
      if (td) td.textContent = name;
    } catch (_) {
      return; // iNat rate limit or offline: keep the coordinate labels.
    }
  }
}

async function runAbundance(signal) {
  const organism = await resolveTaxon("organism", signal);
  if (!organism) throw new Error("Choose an organism to map where it is observed most.");
  const exclude = await resolveTaxon("exclude", signal);
  const [start, end] = seasonRange();
  const request = {
    taxonId: organism.id,
    organismLabel: organism.label,
    excludeId: exclude?.id,
    excludeLabel: exclude?.label,
    seasonStart: monthDayString(start),
    seasonEnd: monthDayString(end),
    dateFrom: $("date-from").value,
    dateTo: $("date-to").value,
    cellDeg: $("cell-deg").value,
    seasonLabel: seasonLabel(),
  };
  const params = new URLSearchParams({
    taxon_id: request.taxonId, season_start: request.seasonStart, season_end: request.seasonEnd,
    date_from: request.dateFrom, date_to: request.dateTo, cell_deg: request.cellDeg,
  });
  if (request.excludeId) params.append("exclude_taxon_id", request.excludeId);
  setStatus("Scanning every observation for this time of year…", "busy");
  const started = performance.now();
  const data = await fetchJson(`api/abundance?${params}`, { signal });
  state.abundance = { request, data };
  state.selectedCell = null;
  $("graph-cell").disabled = true;
  if (highlightRect) { map.removeLayer(highlightRect); highlightRect = null; }
  $("abundance-title").textContent = `Where ${organism.label} is observed, ${request.seasonLabel}`;
  drawHeatmap();
  renderRanking();
  if (state.ranked.length) {
    const lats = state.ranked.map((c) => c.cell_lat);
    const lons = state.ranked.map((c) => c.cell_lon);
    const pad = Math.max(Number(request.cellDeg) * 2, 1);
    map.fitBounds([[Math.min(...lats) - pad, Math.min(...lons) - pad],
      [Math.max(...lats) + Number(request.cellDeg) + pad, Math.max(...lons) + Number(request.cellDeg) + pad]]);
  }
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  setStatus(data.cells.length ? `Done in ${seconds}s.` : "No observations of this organism in that season.");
}

function onRankingChanged() {
  $("min-obs").disabled = $("metric").value !== "share";
  if (!state.abundance) return;
  drawHeatmap();
  renderRanking();
  writeUrl();
}
$("metric").addEventListener("change", onRankingChanged);
$("min-obs").addEventListener("change", onRankingChanged);

/* ---------- permalinks ---------- */

function writeUrl() {
  // Serialize the search that produced the result on screen rather than the
  // form, which may have been edited since, plus the current display settings.
  const result = state.mode === "seasonal" ? state.seasonal : state.abundance;
  if (!result) return;
  const request = result.request;
  const p = new URLSearchParams({ mode: state.mode });
  if (request.taxonId) {
    p.set("organism", request.taxonId);
    p.set("organism_name", request.organismLabel || `Taxon ${request.taxonId}`);
  }
  if (request.excludeId) {
    p.set("exclude", request.excludeId);
    p.set("exclude_name", request.excludeLabel || `Taxon ${request.excludeId}`);
  }
  p.set("from", request.dateFrom);
  p.set("to", request.dateTo);
  if (state.mode === "seasonal") {
    p.set("lat", request.lat);
    p.set("lon", request.lon);
    p.set("radius", request.radius);
    p.set("view", $("view").value);
  } else {
    p.set("season", `${request.seasonStart}_${request.seasonEnd}`);
    p.set("cell", request.cellDeg);
    p.set("metric", $("metric").value);
    p.set("min", $("min-obs").value);
  }
  history.replaceState(null, "", `?${p}`);
}

function readUrl() {
  const p = new URLSearchParams(location.search);
  for (const field of ["organism", "exclude"]) {
    const id = Number(p.get(field));
    if (id) {
      state.taxa[field] = { id, label: p.get(`${field}_name`) || `Taxon ${id}` };
      $(field).value = state.taxa[field].label;
    }
  }
  if (p.get("from")) $("date-from").value = p.get("from");
  if (p.get("to")) $("date-to").value = p.get("to");
  const lat = Number(p.get("lat"));
  const lon = Number(p.get("lon"));
  if (p.has("lat") && p.has("lon") && Number.isFinite(lat) && Number.isFinite(lon)) setCenter(lat, lon);
  if (p.get("radius")) setRadius(Number(p.get("radius")));
  if (["daily", "weekly", "monthly"].includes(p.get("view"))) $("view").value = p.get("view");
  const season = /^(\d\d)-(\d\d)_(\d\d)-(\d\d)$/.exec(p.get("season") || "");
  if (season) setSeasonRange([+season[1], +season[2]], [+season[3], +season[4]]);
  if (["0.25", "0.5", "1", "2"].includes(p.get("cell"))) $("cell-deg").value = p.get("cell");
  if (["count", "share"].includes(p.get("metric"))) $("metric").value = p.get("metric");
  if (p.get("min")) $("min-obs").value = p.get("min");
  $("min-obs").disabled = $("metric").value !== "share";
  const mode = p.get("mode") === "abundance" ? "abundance" : "seasonal";
  setMode(mode);
  return p.has("mode");
}

/* ---------- search lifecycle ---------- */

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.controller) state.controller.abort();
  const controller = new AbortController();
  state.controller = controller;
  $("run").disabled = true;
  $("cancel").hidden = false;
  try {
    if (state.mode === "seasonal") await runSeasonal(controller.signal);
    else await runAbundance(controller.signal);
    writeUrl();
  } catch (error) {
    if (error.name === "AbortError") setStatus("Search cancelled.");
    else setStatus(error.message || String(error), "error");
  } finally {
    if (state.controller === controller) {
      state.controller = null;
      $("run").disabled = false;
      $("cancel").hidden = true;
    }
  }
});

$("cancel").addEventListener("click", () => state.controller?.abort());
$("view").addEventListener("change", () => { renderChart(); if (state.seasonal) writeUrl(); });
$("download-png").addEventListener("click", downloadPng);
$("download-csv").addEventListener("click", downloadCsv);

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  renderChart();
  searchCircle.setStyle({ color: selectionColor() });
  if (highlightRect) highlightRect.setStyle({ color: selectionColor() });
});

let dateToEdited = false;
$("date-to").addEventListener("input", () => { dateToEdited = true; });

async function loadStats() {
  try {
    const stats = await fetchJson("api/stats");
    const newest = stats.newest_date ? ` · newest ${stats.newest_date}` : "";
    $("stats").textContent = `${fmt(stats.total_observations)} observations${newest} · updated ${stats.updated} · v${stats.version}`;
    // Default "To" to the newest observation unless the user already changed it.
    if (stats.newest_date && !dateToEdited) $("date-to").value = stats.newest_date;
  } catch (error) {
    $("stats").textContent = `Database unavailable: ${error.message}`;
  }
}

/* ---------- start ---------- */

setupTaxonPicker("organism");
setupTaxonPicker("exclude");
setupPlacePicker();
setupSeasonPicker();
$("date-from").value = DEFAULTS.dateFrom;
$("date-to").value = new Date().toISOString().slice(0, 10);
$("lat").value = DEFAULTS.lat.toFixed(4);
$("lon").value = DEFAULTS.lon.toFixed(4);
$("radius").value = DEFAULTS.radius;
updateSelectionOverlay();

const hasPermalink = readUrl();
loadStats().then(() => {
  // A permalink's own date range wins over the newest-date default, but not
  // over an edit the user made while the stats were loading.
  const p = new URLSearchParams(location.search);
  if (p.get("to") && !dateToEdited) $("date-to").value = p.get("to");
  if (hasPermalink) $("search-form").requestSubmit();
});
