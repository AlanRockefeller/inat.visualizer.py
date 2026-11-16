/* iNat Seasonal Visualizer - web frontend.
 *
 * Taxon and place names are looked up directly against api.inaturalist.org from
 * the browser; our own server only answers numeric queries under api/.
 */
"use strict";

const INAT_API = "https://api.inaturalist.org/v1";
const INAT_OBSERVATIONS = "https://www.inaturalist.org/observations";
// The database is iNaturalist's GBIF export: research grade, and only these
// licenses, so iNat links filter the same way to show the same observations.
// Once backfilled observations are merged in (api/stats says all_licenses),
// links drop the license filter to match.
const INAT_EXPORT_FILTER = { quality_grade: "research", license: "cc0,cc-by,cc-by-nc" };
const INAT_ALL_LICENSES_FILTER = { quality_grade: "research" };
// One color per compared organism. Blue first (the desktop app's bar color);
// the rest are Okabe-Ito hues chosen to stay apart, colorblind-friendly too.
const SERIES_COLORS = ["#1f77b4", "#e69f00", "#009e73", "#d55e00", "#9467bd", "#cc79a7"];
const MAX_SERIES = SERIES_COLORS.length;
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
  taxa: { exclude: null }, // {id, label}
  organisms: [], // [{id, label}], compared as separate series; empty = all organisms
  place: null, // {id, name, geometry, lat, lon} while a named region is the search area
  placeLoading: null, // promise for the place being picked, awaited by Graph
  placeRelease: null, // ends that wait early when the pick is cancelled
  allLicenses: false, // the database includes backfilled (non-export) licenses
  lastRadius: 10, // restored when leaving region mode without a typed radius
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

const seriesColor = (index) => SERIES_COLORS[index % SERIES_COLORS.length];

function renderOrganismChips() {
  const chips = $("organism-chips");
  chips.innerHTML = state.organisms.map((taxon, i) =>
    `<li class="chip"><span class="swatch" style="background:${seriesColor(i)}"></span>` +
    `<span class="chip-label">${escapeHtml(taxon.label)}</span>` +
    `<button type="button" data-index="${i}" aria-label="Remove ${escapeHtml(taxon.label)}">×</button></li>`).join("");
  chips.hidden = state.organisms.length === 0;
  const full = state.organisms.length >= MAX_SERIES;
  $("organism").disabled = full;
  $("organism").placeholder = full ? `Up to ${MAX_SERIES} organisms`
    : state.organisms.length ? "Add another to compare" : "e.g. Agaricales, Amanita muscaria";
  updateOrganismHint();
}

function addOrganism(taxon) {
  if (!state.organisms.some((t) => t.id === taxon.id) && state.organisms.length < MAX_SERIES) {
    state.organisms.push(taxon);
  }
  $("organism").value = "";
  renderOrganismChips();
}

function setupOrganismPicker() {
  const input = $("organism");
  const list = input.parentElement.querySelector(".suggestions");
  attachAutocomplete(input, list, {
    search: searchTaxa,
    render: renderTaxon,
    onPick: (taxon) => addOrganism({ id: taxon.id, label: taxonLabel(taxon) }),
  });
  $("organism-chips").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-index]");
    if (!button) return;
    state.organisms.splice(Number(button.dataset.index), 1);
    renderOrganismChips();
    $("organism").focus();
  });
}

// Text typed into the organism box but never picked becomes a chip using
// iNat's best match, as the single-organism field did.
async function resolveOrganisms(signal) {
  const text = $("organism").value.trim();
  if (text) {
    const [best] = await searchTaxa(text, signal);
    if ($("organism").value.trim() !== text) {
      throw new Error("The organism field changed during the lookup. Run the search again.");
    }
    if (!best) throw new Error(`No iNaturalist taxon matches "${text}".`);
    addOrganism({ id: best.id, label: taxonLabel(best) });
  }
  return state.organisms.slice();
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
      // Graph waits on this; clearRegion() releases the wait if the pick is
      // superseded by a circle edit.
      let release;
      const cancelled = new Promise((resolve) => { release = resolve; });
      state.placeLoading = Promise.race([pickPlace(place), cancelled]);
      state.placeRelease = release;
    },
  });
}

// Graph waits on this, so a search never runs with the previous area while
// a newly picked place's outline is still loading.
async function pickPlace(place) {
  const [lat, lon] = place.location.split(",").map(Number);
  $("place").value = place.display_name || place.name;
  // Prefer the place's outline; fall back to a circle when it has none.
  if (await selectRegion(place.id, $("place").value, [lat, lon])) return;
  clearRegion();
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
  const bounds = state.place ? regionLayer.getBounds() : searchCircle.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, { padding: [30, 30], maxZoom: 13 });
}

map.on("click", (event) => {
  if (state.mode === "seasonal") {
    clearRegion();
    setCenter(event.latlng.lat, event.latlng.lng);
  } else {
    const cell = cellAt(event.latlng.lat, normalizeLon(event.latlng.lng));
    if (cell) selectCell(cell, { pan: false });
  }
});
centerMarker.on("drag", (e) => setCenter(e.latlng.lat, e.latlng.lng));
radiusHandle.on("drag", (e) => setRadius(map.distance(centerMarker.getLatLng(), e.latlng) / 1000));
radiusHandle.on("dragend", updateSelectionOverlay);
// A typed radius or coordinate means a circle again, centred on the region's
// middle unless the coordinates were edited too.
["lat", "lon", "radius"].forEach((id) => $(id).addEventListener("change", () => {
  // Also cancels an outline still loading, so it cannot replace this edit.
  if (id !== "radius" || $("radius").value !== "") clearRegion();
  if (id === "radius" && $("radius").value) state.lastRadius = Number($("radius").value);
  updateSelectionOverlay();
  fitSelection();
}));

// The radius handle is a small dot that is easy to miss, especially on a
// phone, so it carries a label until it has been dragged once.
radiusHandle.bindTooltip("Drag to resize", {
  permanent: true, direction: "right", offset: [8, 0], className: "handle-tip",
});
radiusHandle.once("dragstart", () => radiusHandle.unbindTooltip());

/* ---------- named regions ---------- */

const regionLayer = L.geoJSON(null, {
  interactive: false,
  style: () => ({ color: selectionColor(), weight: 2, fillOpacity: 0.08 }),
});
let regionRequest = 0;

async function fetchPlace(id) {
  const place = (await fetchJson(`${INAT_API}/places/${id}`)).results?.[0];
  const geometry = place?.geometry_geojson;
  if (!geometry || !["Polygon", "MultiPolygon"].includes(geometry.type)) return null;
  return place;
}

// Search inside a named place's outline instead of a circle. Returns false
// when iNat has no outline for it, so the caller can fall back to a circle.
async function selectRegion(id, name, center) {
  const request = ++regionRequest;
  let place;
  try {
    setStatus("Loading the outline of " + name + "…", "busy");
    place = await fetchPlace(id);
  } catch (error) {
    if (request !== regionRequest) return true; // a later pick or edit won
    setStatus(`Could not load the outline of ${name}: ${error.message}`, "error");
    return false;
  }
  if (request !== regionRequest) return true; // a later pick or edit won
  setStatus("");
  if (!place) return false;
  const [lat, lon] = center || place.location.split(",").map(Number);
  if (!state.place && $("radius").value) state.lastRadius = Number($("radius").value);
  state.place = { id: place.id, name, geometry: place.geometry_geojson, lat, lon };
  regionLayer.clearLayers().addData(place.geometry_geojson);
  $("lat").value = lat.toFixed(4);
  $("lon").value = normalizeLon(lon).toFixed(4);
  $("radius").value = "";
  $("radius").required = false;
  [searchCircle, centerMarker, radiusHandle].forEach((layer) => seasonalLayer.removeLayer(layer));
  seasonalLayer.addLayer(regionLayer);
  updateAreaHint();
  if (state.mode === "seasonal" && !$("map").hidden) fitSelection();
  return true;
}

// Back to a circle centred where the region was (or wherever the caller
// moves it next), with the typed radius or the last one used.
function clearRegion() {
  regionRequest++;
  state.placeRelease?.();
  state.placeLoading = null;
  state.placeRelease = null;
  if (!state.place) return;
  state.place = null;
  seasonalLayer.removeLayer(regionLayer);
  regionLayer.clearLayers();
  [searchCircle, centerMarker, radiusHandle].forEach((layer) => seasonalLayer.addLayer(layer));
  $("radius").required = true;
  if (!$("radius").value) $("radius").value = state.lastRadius;
  $("place").value = "";
  updateSelectionOverlay();
  updateAreaHint();
}

function updateAreaHint() {
  $("area-hint").textContent = state.place
    ? `Searching inside the outline of ${state.place.name}. Type a radius to search a circle instead.`
    : "Drag the dot on the circle's edge to change the radius.";
}

/* ---------- map visibility ---------- */

// The seasonal graph hides the map once it has results (the map is only
// needed to pick the area); the button brings it back.
function setMapVisible(visible) {
  const mapEl = $("map");
  if (mapEl.hidden === !visible) return;
  mapEl.hidden = !visible;
  mapEl.parentElement.classList.toggle("map-hidden", !visible);
  $("map-toggle").textContent = visible ? "Hide map" : "Show map";
  $("map-toggle").setAttribute("aria-expanded", String(visible));
  if (visible) {
    // Leaflet cannot measure a hidden container, so size and fit on reveal.
    map.invalidateSize();
    if (state.mode === "seasonal") fitSelection();
  }
}
$("map-toggle").addEventListener("click", () => setMapVisible($("map").hidden));

function updateOrganismHint() {
  const n = state.organisms.length;
  $("organism-hint").textContent = state.mode === "seasonal"
    ? (n ? `Add up to ${MAX_SERIES} to compare them on one graph.`
      : `Leave empty for all organisms, or add up to ${MAX_SERIES} to compare.`)
    : (n > 1 ? "The abundance map combines the organisms above."
      : "Required for the abundance map.");
}

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
  setMapVisible(true);
  $("run").textContent = mode === "seasonal" ? "Graph" : "Map abundance";
  updateOrganismHint();
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

// Redraw hovered bars with a soft halo in their own color, so they read as
// clickable. Drawn after the datasets so the halo sits on top of neighbours.
const barGlow = {
  id: "barGlow",
  afterDatasetsDraw(chartInstance) {
    const glow = chartInstance.$glow || [];
    if (!glow.length) return;
    const { ctx } = chartInstance;
    for (const { element } of glow) {
      ctx.save();
      ctx.shadowColor = element.options.backgroundColor;
      ctx.shadowBlur = 14;
      element.draw(ctx);
      ctx.restore();
    }
  },
  afterEvent(chartInstance, { event }) {
    if (event.type === "mouseout" && chartInstance.$glow?.length) {
      chartInstance.$glow = [];
      chartInstance.$glowKey = "";
      chartInstance.draw();
    }
  },
};

// Below this chart width, month names become single letters so twelve fit.
const NARROW_CHART_PX = 560;

function renderChart() {
  const result = state.seasonal;
  if (!result) return;
  const { request, results } = result;
  const view = $("view").value;
  const { ticks, tooltip } = binLabels(view);
  const multiple = request.series.length > 1;
  const inkColor = cssVar("--text-2");
  const gridColor = cssVar("--grid");
  const bodyFont = getComputedStyle(document.body).fontFamily;
  const narrowBars = view === "daily" || (multiple && view === "weekly");

  if (chart) chart.destroy();
  chart = new Chart($("chart"), {
    type: "bar",
    plugins: [barGlow],
    data: {
      labels: ticks,
      datasets: request.series.map((series, i) => ({
        label: series.label || "All organisms",
        data: results[i].bins[view],
        // A single series keeps the theme's bar colors (the desktop blue).
        backgroundColor: multiple ? seriesColor(i) : cssVar("--bar"),
        hoverBackgroundColor: multiple ? seriesColor(i) : cssVar("--bar-hover"),
        borderRadius: narrowBars ? 1 : 4,
        borderSkipped: "start",
        categoryPercentage: narrowBars ? 1 : 0.9,
        barPercentage: narrowBars ? 1 : 0.9,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      onClick: (event, _elements, chartInstance) => openBar(event, chartInstance),
      onHover: (event, elements, chartInstance) => {
        event.native.target.style.cursor = elements.length ? "pointer" : "default";
        // Glow the bar a click would open, or the whole column above the bars.
        const [hit] = chartInstance.getElementsAtEventForMode(event, "nearest", { intersect: true }, false);
        const glow = hit ? [hit] : elements;
        const key = glow.map((e) => `${e.datasetIndex}:${e.index}`).join();
        if (key !== chartInstance.$glowKey) {
          chartInstance.$glowKey = key;
          chartInstance.$glow = glow;
          chartInstance.draw();
        }
      },
      plugins: {
        legend: { display: false }, // the HTML legend above the chart has links
        tooltip: {
          displayColors: multiple,
          callbacks: {
            title: (items) => tooltip[items[0].dataIndex],
            label: (item) => multiple
              ? `${item.dataset.label}: ${fmt(item.raw)}` : `${fmt(item.raw)} observations`,
            footer: () => "Click to open on iNaturalist",
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
            // One label per month, at the first bin of that month; a single
            // letter when the chart is too narrow for twelve names.
            callback(_value, index) {
              if (index !== 0 && ticks[index] === ticks[index - 1]) return null;
              return this.chart.width < NARROW_CHART_PX ? ticks[index][0] : ticks[index];
            },
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

// A click on a bar opens that organism's observations for the bar's day, week
// or month; a click elsewhere in a column opens every series for it.
function openBar(event, chartInstance) {
  const { request } = state.seasonal;
  const [hit] = chartInstance.getElementsAtEventForMode(event, "nearest", { intersect: true }, false);
  const [column] = hit ? [hit]
    : chartInstance.getElementsAtEventForMode(event, "index", { intersect: false }, false);
  if (!column) return;
  const series = hit ? [request.series[hit.datasetIndex]] : request.series;
  const url = inatUrl(request, series, { view: $("view").value, index: column.index });
  window.open(url, "_blank", "noopener");
}

function areaLabel(area) {
  return area.placeId ? `in ${area.placeName}`
    : `within ${fmt(area.radius)} km of ${area.lat.toFixed(3)}, ${area.lon.toFixed(3)}`;
}

function seasonalTitle(request) {
  const labels = request.series.map((s) => s.label || "All organisms");
  const who = labels.length > 2 ? `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`
    : labels.join(" and ");
  const minus = request.excludeLabel ? `, excluding ${request.excludeLabel}` : "";
  return `${who}${minus} ${areaLabel(request.area)}`;
}

// One request per series; `taxonIds` empty means all organisms.
async function fetchSeasonal(request, taxonIds, { format = "json", signal } = {}) {
  const { area } = request;
  if (area.placeId) {
    const response = await fetch("api/seasonal/region", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        region: area.geometry, date_from: request.dateFrom, date_to: request.dateTo,
        taxon_id: taxonIds, exclude_taxon_id: request.excludeId ? [request.excludeId] : [],
        format,
      }),
      signal,
    });
    return checkResponse(response, format);
  }
  const params = new URLSearchParams({
    lat: area.lat, lon: area.lon, radius: area.radius,
    date_from: request.dateFrom, date_to: request.dateTo, format,
  });
  taxonIds.forEach((id) => params.append("taxon_id", id));
  if (request.excludeId) params.append("exclude_taxon_id", request.excludeId);
  return checkResponse(await fetch(`api/seasonal?${params}`, { signal }), format);
}

async function checkResponse(response, format) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return format === "json" ? response.json() : response.blob();
}

// Wait until the latest place pick has its outline (or was cancelled); a pick
// made while waiting is waited for too.
async function settlePlace() {
  while (state.placeLoading) {
    const pending = state.placeLoading;
    setStatus("Waiting for the place outline…", "busy");
    await pending;
    if (state.placeLoading === pending) state.placeLoading = null;
  }
}

async function runSeasonal(signal) {
  await settlePlace();
  const organisms = await resolveOrganisms(signal);
  const exclude = await resolveTaxon("exclude", signal);
  await settlePlace(); // a place may have been picked during the lookups
  const area = state.place
    ? { placeId: state.place.id, placeName: state.place.name, geometry: state.place.geometry,
        lat: state.place.lat, lon: state.place.lon }
    : { lat: Number($("lat").value), lon: Number($("lon").value), radius: Number($("radius").value) };
  const request = {
    area,
    dateFrom: $("date-from").value,
    dateTo: $("date-to").value,
    series: organisms.length ? organisms.map((t) => ({ taxonId: t.id, label: t.label }))
      : [{ taxonId: null, label: null }],
    excludeId: exclude?.id,
    excludeLabel: exclude?.label,
  };
  setStatus("Counting observations…", "busy");
  const started = performance.now();
  const results = await Promise.all(request.series.map((series) =>
    fetchSeasonal(request, series.taxonId ? [series.taxonId] : [], { signal })));
  state.seasonal = { request, results };
  $("chart-title").textContent = seasonalTitle(request);
  const total = results.reduce((sum, r) => sum + r.total, 0);
  const firsts = results.map((r) => r.first_date).filter(Boolean).sort();
  const lasts = results.map((r) => r.last_date).filter(Boolean).sort();
  const span = firsts.length ? `, ${firsts[0]} to ${lasts.at(-1)}` : "";
  const taxa = results.length === 1 && results[0].taxa_included
    ? ` · ${fmt(results[0].taxa_included)} taxa included` : "";
  // Series can overlap (a genus and one of its species), so with several the
  // figure is a sum of per-organism counts, not a count of distinct observations.
  const counted = results.length > 1
    ? `${fmt(total)} observations summed over ${results.length} organisms`
    : `${fmt(total)} observations`;
  $("chart-summary").textContent = `${counted}${span}${taxa}.`;
  $("download-png").disabled = total === 0;
  $("download-csv").disabled = total === 0;
  renderLegend();
  renderChart();
  // Only while still on the seasonal view: the mode can change mid-search, and
  // an empty result keeps the map out so the area can be adjusted.
  if (state.mode === "seasonal") setMapVisible(!total);
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  setStatus(total ? `Done in ${seconds}s.` : "No observations match this search.");
}

/* ---------- iNaturalist links ---------- */

// Observations on inaturalist.org matching a search, optionally narrowed to
// some series and to one bar's day, week or month of the year.
function inatUrl(request, series, bin) {
  const filter = state.allLicenses ? INAT_ALL_LICENSES_FILTER : INAT_EXPORT_FILTER;
  const p = new URLSearchParams({ ...filter, d1: request.dateFrom, d2: request.dateTo });
  const ids = series.map((s) => s.taxonId).filter(Boolean);
  if (ids.length) p.set("taxon_ids", ids.join(","));
  if (request.excludeId) p.set("without_taxon_id", request.excludeId);
  const { area } = request;
  if (area.placeId) p.set("place_id", area.placeId);
  else { p.set("lat", area.lat); p.set("lng", area.lon); p.set("radius", area.radius); }
  if (bin) {
    const { view, index } = bin;
    if (view === "monthly") p.set("month", index + 1);
    else if (view === "weekly") p.set("week", index + 1); // both use ISO weeks
    else {
      const date = new Date(Date.UTC(LEAP_YEAR, 0, index + 1));
      p.set("month", date.getUTCMonth() + 1);
      p.set("day", date.getUTCDate());
    }
  }
  return `${INAT_OBSERVATIONS}?${p}`;
}

function renderLegend() {
  const { request, results } = state.seasonal;
  const legend = $("series-legend");
  const multiple = request.series.length > 1;
  legend.hidden = !multiple;
  legend.innerHTML = multiple ? request.series.map((series, i) =>
    `<li><span class="swatch" style="background:${seriesColor(i)}"></span>` +
    `<a href="${escapeHtml(inatUrl(request, [series]))}" target="_blank" rel="noopener">` +
    `${escapeHtml(series.label)}</a> <span class="count">${fmt(results[i].total)}</span></li>`).join("") : "";
  $("inat-link").href = inatUrl(request, request.series);
  $("inat-link-row").hidden = false;
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
  // Leave out a tooltip that is showing from the last hover or click.
  chart.tooltip.setActiveElements([], { x: 0, y: 0 });
  chart.setActiveElements([]);
  chart.$glow = [];
  chart.$glowKey = "";
  chart.update("none");
  const source = $("chart");
  const { request, results } = state.seasonal;
  const font = getComputedStyle(document.body).fontFamily;
  const pad = 20;
  const scale = source.width / source.clientWidth;
  const ctx0 = document.createElement("canvas").getContext("2d");
  ctx0.font = `${12 * scale}px ${font}`;
  // Lay out the legend (only for several organisms) in rows that fit the width.
  const legendRows = [];
  if (request.series.length > 1) {
    let row = [];
    let x = 0;
    request.series.forEach((series, i) => {
      const text = `${series.label} (${fmt(results[i].total)})`;
      const width = (14 + 6) * scale + ctx0.measureText(text).width + 16 * scale;
      if (row.length && x + width > source.width) { legendRows.push(row); row = []; x = 0; }
      row.push({ text, color: seriesColor(i), x });
      x += width;
    });
    legendRows.push(row);
  }
  const titleHeight = 48 + legendRows.length * 20;
  const out = document.createElement("canvas");
  out.width = source.width + pad * 2 * scale;
  out.height = source.height + (titleHeight + pad) * scale;
  const ctx = out.getContext("2d");
  ctx.fillStyle = cssVar("--surface");
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.fillStyle = cssVar("--text");
  ctx.font = `600 ${15 * scale}px ${font}`;
  ctx.fillText($("chart-title").textContent, pad * scale, 24 * scale);
  ctx.fillStyle = cssVar("--text-2");
  ctx.font = `${12 * scale}px ${font}`;
  ctx.fillText(`${$("chart-summary").textContent} Source: iNaturalist via visualizer.dikarya.us`,
    pad * scale, 42 * scale);
  legendRows.forEach((row, r) => {
    const y = (62 + r * 20) * scale;
    row.forEach((item) => {
      const x = pad * scale + item.x;
      ctx.fillStyle = item.color;
      ctx.fillRect(x, y - 10 * scale, 14 * scale, 10 * scale);
      ctx.fillStyle = cssVar("--text-2");
      ctx.fillText(item.text, x + 20 * scale, y);
    });
  });
  ctx.drawImage(source, pad * scale, titleHeight * scale);
  out.toBlob((blob) => downloadBlob(blob, "inat-seasonal.png"), "image/png");
}

async function downloadCsv() {
  if (!state.seasonal) return;
  setStatus("Preparing CSV…", "busy");
  try {
    const { request } = state.seasonal;
    const ids = request.series.map((series) => series.taxonId).filter(Boolean);
    downloadBlob(await fetchSeasonal(request, ids, { format: "csv" }), "inat-observations.csv");
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
  clearRegion();
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
  const organisms = await resolveOrganisms(signal);
  if (!organisms.length) throw new Error("Choose an organism to map where it is observed most.");
  const exclude = await resolveTaxon("exclude", signal);
  const [start, end] = seasonRange();
  // Several organisms are mapped together, as one combined group.
  const request = {
    taxonIds: organisms.map((t) => t.id),
    organismLabels: organisms.map((t) => t.label),
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
    season_start: request.seasonStart, season_end: request.seasonEnd,
    date_from: request.dateFrom, date_to: request.dateTo, cell_deg: request.cellDeg,
  });
  request.taxonIds.forEach((id) => params.append("taxon_id", id));
  if (request.excludeId) params.append("exclude_taxon_id", request.excludeId);
  setStatus("Scanning every observation for this time of year…", "busy");
  const started = performance.now();
  const data = await fetchJson(`api/abundance?${params}`, { signal });
  state.abundance = { request, data };
  state.selectedCell = null;
  $("graph-cell").disabled = true;
  if (highlightRect) { map.removeLayer(highlightRect); highlightRect = null; }
  const who = request.organismLabels.length > 1
    ? `${request.organismLabels.slice(0, -1).join(", ")} or ${request.organismLabels.at(-1)}`
    : request.organismLabels[0];
  $("abundance-title").textContent = `Where ${who} is observed, ${request.seasonLabel}`;
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
  // Repeated organism/organism_name pairs, in series order.
  const organisms = state.mode === "seasonal"
    ? request.series.filter((s) => s.taxonId).map((s) => [s.taxonId, s.label])
    : request.taxonIds.map((id, i) => [id, request.organismLabels[i]]);
  for (const [id, label] of organisms) {
    p.append("organism", id);
    p.append("organism_name", label || `Taxon ${id}`);
  }
  if (request.excludeId) {
    p.set("exclude", request.excludeId);
    p.set("exclude_name", request.excludeLabel || `Taxon ${request.excludeId}`);
  }
  p.set("from", request.dateFrom);
  p.set("to", request.dateTo);
  if (state.mode === "seasonal") {
    const { area } = request;
    if (area.placeId) {
      p.set("place", area.placeId);
      p.set("place_name", area.placeName);
    } else {
      p.set("lat", area.lat);
      p.set("lon", area.lon);
      p.set("radius", area.radius);
    }
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
  const names = p.getAll("organism_name");
  p.getAll("organism").forEach((value, i) => {
    const id = Number(value);
    if (id) addOrganism({ id, label: names[i] || `Taxon ${id}` });
  });
  const excludeId = Number(p.get("exclude"));
  if (excludeId) {
    state.taxa.exclude = { id: excludeId, label: p.get("exclude_name") || `Taxon ${excludeId}` };
    $("exclude").value = state.taxa.exclude.label;
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
  // A region permalink needs the outline from iNat before it can search.
  const placeId = Number(p.get("place"));
  let ready = Promise.resolve(true);
  if (placeId) {
    const name = p.get("place_name") || `Place ${placeId}`;
    $("place").value = name;
    ready = selectRegion(placeId, name).then((ok) => {
      if (!ok) setStatus(`iNaturalist has no outline for ${name}.`, "error");
      return ok;
    });
  }
  return { hasPermalink: p.has("mode"), ready };
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
  regionLayer.setStyle({ color: selectionColor() });
  if (highlightRect) highlightRect.setStyle({ color: selectionColor() });
});

let dateToEdited = false;
$("date-to").addEventListener("input", () => { dateToEdited = true; });

async function loadStats() {
  try {
    const stats = await fetchJson("api/stats");
    state.allLicenses = Boolean(stats.all_licenses);
    const newest = stats.newest_date ? ` · newest ${stats.newest_date}` : "";
    $("stats").textContent = `${fmt(stats.total_observations)} observations${newest} · updated ${stats.updated} · v${stats.version}`;
    // Default "To" to the newest observation unless the user already changed it.
    if (stats.newest_date && !dateToEdited) $("date-to").value = stats.newest_date;
  } catch (error) {
    $("stats").textContent = `Database unavailable: ${error.message}`;
  }
}

/* ---------- start ---------- */

setupOrganismPicker();
renderOrganismChips();
setupTaxonPicker("exclude");
setupPlacePicker();
setupSeasonPicker();
$("date-from").value = DEFAULTS.dateFrom;
$("date-to").value = new Date().toISOString().slice(0, 10);
$("lat").value = DEFAULTS.lat.toFixed(4);
$("lon").value = DEFAULTS.lon.toFixed(4);
$("radius").value = DEFAULTS.radius;
updateSelectionOverlay();

const permalink = readUrl();
Promise.all([loadStats(), permalink.ready]).then(([, placeReady]) => {
  // A permalink's own date range wins over the newest-date default, but not
  // over an edit the user made while the stats were loading.
  const p = new URLSearchParams(location.search);
  if (p.get("to") && !dateToEdited) $("date-to").value = p.get("to");
  if (permalink.hasPermalink && placeReady) $("search-form").requestSubmit();
});
