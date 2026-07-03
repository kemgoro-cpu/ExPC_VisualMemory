const state = {
  events: [], sessions: new Map(), selected: new Map(), query: "", activePack: null,
  redact: null, regionEditor: null, collapsedSessions: new Set(), knownSessions: new Set(),
  lastSelected: null, draggedBasketId: null, dragOverBasketId: null,
  offset: 0, hasMore: false, totalEvents: null, loadingEvents: false, status: null,
};
const $ = (id) => document.getElementById(id);
const csrf = () => document.cookie.split("; ").find(v => v.startsWith("vm_csrf="))?.split("=")[1] || "";
const maxDocumentImages = Number(document.body.dataset.maxPackImages || 200);
const warningDocumentImages = Number(document.body.dataset.packWarningImages || 50);

async function api(path, options = {}) {
  const opts = { ...options, headers: { ...(options.headers || {}) } };
  if (opts.body && typeof opts.body !== "string") {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  if (opts.method && !["GET", "HEAD"].includes(opts.method)) opts.headers["X-CSRF-Token"] = csrf();
  const response = await fetch(path, opts);
  if (!response.ok) {
    let message = response.statusText;
    try { message = (await response.json()).detail || message; } catch {}
    throw new Error(message);
  }
  return response.headers.get("content-type")?.includes("application/json") ? response.json() : response;
}

function toast(message, error = false) {
  const el = $("toast"); el.textContent = message; el.style.background = error ? "#ff9b9b" : "#e9eef0";
  el.classList.add("show"); setTimeout(() => el.classList.remove("show"), 3500);
}
function formatBytes(bytes) {
  if (!bytes) return "0 B"; const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** i).toFixed(i > 2 ? 1 : 0)} ${units[i]}`;
}
function formatTime(value) { return new Date(value).toLocaleString("ja-JP", { dateStyle: "short", timeStyle: "medium" }); }
function escapeHtml(value = "") { const el = document.createElement("div"); el.textContent = value; return el.innerHTML; }
function packStatusLabel(status) { return ({approved:"利用可能",expired:"期限切れ",revoked:"共有停止",draft:"未公開"})[status] || status; }

function syncCaptureControls() {
  const status = state.status; if (!status) return;
  const capture = status.capture;
  const loadingModels = [status.ocr, status.embeddings].some(value => value.state === "loading");
  const checkingStorage = ["pending", "scanning"].includes(status.storage.state);
  const stopped = ["stopped", "failed"].includes(capture.state);
  $("startCapture").disabled = !stopped || loadingModels || checkingStorage || !$("deviceSelect").value;
  $("stopCapture").disabled = stopped;
  $("deviceSelect").disabled = !stopped;
  $("refreshDevices").disabled = !stopped;
  $("regionSettings").disabled = !stopped;
}

async function refreshStatus() {
  try {
    const status = await api("/api/status"); const capture = status.capture;
    const badge = $("captureBadge"); badge.className = `status-badge ${capture.state}`;
    badge.innerHTML = `<span></span>${({running:"記録中", reconnecting:"再接続中", starting:"開始中", stopped:"停止中"})[capture.state] || capture.state}`;
    $("metrics").innerHTML = `
      <div class="metric"><strong>${status.processor.processed}</strong><span>処理イベント</span></div>
      <div class="metric"><strong>${status.processor.queue_depth}</strong><span>索引待ち</span></div>
      <div class="metric"><strong>${formatBytes(status.storage.used_bytes)}</strong><span>ローカル保存</span></div>
      <div class="metric"><strong>${formatBytes(status.storage.disk_free_bytes)}</strong><span>空き容量</span></div>`;
    const warnings = [];
    if (status.ocr.state === "loading") warnings.push("OCRモデルを準備中です。準備が終わると記録を開始できます。");
    else if (!status.ocr.available) warnings.push(`OCR未導入: ${status.ocr.reason}`);
    if (status.embeddings.state === "loading") warnings.push("意味検索モデルを準備中です。文字検索は先に利用できます。");
    else if (!status.embeddings.available) warnings.push(`意味検索未導入: ${status.embeddings.reason}`);
    if (["pending", "scanning"].includes(status.storage.state)) warnings.push("保存容量を確認中です。");
    if (status.processor.last_error) warnings.push(`索引処理: ${status.processor.last_error}`);
    if (status.security.bitlocker.status === "checking") warnings.push("BitLockerの状態を確認中です。");
    else if (status.security.bitlocker.status !== "protected") warnings.push("BitLockerの保護を確認できません。権限とドライブ暗号化を確認してください。");
    if (capture.last_error) warnings.push(`キャプチャー: ${capture.last_error}`);
    $("systemWarnings").innerHTML = warnings.map(value => `<div class="warning">${escapeHtml(value)}</div>`).join("");
    state.status = status; syncCaptureControls();
  } catch (error) { console.error(error); }
}

async function loadDevices() {
  const button = $("refreshDevices"); button.disabled = true;
  try {
    const { devices } = await api("/api/devices"); const select = $("deviceSelect");
    const previous = select.value || localStorage.getItem("visual-memory-device") || "";
    select.innerHTML = `<option value="">キャプチャーデバイスを選択</option>` + devices.map(name => `<option>${escapeHtml(name)}</option>`).join("");
    if (devices.includes(previous)) select.value = previous;
    if (!devices.length) toast("映像入力が見つかりません。接続とドライバーを確認してください。", true);
  } catch (error) { toast(error.message, true); } finally { button.disabled = false; syncCaptureControls(); }
}

async function searchEvents(append = false) {
  if (state.loadingEvents) return;
  const query = $("searchQuery").value.trim(); state.query = query;
  if (!append) { state.offset = 0; state.hasMore = false; }
  const params = new URLSearchParams({ q: query, limit: "60", offset: String(state.offset) });
  if ($("searchStart").value) params.set("start", new Date($("searchStart").value).toISOString());
  if ($("searchEnd").value) params.set("end", new Date($("searchEnd").value).toISOString());
  state.loadingEvents = true; $("loadMore").disabled = true;
  try {
    const data = await api(`/api/events?${params}`);
    if (append) {
      const known = new Set(state.events.map(event => event.id));
      state.events.push(...data.events.filter(event => !known.has(event.id)));
      (data.sessions || []).forEach(session => state.sessions.set(session.id, session));
    } else {
      state.events = data.events;
      state.sessions = new Map((data.sessions || []).map(session => [session.id, session]));
    }
    state.offset = data.next_offset; state.hasMore = data.has_more; state.totalEvents = data.total;
    const orderedSessions = [...new Set(state.events.map(event => event.session_id))];
    orderedSessions.forEach((sessionId, index) => {
      if (!state.knownSessions.has(sessionId) && index > 0) state.collapsedSessions.add(sessionId);
      state.knownSessions.add(sessionId);
    });
    renderEvents();
  } catch (error) { toast(error.message, true); }
  finally { state.loadingEvents = false; $("loadMore").disabled = false; $("loadMore").hidden = !state.hasMore; }
}

function eventGroups() {
  const groups = new Map();
  state.events.forEach(event => {
    if (!groups.has(event.session_id)) groups.set(event.session_id, []);
    groups.get(event.session_id).push(event);
  });
  return [...groups.entries()]
    .map(([sessionId, events]) => ({
      sessionId,
      session: state.sessions.get(sessionId) || {},
      events: events.sort((a, b) => new Date(a.started_at) - new Date(b.started_at)),
    }))
    .sort((a, b) => new Date(b.session.started_at || b.events[0].started_at) - new Date(a.session.started_at || a.events[0].started_at));
}

function renderEvents() {
  const countLabel = state.totalEvents == null ? `${state.events.length}${state.hasMore ? "+" : ""}` : `${state.events.length} / ${state.totalEvents}`;
  $("resultsMeta").textContent = `${countLabel} 件${state.query ? ` · 「${state.query}」` : " · 新しい順"}`;
  $("eventGrid").innerHTML = eventGroups().map(({sessionId, session, events}) => {
    const collapsed = state.collapsedSessions.has(sessionId);
    const selectedCount = events.filter(event => state.selected.has(event.id)).length;
    const allSelected = selectedCount === events.length;
    const total = Number(session.event_count || events.length);
    const selectLabel = allSelected ? "全解除" : (state.query ? `一致した${events.length}枚を全選択` : "全選択");
    const started = session.started_at || events[0].started_at;
    const ended = session.ended_at || events.at(-1).ended_at;
    return `<section class="recording-session" data-session="${sessionId}">
      <header class="recording-header">
        <button class="session-toggle" data-toggle-session="${sessionId}" aria-expanded="${!collapsed}">${collapsed ? "▶" : "▼"}</button>
        <div class="recording-title"><strong>録画 ${formatTime(started)}</strong><span>${formatTime(ended)}まで · ${escapeHtml(session.source_name || "画面入力")} · ${state.query ? `${events.length}/${total}件一致` : `${total}枚`}</span></div>
        <span class="session-selected">${selectedCount ? `${selectedCount}枚選択中` : ""}</span>
        <button class="ghost session-select" data-select-session="${sessionId}">${selectLabel}</button>
      </header>
      ${collapsed ? "" : `<div class="session-event-grid" data-lasso-session="${sessionId}">
        ${events.map(event => `<article class="event-card ${state.selected.has(event.id) ? "selected" : ""}" data-event-id="${event.id}">
          <div class="event-image"><img draggable="false" loading="lazy" src="/api/events/${event.id}/frame?thumbnail=true" alt=""><span class="selection-mark">✓</span><span class="event-score">${event.event_kind}${event.score ? ` · ${(event.score * 1000).toFixed(1)}` : ""}</span></div>
          <div class="event-body"><div class="event-time">${formatTime(event.started_at)}</div><div class="event-text">${escapeHtml(event.ocr_excerpt || "OCR待ち、または文字なし")}</div>
          <div class="event-actions"><label class="check-label"><input type="checkbox" data-select="${event.id}" ${state.selected.has(event.id) ? "checked" : ""}>選択</label><button class="ghost" data-detail="${event.id}">詳細</button></div></div>
        </article>`).join("")}
      </div>`}
    </section>`;
  }).join("") || `<p class="muted">該当する画面履歴がありません。</p>`;
  document.querySelectorAll("[data-select]").forEach(input => input.addEventListener("change", event => toggleSelected(Number(input.dataset.select), input.checked, event.shiftKey)));
  document.querySelectorAll("[data-detail]").forEach(button => button.addEventListener("click", () => showDetail(Number(button.dataset.detail))));
  document.querySelectorAll("[data-select-session]").forEach(button => button.addEventListener("click", () => toggleSessionSelection(button.dataset.selectSession)));
  document.querySelectorAll("[data-toggle-session]").forEach(button => button.addEventListener("click", () => {
    const sessionId = button.dataset.toggleSession;
    if (state.collapsedSessions.has(sessionId)) state.collapsedSessions.delete(sessionId); else state.collapsedSessions.add(sessionId);
    renderEvents();
  }));
  document.querySelectorAll("[data-lasso-session]").forEach(setupLassoSelection);
}

function addEventsToSelection(events) {
  let limited = false;
  events.forEach(event => {
    if (state.selected.has(event.id)) return;
    if (state.selected.size >= maxDocumentImages) { limited = true; return; }
    state.selected.set(event.id, event);
  });
  if (limited) toast(`1文書の安全上限は${maxDocumentImages}枚です。`, true);
}

function toggleSelected(id, checked, range = false) {
  const event = state.events.find(value => value.id === id) || state.selected.get(id);
  if (!event) return;
  if (range && state.lastSelected?.sessionId === event.session_id) {
    const events = eventGroups().find(group => group.sessionId === event.session_id)?.events || [];
    const from = events.findIndex(item => item.id === state.lastSelected.id);
    const to = events.findIndex(item => item.id === id);
    const selection = events.slice(Math.min(from, to), Math.max(from, to) + 1);
    if (checked) addEventsToSelection(selection); else selection.forEach(item => state.selected.delete(item.id));
  } else if (checked) addEventsToSelection([event]); else state.selected.delete(id);
  state.lastSelected = {id, sessionId: event.session_id};
  renderBasket(); renderEvents();
}

function toggleSessionSelection(sessionId) {
  const events = eventGroups().find(group => group.sessionId === sessionId)?.events || [];
  if (events.every(event => state.selected.has(event.id))) events.forEach(event => state.selected.delete(event.id));
  else addEventsToSelection(events);
  renderBasket(); renderEvents();
}

function syncSelectionCards() {
  document.querySelectorAll(".event-card[data-event-id]").forEach(card => {
    const selected = state.selected.has(Number(card.dataset.eventId));
    card.classList.toggle("selected", selected);
    const input = card.querySelector("[data-select]"); if (input) input.checked = selected;
  });
}

function setupLassoSelection(grid) {
  let drag = null;
  grid.addEventListener("pointerdown", event => {
    if (event.pointerType !== "mouse" || event.button !== 0 || event.target.closest("button,input,label")) return;
    const bounds = grid.getBoundingClientRect();
    drag = {startX:event.clientX, startY:event.clientY, bounds, base:new Map(state.selected), remove:event.altKey, active:false};
    grid.setPointerCapture(event.pointerId);
  });
  grid.addEventListener("pointermove", event => {
    if (!drag) return;
    if (!drag.active && Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY) < 6) return;
    drag.active = true; event.preventDefault();
    let rectangle = grid.querySelector(".selection-rectangle");
    if (!rectangle) { rectangle=document.createElement("div"); rectangle.className="selection-rectangle"; grid.appendChild(rectangle); }
    const left=Math.min(drag.startX,event.clientX), top=Math.min(drag.startY,event.clientY), right=Math.max(drag.startX,event.clientX), bottom=Math.max(drag.startY,event.clientY);
    Object.assign(rectangle.style,{left:`${left-drag.bounds.left}px`,top:`${top-drag.bounds.top}px`,width:`${right-left}px`,height:`${bottom-top}px`});
    state.selected = new Map(drag.base);
    grid.querySelectorAll(".event-card[data-event-id]").forEach(card => {
      const box=card.getBoundingClientRect(); const hit=box.left<right&&box.right>left&&box.top<bottom&&box.bottom>top;
      if (!hit) return; const id=Number(card.dataset.eventId); const item=state.events.find(value=>value.id===id);
      if (drag.remove) state.selected.delete(id); else if (item && (state.selected.has(id) || state.selected.size<maxDocumentImages)) state.selected.set(id,item);
    });
    syncSelectionCards();
  });
  grid.addEventListener("pointerup", () => {
    if (!drag) return; grid.querySelector(".selection-rectangle")?.remove(); const changed=drag.active; drag=null;
    if (changed) { renderBasket(); renderEvents(); }
  });
}
function renderBasket() {
  const selected = [...state.selected.values()];
  $("basketSummary").textContent = `${selected.length}枚選択中${selected.length > warningDocumentImages ? " · 大きな文書になります" : ""}`;
  $("basketSummary").classList.toggle("warning-text", selected.length > warningDocumentImages);
  $("basketItems").innerHTML = selected.map((event, index) => `<div class="basket-item" draggable="true" data-order-event="${event.id}"><span class="drag-handle" title="ドラッグして順序変更">⋮⋮</span><span class="basket-position">${index + 1}</span><img src="/api/events/${event.id}/frame?thumbnail=true"><span>${formatTime(event.started_at)}</span><button data-remove="${event.id}" title="候補から外す">×</button></div>`).join("") || `<p class="muted">検索結果から場面を選択します。</p>`;
  document.querySelectorAll("[data-remove]").forEach(button => button.addEventListener("click", () => { state.selected.delete(Number(button.dataset.remove)); renderBasket(); renderEvents(); }));
  document.querySelectorAll("[data-order-event]").forEach(item => {
    item.addEventListener("dragstart", event => { state.draggedBasketId = Number(item.dataset.orderEvent); item.classList.add("dragging"); event.dataTransfer.effectAllowed = "move"; });
    item.addEventListener("dragover", event => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; item.classList.add("drag-over"); });
    item.addEventListener("dragleave", () => item.classList.remove("drag-over"));
    item.addEventListener("drop", event => { event.preventDefault(); reorderBasket(state.draggedBasketId, Number(item.dataset.orderEvent)); });
    item.addEventListener("dragend", () => { state.draggedBasketId = null; document.querySelectorAll(".basket-item").forEach(value => value.classList.remove("dragging", "drag-over")); });
    const handle = item.querySelector(".drag-handle");
    handle.addEventListener("pointerdown", event => {
      event.preventDefault(); state.draggedBasketId = Number(item.dataset.orderEvent); state.dragOverBasketId = null;
      item.classList.add("dragging"); handle.setPointerCapture(event.pointerId);
    });
    handle.addEventListener("pointermove", event => {
      if (!state.draggedBasketId) return;
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-order-event]");
      document.querySelectorAll(".basket-item").forEach(value => value.classList.remove("drag-over"));
      if (target && Number(target.dataset.orderEvent) !== state.draggedBasketId) {
        state.dragOverBasketId = Number(target.dataset.orderEvent); target.classList.add("drag-over");
      }
    });
    handle.addEventListener("pointerup", event => {
      if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
      const sourceId = state.draggedBasketId, targetId = state.dragOverBasketId;
      state.draggedBasketId = null; state.dragOverBasketId = null;
      if (sourceId && targetId) reorderBasket(sourceId, targetId);
      else document.querySelectorAll(".basket-item").forEach(value => value.classList.remove("dragging", "drag-over"));
    });
  });
}

function setSelectionOrder(events) {
  state.selected = new Map(events.map(event => [event.id, event]));
  renderBasket(); renderEvents();
}

function reorderBasket(sourceId, targetId) {
  if (!sourceId || sourceId === targetId) return;
  const items = [...state.selected.values()];
  const sourceIndex = items.findIndex(event => event.id === sourceId);
  const targetIndex = items.findIndex(event => event.id === targetId);
  if (sourceIndex < 0 || targetIndex < 0) return;
  const [moved] = items.splice(sourceIndex, 1); items.splice(targetIndex, 0, moved);
  setSelectionOrder(items);
}

async function showDetail(id) {
  try {
    const data = await api(`/api/events/${id}`); const event = data.event;
    $("detailContent").innerHTML = `<p class="eyebrow">EVENT ${event.id}</p><h2>${formatTime(event.started_at)}</h2><img class="detail-image" src="/api/events/${event.id}/frame"><pre class="ocr-box">${escapeHtml(event.ocr_text || "OCRテキストなし")}</pre>`;
    $("detailDialog").showModal();
  } catch (error) { toast(error.message, true); }
}

async function createPack() {
  if (!state.selected.size) return toast("場面を1枚以上選択してください。", true);
  if (state.selected.size > warningDocumentImages && !window.confirm(`${state.selected.size}枚を1つの文書にまとめます。ファイルが大きくなる可能性があります。続けますか？`)) return;
  const button = $("createPack"); const label = button.textContent; button.disabled = true; button.textContent = "文書を作成中…";
  try {
    await api("/api/packs", { method: "POST", body: { title: $("packTitle").value, note: $("packNote").value, query: state.query, event_ids: [...state.selected.keys()], deduplicate_overlaps: $("deduplicateOverlaps").checked } });
    state.selected.clear(); renderBasket(); renderEvents(); $("packTitle").value = ""; $("packNote").value = "";
    toast("コンテキスト文書を作成しました。PDF・HTML・MCPですぐ利用できます。"); await showPacks();
  } catch (error) { toast(error.message, true); }
  finally { button.disabled = false; button.textContent = label; }
}

async function showPacks() {
  try {
    const { packs } = await api("/api/packs");
    $("packList").innerHTML = packs.map(pack => `<article class="pack-card"><div class="pack-card-head"><div><span class="pack-status">${packStatusLabel(pack.status)}</span><h3>${escapeHtml(pack.title)}</h3><p class="muted">${pack.item_count}枚 · 重複除去 ${pack.deduplicate_overlaps ? "ON" : "OFF"} · ${formatTime(pack.created_at)}${pack.expires_at ? ` · MCP期限 ${formatTime(pack.expires_at)}` : ""}</p>${pack.build_error ? `<p class="build-error">生成に失敗しました: ${escapeHtml(pack.build_error)}</p>` : ""}</div><button class="ghost" data-open-pack="${pack.id}">開く</button></div></article>`).join("") || `<p class="muted">まだ文書はありません。</p>`;
    document.querySelectorAll("[data-open-pack]").forEach(button => button.addEventListener("click", () => openPack(button.dataset.openPack)));
    $("packsDialog").showModal();
  } catch (error) { toast(error.message, true); }
}

async function openPack(id) {
  try {
    const pack = await api(`/api/packs/${id}`); state.activePack = pack;
    const exportActions = pack.status === "draft" ? `<p class="muted">この文書は未公開です。${pack.build_error ? `生成エラー: ${escapeHtml(pack.build_error)}` : "再生成が完了していません。"}</p>` : `<a href="/api/packs/${id}/document?format=pdf"><button class="primary">PDF文書</button></a><a href="/api/packs/${id}/document?format=html"><button class="ghost">単一HTML</button></a>`;
    const actions = `${exportActions}${pack.status === "approved" ? `<button class="danger" data-revoke="${id}">MCP共有を停止</button>` : ""}`;
    $("packList").innerHTML = `<button class="ghost" id="backToPacks">← 一覧</button><article class="pack-card"><span class="pack-status">${packStatusLabel(pack.status)}</span><h3>${escapeHtml(pack.title)}</h3><p class="muted">${escapeHtml(pack.note)}</p><div class="pack-items">${pack.items.map((item, index) => `<div class="pack-item"><span class="pack-item-position">${index + 1}</span><img src="/api/events/${item.event_id}/frame">${pack.status === "approved" ? `<button data-redact="${item.event_id}">墨消し</button>` : ""}</div>`).join("")}</div><div class="dialog-actions">${actions}</div></article>`;
    $("backToPacks").onclick = showPacks;
    document.querySelectorAll("[data-redact]").forEach(button => button.addEventListener("click", () => beginRedaction(pack, Number(button.dataset.redact))));
    document.querySelector("[data-revoke]")?.addEventListener("click", async () => { try { await api(`/api/packs/${id}/revoke`, {method:"POST"}); toast("MCP共有を停止しました。"); openPack(id); } catch(error) { toast(error.message,true); } });
  } catch (error) { toast(error.message, true); }
}

function beginRedaction(pack, eventId) {
  const item = pack.items.find(value => value.event_id === eventId); const image = new Image();
  image.onload = () => {
    const canvas = $("redactionCanvas"); const maxWidth = Math.min(image.width, 1100); const scale = maxWidth / image.width;
    canvas.width = maxWidth; canvas.height = image.height * scale;
    state.redact = { packId: pack.id, eventId, image, rectangles: JSON.parse(item.redactions_json || "[]"), drawing: null };
    drawRedactions(); $("redactionDialog").showModal();
  };
  image.src = `/api/events/${eventId}/frame`;
}
function drawRedactions() {
  const { image, rectangles, drawing } = state.redact; const canvas = $("redactionCanvas"); const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height); ctx.fillStyle = "rgba(0,0,0,.9)";
  [...rectangles, ...(drawing ? [drawing] : [])].forEach(rect => ctx.fillRect(rect.x * canvas.width, rect.y * canvas.height, rect.width * canvas.width, rect.height * canvas.height));
}
function canvasPoint(event) { const box = $("redactionCanvas").getBoundingClientRect(); return {x:(event.clientX-box.left)/box.width,y:(event.clientY-box.top)/box.height}; }

async function openRegionEditor() {
  try {
    const [{ ignore_regions, watch_regions }, { events }] = await Promise.all([
      api("/api/capture/regions"),
      api("/api/events?limit=1"),
    ]);
    if (!events.length) return toast("領域設定には保存済みのフレームが1枚必要です。", true);
    const image = new Image();
    image.onload = () => {
      const canvas = $("regionCanvas"); const maxWidth = Math.min(image.width, 1100); const scale = maxWidth / image.width;
      canvas.width = maxWidth; canvas.height = image.height * scale;
      state.regionEditor = { image, ignore: ignore_regions, watch: watch_regions, drawing: null };
      drawRegions(); $("regionDialog").showModal();
    };
    image.src = `/api/events/${events[0].id}/frame`;
  } catch (error) { toast(error.message, true); }
}

function drawRegions() {
  const editor = state.regionEditor; if (!editor) return;
  const canvas = $("regionCanvas"); const ctx = canvas.getContext("2d");
  ctx.drawImage(editor.image, 0, 0, canvas.width, canvas.height);
  const draw = (rect, color) => {
    ctx.fillStyle = color; ctx.strokeStyle = color.replace(".24", ".95"); ctx.lineWidth = 2;
    ctx.fillRect(rect.x * canvas.width, rect.y * canvas.height, rect.width * canvas.width, rect.height * canvas.height);
    ctx.strokeRect(rect.x * canvas.width, rect.y * canvas.height, rect.width * canvas.width, rect.height * canvas.height);
  };
  editor.ignore.forEach(rect => draw(rect, "rgba(255,107,107,.24)"));
  editor.watch.forEach(rect => draw(rect, "rgba(166,255,77,.24)"));
  if (editor.drawing) draw(editor.drawing, editor.drawing.kind === "ignore" ? "rgba(255,107,107,.24)" : "rgba(166,255,77,.24)");
}

function regionPoint(event) {
  const box = $("regionCanvas").getBoundingClientRect();
  return { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
}

document.addEventListener("DOMContentLoaded", () => {
  $("refreshDevices").onclick = loadDevices; $("regionSettings").onclick = openRegionEditor; $("startCapture").onclick = async () => { const source_name = $("deviceSelect").value; if (!source_name) return toast("デバイスを選択してください。", true); try { await api("/api/capture/start", {method:"POST",body:{source_name}}); toast("記録を開始しました。"); refreshStatus(); } catch(error) { toast(error.message,true); } };
  $("stopCapture").onclick = async () => { try { await api("/api/capture/stop", {method:"POST"}); toast("記録を停止しました。"); refreshStatus(); } catch(error) { toast(error.message,true); } };
  $("searchForm").onsubmit = event => { event.preventDefault(); searchEvents(); }; $("createPack").onclick = createPack; $("showPacks").onclick = showPacks;
  $("loadMore").onclick = () => searchEvents(true);
  $("deviceSelect").onchange = () => { if ($("deviceSelect").value) localStorage.setItem("visual-memory-device", $("deviceSelect").value); syncCaptureControls(); };
  $("sortChronological").onclick = () => setSelectionOrder([...state.selected.values()].sort((a,b) => new Date(a.started_at) - new Date(b.started_at)));
  $("reverseOrder").onclick = () => setSelectionOrder([...state.selected.values()].reverse());
  document.querySelectorAll("[data-close]").forEach(button => button.onclick = () => $(button.dataset.close).close());
  const canvas = $("redactionCanvas");
  canvas.onpointerdown = event => { if (!state.redact) return; const point = canvasPoint(event); state.redact.drawing = {x:point.x,y:point.y,width:0,height:0}; canvas.setPointerCapture(event.pointerId); };
  canvas.onpointermove = event => { if (!state.redact?.drawing) return; const point = canvasPoint(event), start = state.redact.drawing; start.width = point.x - start.x; start.height = point.y - start.y; drawRedactions(); };
  canvas.onpointerup = () => { if (!state.redact?.drawing) return; let {x,y,width,height} = state.redact.drawing; if (width < 0) {x += width; width *= -1;} if (height < 0) {y += height; height *= -1;} if (width > .003 && height > .003) state.redact.rectangles.push({x,y,width,height}); state.redact.drawing = null; drawRedactions(); };
  $("clearRedactions").onclick = () => { if (state.redact) {state.redact.rectangles=[]; drawRedactions();} };
  $("saveRedactions").onclick = async () => { try { const r=state.redact; await api(`/api/packs/${r.packId}/items/${r.eventId}/redactions`,{method:"PUT",body:{redactions:r.rectangles}}); $("redactionDialog").close(); toast("墨消しを保存しました。"); openPack(r.packId); } catch(error) { toast(error.message,true); } };
  const regionCanvas = $("regionCanvas");
  regionCanvas.onpointerdown = event => { if (!state.regionEditor) return; const point=regionPoint(event); state.regionEditor.drawing={...point,width:0,height:0,kind:$("regionMode").value}; regionCanvas.setPointerCapture(event.pointerId); };
  regionCanvas.onpointermove = event => { if (!state.regionEditor?.drawing) return; const point=regionPoint(event),start=state.regionEditor.drawing; start.width=point.x-start.x; start.height=point.y-start.y; drawRegions(); };
  regionCanvas.onpointerup = () => { const editor=state.regionEditor;if(!editor?.drawing)return;let{x,y,width,height,kind}=editor.drawing;if(width<0){x+=width;width*=-1;}if(height<0){y+=height;height*=-1;}if(width>.003&&height>.003)editor[kind].push({x,y,width,height});editor.drawing=null;drawRegions(); };
  $("clearIgnoreRegions").onclick = () => { if(state.regionEditor){state.regionEditor.ignore=[];drawRegions();} };
  $("clearWatchRegions").onclick = () => { if(state.regionEditor){state.regionEditor.watch=[];drawRegions();} };
  $("saveRegions").onclick = async () => { try { const editor=state.regionEditor;await api("/api/capture/regions",{method:"PUT",body:{ignore_regions:editor.ignore,watch_regions:editor.watch}});$("regionDialog").close();toast("差分検出領域を保存しました。"); } catch(error){toast(error.message,true);} };
  renderBasket(); loadDevices(); searchEvents(); refreshStatus(); setInterval(refreshStatus, 5000);
});
