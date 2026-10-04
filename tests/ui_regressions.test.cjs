// Run with: node --test tests/ui_regressions.test.cjs
// Executes the shipped browser functions; no browser or third-party package is required.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../src/visual_memory/static/app.js'), 'utf8');

function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; }
function page(id, more = false, offset = 100) { return {events:[{id,session_id:'s',started_at:'2026-10-04T00:00:00Z'}],sessions:[],next_offset:offset,has_more:more,total:more ? 200 : 1}; }
function response(data) { return {ok:true,headers:{get:()=> 'application/json'},json:async()=>data}; }

function harness(fetch = async () => response(page(1)), render = false) {
  const nodes = new Map(), storage = new Map(), notices = [], requests = [];
  const makeNode = () => ({value:'',checked:false,hidden:false,disabled:false,textContent:'',innerHTML:'',dataset:{},style:{},classList:{toggle(){},add(){},remove(){}},addEventListener(){}});
  const document = {cookie:'vm_csrf=test',body:{dataset:{}},activeElement:null,addEventListener(){},querySelectorAll:()=>[],querySelector:()=>null,
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, makeNode()); return nodes.get(id); },
    createElement() { const n = makeNode(); Object.defineProperty(n,'innerHTML',{get:()=>String(n.textContent).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}); return n; }};
  const context = {console,URLSearchParams,AbortController,AbortSignal,CSS:{escape:x=>x},document,
    crypto:{randomUUID:()=> 'same-request-id'},
    localStorage:{getItem:key=>storage.get(key) ?? null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
    fetch:(url,opts)=>{requests.push({url,opts});return fetch(url,opts);},
    window:{confirm:()=>true,scrollX:0,scrollY:120,scrollTo(){}},
    setTimeout:fn=>{queueMicrotask(fn);return 1;},clearTimeout(){}};
  vm.createContext(context);
  vm.runInContext(source + '\nglobalThis.exposed={state,searchEvents,loadEvents,toggleSelected,submitManualCapture,createPack,refreshStatus,downloadPdf,packSizeLabel,startNewPack};',context);
  context.recordNotice = (message,error)=>notices.push({message,error});
  vm.runInContext('toast=recordNotice; renderBasket=()=>{};' + (render ? '' : 'renderEvents=()=>{};'),context);
  return {context,document,nodes,storage,notices,requests,...context.exposed,element:id=>document.getElementById(id),stub:(name,fn)=>{context.stubFunction=fn;vm.runInContext(`${name}=stubFunction`,context);}};
}

test('a newer search starts at zero and rejects a late older result even when abort is ignored', async () => {
  const a=deferred(), b=deferred();
  const h=harness(url=>url.includes('q=alpha') ? a.promise : b.promise);
  h.element('searchQuery').value='alpha'; const old=h.searchEvents();
  h.element('searchQuery').value='beta'; const current=h.searchEvents();
  assert.equal(h.requests[0].opts.signal.aborted,true);
  assert.match(h.requests[1].url,/q=beta&limit=100&offset=0/);
  b.resolve(response(page(2))); await current;
  a.resolve(response(page(1,true))); await old;
  assert.equal(h.state.query,'beta'); assert.deepEqual(Array.from(h.state.events,x=>x.id),[2]);
  assert.equal(h.requests.length,2); assert.equal(h.state.loadingEvents,false);
});

test('automatic pagination keeps its initial query and date snapshot while the form is edited', async () => {
  const first=deferred(); const h=harness((url)=>h.requests.length===1 ? first.promise : Promise.resolve(response(page(2))));
  h.element('searchQuery').value='alpha'; h.element('searchStart').value='2026-10-01T10:00';
  const search=h.searchEvents(); h.element('searchQuery').value='beta'; h.element('searchStart').value='2026-10-02T10:00';
  first.resolve(response(page(1,true))); await search;
  const initial=new URLSearchParams(h.requests[0].url.split('?')[1]), next=new URLSearchParams(h.requests[1].url.split('?')[1]);
  assert.equal(next.get('q'),'alpha'); assert.equal(next.get('start'),initial.get('start')); assert.equal(next.get('offset'),'100');
});

test('manual load more uses the submitted search, not unsent edits', async () => {
  const h=harness(async()=>response(page(h.requests.length,h.requests.length===1)));
  h.state.showAll=false; h.element('searchQuery').value='alpha'; await h.searchEvents();
  h.element('searchQuery').value='beta'; await h.loadEvents(true);
  assert.match(h.requests[1].url,/q=alpha&limit=60&offset=100/);
});

test('selection restores the focused card after replacing the result DOM', () => {
  const h=harness(undefined,true); h.state.events=page(1).events;
  const card={dataset:{eventId:'1'},closest(){return this;}};
  h.document.activeElement=card; let focused;
  Object.defineProperty(h.element('eventGrid'),'innerHTML',{set(){h.document.activeElement={tagName:'BODY'};}});
  h.document.querySelector=selector=>({focus(options){focused={selector,options};h.document.activeElement=card;}});
  h.toggleSelected(1,true);
  assert.equal(h.state.selected.has(1),true); assert.equal(h.document.activeElement,card);
  assert.equal(focused.selector,'[data-event-id="1"]'); assert.equal(focused.options.preventScroll,true);
});

function manualHarness(fetch) {
  const h=harness(fetch); h.stub('syncCaptureControls',()=>{});h.stub('triggerCaptureFeedback',()=>{});
  h.stub('refreshStatus',async()=>{});h.stub('refreshHistory',async()=>{});return h;
}

test('manual capture waits for the saved result before announcing success', async () => {
  const completion=deferred(); const h=manualHarness((url)=>url==='/api/capture/manual' ? Promise.resolve(response({state:'pending'})) : completion.promise);
  const pending=h.submitManualCapture(); await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.notices.some(x=>x.message.includes('保存しました')),false);
  completion.resolve(response({state:'saved',event_id:42})); await pending;
  assert.equal(h.notices.some(x=>x.message.includes('保存しました（画面 42）')),true);
  assert.equal(h.state.manualRequestId,null);
});

test('an uncertain manual request is retried with the same id and failed results never announce saved', async () => {
  let attempt=0; const h=manualHarness(async()=>{if (++attempt===1) throw new Error('offline'); return response({state:'failed',error:'disk full'});});
  await h.submitManualCapture(); assert.equal(h.state.manualRequestId,'same-request-id');
  await h.submitManualCapture();
  assert.equal(JSON.parse(h.requests[0].opts.body).request_id,JSON.parse(h.requests[1].opts.body).request_id);
  assert.equal(h.notices.some(x=>x.message.includes('保存しました')),false); assert.equal(h.state.manualRequestId,null);
});

test('a failed document build retries the original draft id and selection order', async () => {
  const h=harness(async url=>url==='/api/packs' ? {ok:false,status:500,statusText:'error',json:async()=>({detail:'build failed',pack_id:'draft-1'})} : response({id:'draft-1'}));
  h.stub('showPacks',async()=>{}); h.state.selected.set(3,{id:3});h.state.selected.set(1,{id:1});
  h.element('packTitle').value='original'; await h.createPack();
  assert.equal(h.state.failedPackId,'draft-1');
  h.state.selected.set(7,{id:7}); h.element('includeAppendix').checked=true; await h.createPack();
  assert.equal(h.requests[1].url,'/api/packs/draft-1/rebuild');
  assert.deepEqual(JSON.parse(h.requests[0].opts.body).event_ids,[3,1]);
  assert.equal(JSON.parse(h.requests[1].opts.body).include_appendix,true);
  assert.deepEqual(Array.from(h.state.selected.keys()),[7]); assert.equal(h.state.failedPackId,null);
});

test('document detail derives its counts from items and pages when list-only counts are absent', () => {
  const h=harness();
  const detail={items:[{event_id:3},{event_id:1}],pages:[{position:0,source_event_ids:[3,1]}]};
  assert.equal(h.packSizeLabel(detail),'元画面 2枚 → 1ページ（1枚を統合）');
  assert.equal(h.packSizeLabel({items:detail.items,pages:[]}),'元画面 2枚');
});

test('explicitly switching away from a failed draft creates a new document from current candidates', async () => {
  const h=harness(async()=>response({id:'new-pack'})); h.stub('showPacks',async()=>{});
  h.state.failedPackId='old-draft';h.state.failedPackDraft={event_ids:[1]};
  h.state.selected.set(7,{id:7});h.element('packTitle').value='new title';
  h.startNewPack();
  assert.equal(h.state.failedPackId,null);assert.equal(h.state.failedPackDraft,null);
  assert.equal(h.element('startNewPack').hidden,true);assert.match(h.element('createPackStatus').textContent,/一覧に残っています/);
  assert.deepEqual(Array.from(h.state.selected.keys()),[7]);
  await h.createPack();
  assert.equal(h.requests[0].url,'/api/packs');
  assert.deepEqual(JSON.parse(h.requests[0].opts.body).event_ids,[7]);
  assert.equal(JSON.parse(h.requests[0].opts.body).title,'new title');
});

test('status failure replaces the previous recording badge with an unknown connection state', async () => {
  const h=harness(async()=>{throw new Error('offline');});
  h.state.status={capture:{state:'running'}};h.state.lastStatusAt='2026-10-04T00:00:00Z';
  h.stub('syncCaptureControls',()=>{});h.stub('stopPreviewPolling',()=>{});
  await h.refreshStatus();
  assert.equal(h.state.statusConnected,false); assert.equal(h.element('captureBadge').textContent,'接続を確認できません');
  assert.match(h.element('systemWarnings').textContent,/現在の記録状態は不明/);
});

test('PDF errors stay in the document view and enable retry on the same document', async () => {
  const h=harness(async()=>({ok:false,status:500,statusText:'error',json:async()=>({detail:'PDF renderer failed'})}));
  await h.downloadPdf({id:'p1',title:'document'});
  assert.equal(h.element('downloadPdf').disabled,false); assert.equal(h.element('downloadPdf').textContent,'PDF生成を再試行');
  assert.match(h.element('pdfStatus').textContent,/PDF renderer failed/);
  await h.downloadPdf({id:'p1',title:'document'});
  assert.equal(h.requests[0].url,h.requests[1].url);
});
