const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const root = require('node:path').join(__dirname, '..');

async function main() {
    let stored = {schema_version: 1, saved_at: '2026-09-30T00:00:00Z', papers: [], questions: [
        {id:'one', stem:'Actual source question?', options:{A:'one', B:'two'}, correct_answer:'A',
            trusted_answer:true, answer_source:'past_paper_key', subject:'Ortho', topic:'Growth', year:'2025', stem_hash:'same', repeat_count:2},
        {id:'two', stem:'Another source question?', options:{A:'one', B:'two'}, correct_answer:'B',
            trusted_answer:false, answer_source:'ai_prepared', subject:'Operative', topic:'Caries', year:'2024', stem_hash:'other', repeat_count:1},
        {id:'three', stem:'Repeated source question?', options:{A:'one', B:'two'}, correct_answer:'A',
            trusted_answer:true, answer_source:'past_paper_key', subject:'Ortho', topic:'Growth', year:'2025', stem_hash:'same', repeat_count:2}
    ]};
    const reliable = {checked:true};
    let requests = 0;
    const database = {transaction() {
        const tx = {objectStore() { return {get() { const request={result:stored}; queueMicrotask(()=>tx.oncomplete()); return request; },
            delete() {stored=undefined; const request={}; queueMicrotask(()=>tx.oncomplete()); return request;}}; }};
        return tx;
    }};
    const navigator = {onLine:false};
    const window = {addEventListener(){}};
    const sandbox = vm.createContext({window,navigator,Response,Blob,AbortController,setTimeout,clearTimeout,console,
        document:{getElementById:id=>id==='offlineReliableOnly'?reliable:null},
        localStorage:{getItem(){return null;},setItem(){},removeItem(){}},
        indexedDB:{open(){const request={result:database};queueMicrotask(()=>request.onsuccess());return request;}},
        fetch:async()=>{requests++;return new Response('{}',{status:401});}});
    vm.runInContext(fs.readFileSync(root+'/docs/offline.js','utf8'),sandbox);
    const app = window.DentoraOffline;
    await app.ready;
    const catalog = await (await app.request('catalog','https://api/test/catalog')).json();
    assert.equal(catalog.question_count,3);
    assert.equal(catalog.test_ready_questions,2);
    assert.equal(catalog.eligible_test_questions,3);
    const strict = await (await app.request('start','https://api/test/start',{body:JSON.stringify({count:10})})).json();
    assert.equal(strict.count,1); // Repeated stems are not repeated in the same test.
    assert.ok(['Actual source question?', 'Repeated source question?'].includes(strict.questions[0].stem));
    assert.equal('correct_answer' in strict.questions[0],false);
    assert.equal('verification_rationale' in strict.questions[0],false);
    reliable.checked=false;
    const selected = await (await app.request('start','https://api/test/start',{body:JSON.stringify({subject:'Operative'})})).json();
    assert.equal(selected.questions[0].id,'two');
    const grade = await (await app.request('grade','https://api/test/grade',{body:JSON.stringify({responses:[
        {question_id:'one',answer:'a'},{question_id:'two',answer:'A'}]})})).json();
    assert.equal(grade.score_percent,50);
    assert.equal(grade.details[1].answer_source,'ai_prepared');
    assert.equal(requests,0); // Offline tests and grading never call a provider/API.
    await assert.rejects(app.request('grade','https://api/test/grade',{body:JSON.stringify({responses:[{question_id:'missing'}]})}), /not in your saved pack/);
    navigator.onLine=true;
    assert.equal((await app.request('catalog','https://api/test/catalog')).status,401); // Do not hide an auth failure with cached data.
    assert.equal(requests,1);
    // An online test must never silently switch to the saved pack's different key.
    sandbox.fetch = async () => {throw new Error('network down');};
    await assert.rejects(app.request('grade','https://api/test/grade', {body:JSON.stringify({
        test_token:'frozen-online-key',responses:[{question_id:'one',answer:'A'}]})}), /connect/);
    assert.equal((await (await app.request('catalog','https://api/test/catalog')).json()).offline,true);
    await app.remove();
    navigator.onLine=false;
    await assert.rejects(app.request('catalog','https://api/test/catalog'),/save a question pack/);
    navigator.onLine=true;
    let timeoutCallback, timeoutMs;
    let bodyStarted;
    const downloadingBody = new Promise(resolve => bodyStarted=resolve);
    sandbox.setTimeout = (fn, ms) => {timeoutCallback=fn; timeoutMs=ms; return 1;};
    sandbox.clearTimeout = () => {};
    sandbox.fetch = async (_url, options) => ({status:200, headers:{},
        arrayBuffer: () => new Promise((_resolve,reject) => {
            options.signal.addEventListener('abort', () => reject(Object.assign(new Error('signal is aborted without reason'), {name:'AbortError'})));
            bodyStarted();
        })});
    const slowBody = app.fetch('https://api/test/catalog');
    await downloadingBody;
    assert.equal(timeoutMs,120000);
    timeoutCallback();
    await assert.rejects(slowBody, /server took too long/); // Deadline includes body download.

    const handlers = {}, cached = new Map(), unrelated = 'another-app-cache';
    const cache = {addAll:async paths=>paths.forEach(request=>cached.set(request.url,new Response(request.url))),
        match:async request=>cached.get(typeof request==='string'?request:request.url), put:async(request,response)=>cached.set(request.url,response)};
    const self = {location:{href:'https://example.org/DentalStudyAI/sw.js'},addEventListener:(name,fn)=>handlers[name]=fn,
        skipWaiting:async()=>{},clients:{claim:async()=>{}}};
    vm.runInNewContext(fs.readFileSync(root+'/docs/sw.js','utf8'),{self,URL,Response,Request,
        caches:{open:async()=>cache,keys:async()=>['dentora-shell-v0',unrelated],delete:async key=>assert.notEqual(key,unrelated)},
        fetch:async()=>{throw new Error('offline');}});
    let task;
    handlers.install({waitUntil:promise=>task=promise});await task;
    handlers.activate({waitUntil:promise=>task=promise});await task;
    handlers.fetch({request:{method:'GET',url:'https://example.org/DentalStudyAI/'},respondWith:promise=>task=promise});
    assert.equal((await task).status,200);
    let intercepted=false;
    handlers.fetch({request:{method:'GET',url:'https://api.example.org/test/offline-pack'},respondWith:()=>intercepted=true});
    assert.equal(intercepted,false);
    console.log('Offline filtering, answer hiding, grading, provenance, online-key isolation, timeout recovery, auth handling, storage removal, and shell fallback passed.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
