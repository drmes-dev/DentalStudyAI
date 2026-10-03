'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../docs/index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
function functionSource(name) {
    const pattern = new RegExp('(?:async )?function ' + name + '\\(');
    const start = script.search(pattern);
    assert.ok(start >= 0, name);
    const rest = script.slice(start);
    const next = rest.slice(1).search(/\n(?:async )?function [A-Za-z]|\n\/\* =/);
    return next < 0 ? rest : rest.slice(0, next + 1);
}
function element() {
    const classes = new Set();
    return {style: {}, classList: {add: c => classes.add(c), remove: c => classes.delete(c),
        contains: c => classes.has(c), toggle: (c, value) => value ? classes.add(c) : classes.delete(c)},
        value: '', disabled: false, innerHTML: '', textContent: '', querySelectorAll: () => [], focus() {}};
}
async function main() {
    const stored = new Map(), toasts = [], attempts = [];
    let requestCount = 0, intervals = 0, resolveRequest;
    const sandbox = {console, Date, JSON, Math, Number, String, Array, Object, Response,
        activeTest: null, activeTestIndex: 0, activeTestAnswers: {}, activeTestStartedAt: 0,
        activeTestDurationSeconds: 60, activeTestTimer: null, testSubmitting: false, testStarting: false,
        currentMode: 'test', currentChatMode: 'study', currentMessages: [], isSending: false,
        modeInfo: Object.fromEntries(['study','mcq','viva','osce','test','pdf'].map(m=>[m,{title:m, subtitle:m, placeholder:m}])),
        localStorage: {setItem: (k,v)=>stored.set(k,v), getItem: k=>stored.get(k), removeItem: k=>stored.delete(k)},
        setInterval: () => {intervals++; return intervals;}, clearInterval() {},
        showToast: text => toasts.push(text), storeTestAttempt: grade => attempts.push(grade),
        renderTestResults: () => {}, loadTestCatalog() {}, renderReadinessDashboard() {},
        closeMobileMenu() {}, clearMessagesUI() {}, saveCurrentChat() {}, showPdfPanel() {}, uploadedPdf: null,
        handleBetaAuthFailure() {}, saveHistory: () => true, renderHistory() {}, chatHistory: [], currentChatId: null,
        betaHeaders: () => ({}), dentoraSessionId: 'visitor', TEST_GRADE_API: '/test/grade',
        escapeHtml: x => String(x), document: {querySelectorAll: () => []},
        DentoraOffline: {request: () => {requestCount++; return new Promise(resolve=>resolveRequest=resolve);},
            responseError: async r => new Error((await r.json()).detail)},
    };
    for (const name of ['mainWorkspace','welcome','testWorkspace','chatContainer','testDashboard',
        'testSession','testResults','testTimer','testQuestionCard','testProgress','testPreviousButton',
        'testNextButton','submitTestButton','pageTitle','pageSubtitle','messageInput']) sandbox[name] = element();
    const context = vm.createContext(sandbox);
    for (const name of ['showChatInterface','showWelcomeInterface','setMode','stopTestTimer',
        'saveActiveTest','restoreActiveTest','updateTestTimer','renderActiveTestQuestion','submitActiveTest','clearCurrentChat']) {
        vm.runInContext(functionSource(name), context);
    }
    // A late chat reply or welcome callback cannot hide the Test workspace.
    sandbox.testWorkspace.classList.add('show');
    sandbox.showChatInterface(); sandbox.showWelcomeInterface();
    assert.equal(sandbox.testWorkspace.classList.contains('show'), true);
    for (const mode of ['study','mcq','viva','osce','pdf','test']) {
        sandbox.setMode(mode);
        assert.equal(sandbox.currentMode, mode);
        assert.equal(sandbox.testWorkspace.classList.contains('show'), mode === 'test');
    }
    sandbox.activeTest = {questions: [{id:'one', stem:'Actual question', options:{A:'One',B:'Two'}}], test_token:'frozen'};
    sandbox.activeTestStartedAt = Date.now() - 61000;
    sandbox.activeTestAnswers = {one:'B'};
    sandbox.saveActiveTest();
    sandbox.updateTestTimer();
    sandbox.updateTestTimer();
    assert.equal(requestCount, 1, 'Expired test submits once');
    resolveRequest(new Response(JSON.stringify({detail:'Temporarily unavailable'}), {status:503}));
    await new Promise(resolve=>setImmediate(resolve));
    assert.equal(sandbox.testSubmitting, false);
    assert.equal(sandbox.submitTestButton.disabled, false);
    assert.equal(intervals, 0, 'Expired failed submission must not restart its timer');
    sandbox.updateTestTimer();
    assert.equal(requestCount, 1, 'No retry storm');
    assert.equal(sandbox.activeTestAnswers.one, 'B');
    assert.equal(toasts.at(-1), 'Temporarily unavailable');
    assert.ok(stored.has('dentora_active_test'));
    // Manual retry does not allow two concurrent submissions or duplicate history.
    const retry = sandbox.submitActiveTest();
    await sandbox.submitActiveTest();
    assert.equal(requestCount, 2);
    sandbox.currentMode = 'study';
    sandbox.currentChatId = 'saved';
    sandbox.currentMessages = [{role:'user', content:'Clear this'}];
    sandbox.chatHistory = [{id:'saved', messages:sandbox.currentMessages}, {id:'keep', messages:[]}];
    assert.equal(sandbox.clearCurrentChat(), true);
    assert.equal(sandbox.chatHistory.length, 1);
    assert.equal(sandbox.chatHistory[0].id, 'keep');
    assert.equal(sandbox.currentChatId, null);
    assert.equal(sandbox.currentMessages.length, 0);
    resolveRequest(new Response(JSON.stringify({total:1, details:[{question_id:'one'}], score_percent:100})));
    await retry;
    assert.equal(attempts.length, 1);
    assert.equal(stored.has('dentora_active_test'), false);
    await sandbox.submitActiveTest();
    assert.equal(requestCount, 2);
    // Reload recovers questions, selections, index and the original deadline.
    sandbox.activeTest.completed = false;
    sandbox.activeTest.expired = false;
    sandbox.activeTestStartedAt = Date.now();
    sandbox.saveActiveTest();
    sandbox.activeTest = null; sandbox.activeTestAnswers = {};
    sandbox.restoreActiveTest();
    assert.equal(sandbox.activeTest.questions[0].id, 'one');
    assert.equal(sandbox.activeTestAnswers.one, 'B');
    assert.equal(sandbox.activeTest.test_token, 'frozen');
    assert.equal(sandbox.testSession.classList.contains('show'), true);
    assert.equal(requestCount, 2);
    // Switching to local questions while an online catalog loads must win.
    let local = false, resolveCatalog, catalogs = 0;
    sandbox.testCatalogLoading = false;
    sandbox.testCatalogReloadRequested = false;
    sandbox.ownerAccessCode = '';
    sandbox.ragPastPaperSyncStatus = null;
    sandbox.dentoraAccessCode = 'fixture';
    sandbox.TEST_CATALOG_API = '/test/catalog';
    sandbox.renderOwnerPaperPanel = () => {};
    sandbox.populateTestFilters = () => {};
    sandbox.DentoraOffline = {ready: Promise.resolve(), isLocal: () => local,
        request: () => ++catalogs === 1 ? new Promise(resolve => resolveCatalog=resolve)
            : Promise.resolve(new Response(JSON.stringify({offline:true, mcq_count:27})))};
    for (const name of ['bankPaperCount','bankQuestionCount','bankEligibleCount','bankRepeatedCount']) sandbox[name] = element();
    vm.runInContext(functionSource('loadTestCatalog'), context);
    const catalogLoad = sandbox.loadTestCatalog(false);
    await Promise.resolve();
    local = true;
    await sandbox.loadTestCatalog(false);
    resolveCatalog(new Response(JSON.stringify({mcq_count:1500})));
    await catalogLoad;
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(catalogs, 2);
    assert.equal(sandbox.testCatalog.offline, true);
    assert.equal(sandbox.bankQuestionCount.textContent, '27');
    console.log('All six mode transitions, late reply isolation, reload recovery, expiry failure, manual retry and duplicate grading passed.');
}
main().catch(error => {console.error(error); process.exitCode = 1;});
