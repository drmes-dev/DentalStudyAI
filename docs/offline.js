(() => {
    'use strict';
    let pack = null;
    let registration = null;
    let shellReady = false;
    let lastLocal = false;
    let useSaved = localStorage.getItem('dentora_use_saved_pack') === 'true';
    const database = new Promise((resolve, reject) => {
        const open = indexedDB.open('dentora-offline', 1);
        open.onupgradeneeded = () => open.result.createObjectStore('packs');
        open.onsuccess = () => resolve(open.result);
        open.onerror = () => reject(open.error);
    });
    async function storage(action, value) {
        const db = await database;
        return new Promise((resolve, reject) => {
            const transaction = db.transaction('packs', action === 'get' ? 'readonly' : 'readwrite');
            const store = transaction.objectStore('packs');
            const request = action === 'get' ? store.get('current') : action === 'put'
                ? store.put(value, 'current') : store.delete('current');
            transaction.oncomplete = () => resolve(request.result);
            transaction.onerror = transaction.onabort = () => reject(transaction.error || new Error('Device storage unavailable.'));
        });
    }
    function valid(value) {
        return value && value.schema_version === 1 && Array.isArray(value.questions)
            && Array.isArray(value.papers) && value.questions.every(q => q && q.id && q.stem
                && q.options && typeof q.options === 'object' && q.correct_answer in q.options);
    }
    const ready = storage('get').then(value => { if (valid(value)) pack = value; }).catch(() => {});
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./sw.js').then(async result => {
            registration = result;
            await navigator.serviceWorker.ready;
            shellReady = true;
            update();
        }).catch(() => update());
    }
    function localMode() { return useSaved || !navigator.onLine; }
    function questions(value = pack) {
        const reliableOnly = document.getElementById('offlineReliableOnly')?.checked !== false;
        return (value?.questions || []).filter(q => !reliableOnly || q.trusted_answer);
    }
    function catalog(value) {
        const all = value.questions;
        const counts = key => Object.entries(all.reduce((result, q) => {
            result[q[key]] = (result[q[key]] || 0) + 1; return result;
        }, {})).sort(([a], [b]) => a.localeCompare(b)).map(([name, count]) => ({name, count}));
        const topics = new Map();
        all.forEach(q => { const key = JSON.stringify([q.subject, q.topic]); topics.set(key, (topics.get(key) || 0) + 1); });
        const repeated = new Set(all.filter(q => q.repeat_count > 1).map(q => q.stem_hash));
        return {configured: true, offline: true, paper_count: value.papers.length,
            question_count: all.length, mcq_count: all.length, test_ready_questions: questions(value).length,
            // Readiness coverage uses the whole saved bank, so tightening the
            // answer-key filter does not falsely improve the readiness score.
            eligible_test_questions: all.length, subjects: counts('subject'), years: counts('year'),
            topics: [...topics].map(([key, count]) => { const [subject, name] = JSON.parse(key); return {subject, name, count}; }),
            papers: value.papers, repeated_question_groups: repeated.size, review_pending_mcqs: 0};
    }
    function shuffle(items) {
        const result = items.slice();
        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }
        return result;
    }
    function sample(value, config) {
        const weak = new Set((config.weak_topics || []).map(v => v.toLowerCase()));
        let pool = shuffle(questions(value).filter(q => (!config.subject || q.subject === config.subject)
            && (!config.year || String(q.year) === String(config.year)) && (!config.topic || q.topic === config.topic)
            && (!config.repeated_only || q.repeat_count > 1)));
        if (weak.size) pool = [...pool.filter(q => weak.has(q.topic.toLowerCase())), ...pool.filter(q => !weak.has(q.topic.toLowerCase()))];
        const seen = new Set();
        const selected = pool.filter(q => { if (seen.has(q.stem_hash)) return false; seen.add(q.stem_hash); return true; })
            .slice(0, Math.max(1, Math.min(100, config.count || 20))).map(q => {
                const {correct_answer, answer_source, trusted_answer, verification_rationale, verification_sources,
                    verification_status, verification_confidence, stem_hash, ...publicQuestion} = q;
                return publicQuestion;
            });
        return {questions: selected, count: selected.length, answers_hidden: true, offline: true};
    }
    function grade(value, responses) {
        const byId = new Map(value.questions.map(q => [q.id, q]));
        const chosen = new Map(responses.map(r => [r.question_id, String(r.answer || '').trim().toUpperCase()]));
        const details = [...chosen].filter(([id]) => byId.has(id)).map(([id, answer]) => {
            const q = byId.get(id);
            return {question_id: id, stem: q.stem, chosen_answer: answer, correct_answer: q.correct_answer,
                answer_source: q.answer_source, answered: Boolean(answer), correct: Boolean(answer && answer === q.correct_answer),
                ...Object.fromEntries(['subject', 'topic', 'subtopic', 'paper_title', 'year', 'repeat_count',
                    'verification_status', 'verification_confidence', 'verification_rationale', 'verification_sources'].map(key => [key, q[key]]))};
        });
        const total = details.length, answered = details.filter(q => q.answered).length, correct = details.filter(q => q.correct).length;
        return {total, answered, correct, unanswered: total - answered, incorrect: answered - correct,
            score_percent: total ? Math.round(correct / total * 1000) / 10 : 0, details, offline: true};
    }
    async function timedFetch(url, options = {}, timeoutMs = 120000) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, {...options, signal: controller.signal});
            // Keep the deadline active through body download, not just headers.
            const body = await response.arrayBuffer();
            return new Response([204, 205, 304].includes(response.status) ? null : body,
                {status: response.status, statusText: response.statusText, headers: response.headers});
        } catch (error) {
            if (controller.signal.aborted || error.name === 'AbortError') {
                throw new Error('The server took too long to respond. Retry when the service is ready.');
            }
            throw new Error('Unable to connect to Dentora. Check your connection and retry.');
        } finally { clearTimeout(timer); }
    }
    async function responseError(response, fallback) {
        const data = await response.json().catch(() => ({}));
        if (typeof data.detail === 'string') return new Error(data.detail);
        return new Error(response.status === 401 ? 'Your access code has expired. Unlock Dentora and retry.'
            : response.status === 429 ? 'Too many requests. Please wait a moment before retrying.' : fallback);
    }
    async function request(kind, url, options = {}, localOnly = false) {
        await ready;
        const body = JSON.parse(options.body || '{}');
        const onlineGrade = kind === 'grade' && !localOnly && (body.test_token || !localMode());
        if (!localOnly && !localMode()) {
            try {
                const response = await timedFetch(url, options);
                if (response.status < 500 && response.status !== 429 || !pack || onlineGrade) {
                    lastLocal = false; update(); return response;
                }
            } catch (error) { if (!pack || onlineGrade) throw error; }
        }
        if (onlineGrade) throw new Error('Reconnect to submit this online test. Your answers are saved; grading uses the original test key.');
        if (!pack) throw new Error('Connect once and save a question pack in Test Mode before using offline tests.');
        if (kind === 'grade' && (body.responses || []).some(r => !pack.questions.some(q => q.id === r.question_id))) {
            throw new Error('Some questions in this test are not in your saved pack. Reconnect to grade this test; your selected answers are still here.');
        }
        const result = kind === 'catalog' ? catalog(pack) : kind === 'start' ? sample(pack, body) : grade(pack, body.responses || []);
        if (kind === 'start' && result.count === 0) {
            throw new Error('No saved questions match these filters and the answer-key rule. Try All subjects; provisional AI keys need review before relying on them.');
        }
        lastLocal = true; update();
        return new Response(JSON.stringify(result), {headers: {'Content-Type': 'application/json'}});
    }
    function update() {
        const status = document.getElementById('offlinePackStatus');
        if (!status) return;
        const saved = document.getElementById('offlineUseSaved');
        saved.checked = useSaved;
        saved.disabled = !pack;
        document.getElementById('offlineRemovePack').disabled = !pack;
        const prefix = localMode() || lastLocal ? 'Using saved questions. ' : '';
        if (!pack) status.textContent = 'Connect once and save questions. Chats already saved on this device are also available offline.';
        else {
            const size = new Blob([JSON.stringify(pack)]).size / (1024 * 1024);
            const trusted = pack.questions.filter(q => q.trusted_answer).length;
            status.textContent = prefix + pack.questions.length + ' questions · ' + size.toFixed(2) + ' MB · '
                + trusted + ' with textbook-reviewed or printed keys · saved ' + new Date(pack.saved_at).toLocaleString()
                + (shellReady ? ' · offline app ready.' : ' · app cache is not ready yet; keep this page open.');
        }
    }
    async function save(url, options) {
        if (!navigator.onLine) throw new Error('Connect to the internet to download or update your question pack.');
        if (!('serviceWorker' in navigator)) throw new Error('This browser cannot save the offline app. Use a current Chrome, Edge, or Firefox browser.');
        const response = await timedFetch(url, options);
        if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.detail || 'The pack could not be downloaded.'); }
        const value = await response.json();
        if (!valid(value) || !value.questions.length) throw new Error('There are no questions with usable answer keys for this subject yet.');
        if (new Blob([JSON.stringify(value)]).size > 8 * 1024 * 1024) throw new Error('This pack exceeds 8 MB. Choose one subject first.');
        if (!registration) registration = await navigator.serviceWorker.register('./sw.js');
        await navigator.serviceWorker.ready;
        shellReady = true;
        await storage('put', value); // Replace atomically; a failed save keeps the old pack.
        pack = value;
        if (navigator.storage?.persist) await navigator.storage.persist().catch(() => false);
        update();
        return value.questions.length;
    }
    async function remove() {
        await storage('delete'); pack = null; useSaved = false; lastLocal = false;
        localStorage.removeItem('dentora_use_saved_pack'); update();
    }
    function setLocal(value) { useSaved = Boolean(value); localStorage.setItem('dentora_use_saved_pack', String(useSaved)); update(); }
    window.DentoraOffline = {ready, request, save, remove, setLocal, update, fetch: timedFetch, responseError,
        isLocal: localMode, hasPack: () => Boolean(pack)};
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    ready.then(update);
})();
