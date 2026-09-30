const CACHE = 'dentora-shell-v3';
const ROOT = new URL('./', self.location.href);
const SHELL = ['./', './index.html', './offline.js?v=2', './manifest.webmanifest', './icon.svg',
    './about.html', './privacy.html', './terms.html'].map(path => new URL(path, ROOT).href);
self.addEventListener('install', event => {
    event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL.map(url => new Request(url, {cache: 'reload'}))))
        .then(() => self.skipWaiting()));
});
self.addEventListener('activate', event => {
    event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key =>
        key.startsWith('dentora-shell-') && key !== CACHE).map(key => caches.delete(key))))
        .then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    // Never cache API requests, credentials, uploaded PDFs, or arbitrary links.
    if (event.request.method !== 'GET' || url.origin !== ROOT.origin || !SHELL.includes(url.href)) return;
    event.respondWith((async () => {
        const cache = await caches.open(CACHE);
        try {
            const response = await fetch(event.request, {cache: 'no-store'});
            if (response.ok) await cache.put(event.request, response.clone());
            return response;
        } catch (_) {
            return await cache.match(event.request) || await cache.match(new URL('./index.html', ROOT).href);
        }
    })());
});
