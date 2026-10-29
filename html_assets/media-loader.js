/* Native, range-capable video loading. No full-file fetch or Blob conversion. */
(() => {
    'use strict';
    const images = [...document.querySelectorAll('img[data-src]')];
    function loadImage(image) {
        image.src = image.dataset.src;
        image.removeAttribute('data-src');
    }
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver(entries => {
            for (const { target, isIntersecting } of entries) {
                if (isIntersecting) { loadImage(target); imageObserver.unobserve(target); }
            }
        }, { rootMargin: '200px 0px' });
        images.forEach(image => imageObserver.observe(image));
    } else images.forEach(loadImage);
    const videos = [...document.querySelectorAll('video[data-src]')];
    const hero = document.getElementById('heroVideo');
    const states = new Map(videos.map(video => [video, {
        near: false, visible: false, loading: false, loaded: false, failed: false,
        release: null
    }]));
    const MAX_LOADS = 2;
    let active = 0;
    let heroReady = !hero || hero.readyState >= 2;

    function play(video) {
        if (!document.hidden && video.paused) {
            const attempt = video.play();
            if (attempt) attempt.catch(() => { /* Retry on re-entry or user interaction. */ });
        }
    }

    function showPoster(video) {
        if (video.dataset.poster && !video.poster) video.poster = video.dataset.poster;
    }

    function start(video, state) {
        state.loading = true;
        active++;
        let timer;
        const loadEvents = ['progress', 'suspend', 'loadeddata', 'canplaythrough', 'error'];
        function release() {
            if (!state.loading) return;
            state.loading = false;
            active--;
            clearTimeout(timer);
            for (const event of loadEvents) video.removeEventListener(event, settled);
            state.release = null;
            pump();
        }
        function settled(event) {
            const complete = Number.isFinite(video.duration) && video.buffered.length &&
                video.buffered.end(video.buffered.length - 1) >= video.duration - 0.1;
            // Hold the slot through initial buffering, while playback can already begin.
            if (event.type === 'error') {
                state.failed = true;
                state.loaded = false;
            }
            if (complete || event.type === 'error' || (video.networkState === video.NETWORK_IDLE && video.readyState >= 2)) release();
        }
        state.release = release;
        for (const event of loadEvents) video.addEventListener(event, settled);
        timer = setTimeout(() => {
            // A stalled or unreachable file must not block every subsequent card.
            // Cancel its request before freeing the slot; retry on the next re-entry.
            state.failed = true;
            state.loaded = false;
            video.pause();
            video.removeAttribute('src');
            video.load();
            release();
        }, 30000);
        showPoster(video);
        video.preload = 'auto';
        video.src = video.dataset.src;
        state.loaded = true;
        video.load();
        if (state.visible) play(video);
    }

    function pump() {
        if (document.hidden || !heroReady) return;
        const queue = videos.filter(video => {
            const state = states.get(video);
            return state.near && !state.loaded && !state.failed;
        }).sort((a, b) => {
            const av = states.get(a).visible, bv = states.get(b).visible;
            return Number(bv) - Number(av) ||
                Math.abs(a.getBoundingClientRect().top) - Math.abs(b.getBoundingClientRect().top);
        });
        while (active < MAX_LOADS && queue.length) {
            const video = queue.shift();
            start(video, states.get(video));
        }
    }

    if (hero && !heroReady) {
        const ready = () => { heroReady = true; pump(); };
        hero.addEventListener('loadeddata', ready, { once: true });
        hero.addEventListener('error', ready, { once: true });
        setTimeout(ready, 8000);
    }

    if ('IntersectionObserver' in window) {
        const nearby = new IntersectionObserver(entries => {
            for (const { target: video, isIntersecting } of entries) {
                const state = states.get(video);
                state.near = isIntersecting;
                if (isIntersecting) {
                    state.failed = false;
                    showPoster(video);
                } else if (state.loading) {
                    // Stop unfinished downloads if a fast scroll takes the card away.
                    video.pause();
                    video.removeAttribute('src');
                    video.load();
                    state.loaded = false;
                    state.release();
                }
            }
            pump();
        }, { rootMargin: '200px 0px', threshold: 0 });
        const visible = new IntersectionObserver(entries => {
            for (const { target: video, isIntersecting } of entries) {
                if (video === hero) {
                    if (isIntersecting) play(video); else video.pause();
                    continue;
                }
                const state = states.get(video);
                state.visible = isIntersecting;
                if (isIntersecting && state.loaded) play(video); else video.pause();
            }
            pump();
        }, { threshold: 0.01 });
        for (const video of videos) { nearby.observe(video); visible.observe(video); }
        if (hero) visible.observe(hero);
    } else {
        // Legacy browsers retain the autoplay/loop experience.
        for (const video of videos) {
            Object.assign(states.get(video), { near: true, visible: true });
        }
        pump();
    }

    function resume() {
        for (const video of videos) {
            const state = states.get(video);
            if (state.visible && state.loaded) play(video);
        }
        if (hero) {
            const rect = hero.getBoundingClientRect();
            if (rect.bottom > 0 && rect.top < innerHeight) play(hero);
        }
        pump();
    }
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            for (const video of [...videos, ...(hero ? [hero] : [])]) video.pause();
        } else resume();
    });
    document.addEventListener('pointerdown', resume, { passive: true });
    document.addEventListener('keydown', resume);
    window.addEventListener('pageshow', resume);
})();
