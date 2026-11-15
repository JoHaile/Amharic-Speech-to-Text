/* Amharic STT - front end
   Record with the microphone (live wave), pick a model, send it to the local
   CPU, and browse history. No external libraries: works fully offline. */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---- elements -----------------------------------------------------------
  const tabNew = $("tabNew");
  const tabHistory = $("tabHistory");
  const tabUpload = $("tabUpload");
  const paneNew = $("paneNew");
  const paneHistory = $("paneHistory");
  const paneUpload = $("paneUpload");

  const modelTabs = Array.from(document.querySelectorAll(".model-tab"));
  const modelDots = {};
  document.querySelectorAll("[data-model-dot]").forEach((el) => {
    modelDots[el.dataset.modelDot] = el;
  });

  const recBtn = $("recBtn");
  const recTimer = $("recTimer");
  const recState = $("recState");
  const recHint = $("recHint");
  const preview = $("preview");
  const audioPreview = $("audioPreview");
  const previewModel = $("previewModel");
  const previewLabel = $("previewLabel");
  const clearBtn = $("clearBtn");
  const redoBtn = $("redoBtn");
  const sendBtn = $("sendBtn");
  const sendLabel = sendBtn.querySelector(".btn-label");
  const sendSpinner = sendBtn.querySelector(".spinner");

  const waveWrap = $("waveWrap");
  const waveCanvas = $("waveCanvas");
  const waveCaption = $("waveCaption");

  const drop = $("drop");
  const fileInput = $("fileInput");

  const resultCard = $("resultCard");
  const resultText = $("resultText");
  const resultMeta = $("resultMeta");
  const resultModel = $("resultModel");
  const copyBtn = $("copyBtn");

  const historyList = $("historyList");
  const historyEmpty = $("historyEmpty");
  const historyCount = $("historyCount");
  const clearAllBtn = $("clearAllBtn");
  const historyFilter = $("historyFilter");
  const filterChips = Array.from(historyFilter.querySelectorAll(".filter-chip"));
  const emptyTitle = historyEmpty.querySelector(".empty-title");
  const emptySub = historyEmpty.querySelector(".empty-sub");

  const statusDot = $("statusDot");
  const statusText = $("statusText");

  const toast = $("toast");
  const toastText = $("toastText");

  // ---- state --------------------------------------------------------------
  const DEFAULT_MODEL = "shook";
  let activeModel = DEFAULT_MODEL;
  const modelStates = {};

  let mediaRecorder = null;
  let mediaStream = null;
  let chunks = [];
  let pendingBlob = null;      // recorded or selected audio waiting to be sent
  let pendingName = "recording.wav";
  let pendingSource = "recording";
  let pendingDuration = 0;     // known length in seconds, when we have it
  let previewUrl = null;       // object URL currently loaded in the preview player
  let timerId = null;
  let startedAt = 0;
  let isRecording = false;
  let isSending = false;
  let captureId = 0;           // invalidates a slow capture preparation

  let historyItems = [];
  let historyFilterValue = "all";

  // ---- live microphone wave -----------------------------------------------
  let audioCtx = null;
  let analyser = null;
  let sourceNode = null;
  let waveData = null;
  let waveRaf = null;
  let waveBars = [];
  let waveBarCount = 48;
  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- helpers ------------------------------------------------------------
  function showToast(message, isError) {
    toastText.textContent = message;
    toast.classList.toggle("err", !!isError);
    toast.hidden = false;
    requestAnimationFrame(() => toast.classList.add("show"));
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => { toast.hidden = true; }, 220);
    }, isError ? 6000 : 3000);
  }

  const fmtClock = (sec) => {
    const s = Math.max(0, Math.floor(sec));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };

  const fmtPlayTime = (sec) => {
    if (!Number.isFinite(sec) || sec < 0) return "0:00";
    return fmtClock(sec);
  };

  const fmtSecs = (v) => (v == null ? "—" : `${Number(v).toFixed(2)}s`);

  function fmtWhen(iso) {
    const d = new Date(iso);
    if (isNaN(d)) return iso || "";
    const today = new Date();
    const sameDay = d.toDateString() === today.toDateString();
    const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    if (sameDay) return `Today, ${time}`;
    return `${d.toLocaleDateString([], { month: "short", day: "numeric" })}, ${time}`;
  }

  function modelLabel(key) {
    const state = modelStates[key];
    if (state && state.label) return state.label;
    return key === "wav2vec2" ? "Wav2Vec2 XLSR" : "Shook Tiny";
  }

  function isWav2Vec2(key) {
    return key === "wav2vec2";
  }

  // ---- audio player (shared by the preview and history) -------------------
  const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5.9v12.2L19 12z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="7.4" y="5" width="3.5" height="14" rx="1.3"/><rect x="13.1" y="5" width="3.5" height="14" rx="1.3"/></svg>';

  function createPlayer(host, options) {
    const lazy = !!(options && options.lazy);
    host.classList.add("player");
    host.innerHTML = `
      <button class="player-btn" type="button" aria-label="Play">
        <span class="p-ico p-ico-play">${ICON_PLAY}</span>
        <span class="p-ico p-ico-pause">${ICON_PAUSE}</span>
      </button>
      <div class="player-track" role="slider" tabindex="0" aria-label="Seek"
           aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <span class="player-fill"></span>
        <span class="player-knob"></span>
      </div>
      <span class="player-time">0:00 / 0:00</span>
      <audio preload="${lazy ? "none" : "metadata"}" hidden></audio>`;

    const btn = host.querySelector(".player-btn");
    const track = host.querySelector(".player-track");
    const fill = host.querySelector(".player-fill");
    const knob = host.querySelector(".player-knob");
    const time = host.querySelector(".player-time");
    const audio = host.querySelector("audio");

    let knownDuration = 0;
    let dragging = false;
    let needsLoad = false;
    let pendingRatio = null;

    function ensureLoaded() {
      if (lazy && needsLoad) {
        needsLoad = false;
        audio.load();
      }
    }

    function totalDuration() {
      const d = Number.isFinite(audio.duration) && audio.duration > 0
        ? audio.duration : knownDuration;
      return Number.isFinite(d) && d > 0 ? d : 0;
    }

    function paint() {
      const d = totalDuration();
      const raw = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
      const t = d > 0 ? Math.min(raw, d) : 0;
      const pct = d > 0 ? Math.min(100, Math.max(0, (t / d) * 100)) : 0;
      fill.style.width = pct + "%";
      knob.style.left = pct + "%";
      track.setAttribute("aria-valuenow", String(Math.round(pct)));
      track.setAttribute("aria-valuetext", fmtPlayTime(t) + " of " + fmtPlayTime(d));
      time.textContent = fmtPlayTime(t) + " / " + fmtPlayTime(d);
    }

    audio.addEventListener("loadedmetadata", () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        knownDuration = audio.duration;
      } else if (audio.duration === Infinity) {
        // Some recorder streams report Infinity until they are seeked once.
        const settle = () => {
          audio.removeEventListener("timeupdate", settle);
          try { audio.currentTime = 0; } catch (_) { /* ignore */ }
          if (Number.isFinite(audio.duration)) knownDuration = audio.duration;
          paint();
        };
        audio.addEventListener("timeupdate", settle);
        try { audio.currentTime = 1e7; } catch (_) { /* ignore */ }
      }
      if (pendingRatio != null) {
        const d = totalDuration();
        if (d) { try { audio.currentTime = pendingRatio * d; } catch (_) { /* ignore */ } }
        pendingRatio = null;
      }
      paint();
    });

    audio.addEventListener("durationchange", () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        knownDuration = audio.duration;
      }
      paint();
    });
    audio.addEventListener("timeupdate", paint);

    audio.addEventListener("play", () => {
      host.classList.add("is-playing");
      btn.setAttribute("aria-label", "Pause");
    });
    audio.addEventListener("pause", () => {
      host.classList.remove("is-playing");
      btn.setAttribute("aria-label", "Play");
    });
    audio.addEventListener("ended", () => {
      host.classList.remove("is-playing");
      btn.setAttribute("aria-label", "Play");
      try { audio.currentTime = 0; } catch (_) { /* ignore */ }
      paint();
    });
    audio.addEventListener("error", () => {
      if (!audio.getAttribute("src")) return;
      host.classList.add("is-error");
      host.classList.remove("is-playing");
      btn.disabled = true;
      track.setAttribute("tabindex", "-1");
      time.textContent = "Preview unavailable";
    });

    btn.addEventListener("click", async () => {
      if (!audio.getAttribute("src")) return;
      if (audio.paused) {
        ensureLoaded();
        try {
          await audio.play();
        } catch (_) {
          showToast("This clip cannot be played here, but Send still works.", true);
        }
      } else {
        audio.pause();
      }
    });

    function seekToRatio(ratio) {
      const d = totalDuration();
      if (!d) return;
      const clamped = Math.min(1, Math.max(0, ratio));
      if (audio.readyState === 0) {
        pendingRatio = clamped;
        ensureLoaded();
      } else {
        try { audio.currentTime = clamped * d; } catch (_) { /* ignore */ }
      }
      paint();
    }

    function seekTo(clientX) {
      const rect = track.getBoundingClientRect();
      if (!rect.width || !totalDuration()) return;
      seekToRatio((clientX - rect.left) / rect.width);
    }

    track.addEventListener("pointerdown", (e) => {
      if (btn.disabled) return;
      dragging = true;
      try { track.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
      seekTo(e.clientX);
    });
    track.addEventListener("pointermove", (e) => {
      if (dragging) seekTo(e.clientX);
    });
    const endDrag = (e) => {
      if (!dragging) return;
      dragging = false;
      try { track.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    };
    track.addEventListener("pointerup", endDrag);
    track.addEventListener("pointercancel", endDrag);

    track.addEventListener("keydown", (e) => {
      const d = totalDuration();
      if (!d) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const current = audio.readyState === 0 ? 0 : audio.currentTime;
        const next = current + (e.key === "ArrowRight" ? 5 : -5);
        seekToRatio(next / d);
      } else if (e.key === "Home" || e.key === "End") {
        e.preventDefault();
        seekToRatio(e.key === "Home" ? 0 : 1);
      } else if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        btn.click();
      }
    });

    function setSource(url, durationSeconds) {
      host.classList.remove("is-error", "is-playing");
      btn.disabled = false;
      btn.setAttribute("aria-label", "Play");
      track.setAttribute("tabindex", "0");
      knownDuration = Number.isFinite(durationSeconds) && durationSeconds > 0
        ? durationSeconds : 0;
      pendingRatio = null;
      audio.src = url;
      if (lazy) {
        needsLoad = true;
      } else {
        audio.load();
      }
      paint();
    }

    function reset() {
      host.classList.remove("is-error", "is-playing");
      btn.disabled = false;
      btn.setAttribute("aria-label", "Play");
      track.setAttribute("tabindex", "0");
      pendingRatio = null;
      needsLoad = false;
      try { audio.pause(); } catch (_) { /* ignore */ }
      audio.removeAttribute("src");
      audio.load();
      knownDuration = 0;
      paint();
    }

    paint();
    return { audio, setSource, reset, paint };
  }

  // ---- turn a capture into a playable WAV ---------------------------------
  function encodeWav(buffer) {
    const channels = Math.max(1, buffer.numberOfChannels);
    const frames = buffer.length;
    const rate = buffer.sampleRate || 16000;
    const dataBytes = frames * 2;
    const out = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(out);
    const write = (offset, text) => {
      for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
    };

    write(0, "RIFF");
    view.setUint32(4, 36 + dataBytes, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);        // PCM chunk size
    view.setUint16(20, 1, true);         // PCM
    view.setUint16(22, 1, true);         // mono
    view.setUint32(24, rate, true);
    view.setUint32(28, rate * 2, true);  // byte rate
    view.setUint16(32, 2, true);        // block align
    view.setUint16(34, 16, true);        // bits per sample
    write(36, "data");
    view.setUint32(40, dataBytes, true);

    const source = [];
    for (let c = 0; c < channels; c++) source.push(buffer.getChannelData(c));
    let offset = 44;
    for (let i = 0; i < frames; i++) {
      let sample = 0;
      for (let c = 0; c < channels; c++) sample += source[c][i];
      sample = Math.max(-1, Math.min(1, sample / channels));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
    return new Blob([out], { type: "audio/wav" });
  }

  async function prepareCapture(blob) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    let ctx = null;
    try {
      ctx = new Ctx({ sampleRate: 16000 });
      const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
      const wav = encodeWav(decoded);
      return wav.size > 44 ? { blob: wav, duration: decoded.duration } : null;
    } catch (_) {
      return null;
    } finally {
      if (ctx) { try { ctx.close(); } catch (_) { /* ignore */ } }
    }
  }

  // ---- model status -------------------------------------------------------
  function updateStatusUI() {
    const state = modelStates[activeModel];
    if (!state) {
      statusDot.className = "dot";
      statusText.textContent = "Checking models…";
      return;
    }
    if (state.error) {
      statusDot.className = "dot err";
      statusText.textContent = `${state.label} unavailable`;
      statusText.title = state.error;
    } else if (state.loading) {
      statusDot.className = "dot load";
      statusText.textContent = `Loading ${state.label}…`;
      statusText.title = state.name || "";
    } else if (state.ready) {
      statusDot.className = "dot ok";
      statusText.textContent = `${state.label} ready${state.load_seconds ? ` · ${state.load_seconds.toFixed(2)}s` : ""}`;
      statusText.title = `${state.name || ""}\n${state.model_dir || ""}`;
    } else {
      statusDot.className = "dot";
      statusText.textContent = `${state.label} · loads on first use`;
      statusText.title = state.name || "";
    }
    updateModelDots();
  }

  function updateModelDots() {
    Object.entries(modelDots).forEach(([key, dot]) => {
      const state = modelStates[key];
      if (!state) {
        dot.className = "model-dot";
        dot.title = "Checking model";
      } else if (state.error) {
        dot.className = "model-dot err";
        dot.title = state.error;
      } else if (state.loading) {
        dot.className = "model-dot load";
        dot.title = "Loading model…";
      } else if (state.ready) {
        dot.className = "model-dot ok";
        dot.title = `Ready${state.load_seconds ? ` in ${state.load_seconds.toFixed(2)}s` : ""}`;
      } else {
        dot.className = "model-dot";
        dot.title = "Loads when selected";
      }
    });
  }

  function updateModelTabs() {
    modelTabs.forEach((btn) => {
      const on = btn.dataset.model === activeModel;
      btn.classList.toggle("is-active", on);
      btn.setAttribute("aria-selected", String(on));
    });
  }

  function ensureModelLoaded(key) {
    const state = modelStates[key];
    if (!state) {
      return Promise.reject(new Error("Model status is not available yet."));
    }
    if (state.ready) return Promise.resolve(state);
    if (state._promise) return state._promise;

    state.loading = true;
    state.error = null;
    updateStatusUI();

    state._promise = fetch(`/api/models/${encodeURIComponent(key)}/load`, {
      method: "POST",
    })
      .then(async (r) => {
        let data = null;
        try { data = await r.json(); } catch (_) { /* non-JSON */ }
        if (!r.ok) throw new Error((data && data.error) || `Model load failed (${r.status})`);
        if (!data) throw new Error("The server returned an unreadable reply.");
        Object.assign(state, data);
        state.loading = false;
        state.error = null;
        updateStatusUI();
        return state;
      })
      .catch((err) => {
        state.loading = false;
        state.error = err.message || String(err);
        updateStatusUI();
        throw err;
      })
      .finally(() => {
        state._promise = null;
      });

    return state._promise;
  }

  async function loadStatus() {
    try {
      const r = await fetch("/api/status");
      const d = await r.json();
      const list = Array.isArray(d.models) ? d.models : [];
      list.forEach((m) => {
        const state = modelStates[m.id] || (modelStates[m.id] = {});
        Object.assign(state, m, { loading: false });
      });
      activeModel = d.default_model && modelStates[d.default_model]
        ? d.default_model : DEFAULT_MODEL;
      updateModelTabs();
      updateStatusUI();
      // The default model is preloaded by the server; if it is not, fetch it
      // now so the first Send is not delayed by a cold load.
      if (modelStates[activeModel] && !modelStates[activeModel].ready) {
        ensureModelLoaded(activeModel).catch(() => {});
      }
    } catch (e) {
      statusDot.className = "dot err";
      statusText.textContent = "Server not reachable";
    }
  }

  // ---- tabs ---------------------------------------------------------------
  function showTab(which) {
    const tabs = { new: tabNew, history: tabHistory, upload: tabUpload };
    const panes = { new: paneNew, history: paneHistory, upload: paneUpload };
    Object.keys(tabs).forEach((key) => {
      const active = key === which;
      if (tabs[key]) {
        tabs[key].classList.toggle("is-active", active);
        tabs[key].setAttribute("aria-selected", String(active));
      }
      if (panes[key]) panes[key].classList.toggle("is-active", active);
    });
    if (which === "history") loadHistory();
  }

  tabNew.addEventListener("click", () => showTab("new"));
  tabHistory.addEventListener("click", () => showTab("history"));
  if (tabUpload) tabUpload.addEventListener("click", () => showTab("upload"));

  // ---- model selection ----------------------------------------------------
  modelTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (isSending) return;
      const key = btn.dataset.model;
      if (!modelStates[key]) return;
      activeModel = key;
      updateModelTabs();
      updateStatusUI();
      // Switching models never touches the captured clip or the result card;
      // it only changes which model the next Send will use.
      if (!isRecording && !isSending) {
        if (pendingBlob) {
          updatePendingCopy();
        } else if (!resultCard.hidden) {
          recHint.textContent = `Next recording will use ${modelLabel(key)}.`;
        } else {
          recHint.textContent = `Will transcribe with ${modelLabel(key)}. Press the microphone and speak Amharic.`;
        }
      }
      ensureModelLoaded(key).catch((err) => showToast(err.message || "Could not load the model.", true));
    });
  });

  // ---- live wave drawing --------------------------------------------------
  function sizeCanvas() {
    const rect = waveCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (waveCanvas.width !== w || waveCanvas.height !== h) {
      waveCanvas.width = w;
      waveCanvas.height = h;
    }
    return { w: rect.width, h: rect.height, dpr };
  }

  function drawIdleWave() {
    const { w, h, dpr } = sizeCanvas();
    if (!w || !h) return;
    const ctx = waveCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(139, 144, 168, .35)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(w * 0.12, h / 2);
    ctx.lineTo(w * 0.88, h / 2);
    ctx.stroke();
    waveBars = new Array(waveBarCount).fill(0);
  }

  function drawWaveFrame() {
    const { w, h, dpr } = sizeCanvas();
    if (!w || !h || !analyser) return;

    const ctx = waveCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    analyser.getByteTimeDomainData(waveData);
    const n = Math.max(24, Math.min(64, Math.floor(w / 9)));
    if (waveBars.length !== n) waveBars = new Array(n).fill(0);
    const step = Math.max(1, Math.floor(waveData.length / n));
    const mid = h / 2;
    const maxBar = h * 0.42;
    const barW = Math.max(2, (w / n) * 0.52);
    const gap = (w / n) - barW;

    const gradient = ctx.createLinearGradient(0, mid - maxBar, 0, mid + maxBar);
    gradient.addColorStop(0, "#ff6b82");
    gradient.addColorStop(0.5, "#e63b58");
    gradient.addColorStop(1, "#ff6b82");

    // A faint centre line keeps the visualizer readable while quiet.
    ctx.strokeStyle = "rgba(214, 47, 74, .16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.06, mid);
    ctx.lineTo(w * 0.94, mid);
    ctx.stroke();

    ctx.fillStyle = gradient;
    for (let i = 0; i < n; i++) {
      let peak = 0;
      const start = i * step;
      const end = Math.min(waveData.length, start + step);
      for (let j = start; j < end; j++) {
        const v = Math.abs(waveData[j] - 128) / 128;
        if (v > peak) peak = v;
      }
      // A little gain makes quiet speech visible without making silence noisy.
      const target = Math.min(1, Math.pow(peak * 1.7, 0.72));
      const decay = reduceMotion ? 1 : 0.86;
      waveBars[i] = Math.max(target, waveBars[i] * decay);
      const barH = Math.max(2, waveBars[i] * maxBar);
      const x = i * (barW + gap) + (w - n * (barW + gap)) / 2;
      const y = mid - barH;
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH * 2, barW / 2);
        ctx.fill();
      } else {
        ctx.fillRect(x, y, barW, barH * 2);
      }
    }

    waveRaf = requestAnimationFrame(drawWaveFrame);
  }

  async function startWave(stream) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    audioCtx = new AudioCtx();
    if (audioCtx.state === "suspended") await audioCtx.resume();
    sourceNode = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.72;
    sourceNode.connect(analyser);
    waveData = new Uint8Array(analyser.fftSize);
    waveBars = new Array(waveBarCount).fill(0);
    waveWrap.classList.add("is-recording");
    waveCaption.textContent = "Listening…";
    drawWaveFrame();
  }

  function stopWave() {
    if (waveRaf) {
      cancelAnimationFrame(waveRaf);
      waveRaf = null;
    }
    if (sourceNode) {
      try { sourceNode.disconnect(); } catch (_) { /* ignore */ }
      sourceNode = null;
    }
    analyser = null;
    waveData = null;
    if (audioCtx) {
      const ctx = audioCtx;
      audioCtx = null;
      try { ctx.close(); } catch (_) { /* ignore */ }
    }
    waveWrap.classList.remove("is-recording");
    waveCaption.textContent = "Your voice will appear here";
    drawIdleWave();
  }

  window.addEventListener("resize", () => {
    if (!isRecording) drawIdleWave();
  });

  // ---- preview / sending state -------------------------------------------
  let previewPlayer = null;

  // Keeps the pending clip and its labels in sync with the selected model.
  function updatePendingCopy() {
    if (!pendingBlob) return;
    const hasResult = !resultCard.hidden;
    const kind = pendingSource === "upload" ? "File ready" : "Recording captured";
    const length = pendingDuration > 0 ? ` (${fmtPlayTime(pendingDuration)})` : "";
    recState.textContent = hasResult
      ? `Clip kept. Press Send to transcribe with ${modelLabel(activeModel)}.`
      : `${kind}${length}. Press Send to transcribe with ${modelLabel(activeModel)}.`;
    recHint.textContent = hasResult
      ? "Switch model and press Send to compare, or record another."
      : "Listen back, re-record if needed, then press Send.";
    previewLabel.textContent = hasResult ? "Clip kept" : "Ready to send";
    previewModel.textContent = modelLabel(activeModel);
    previewModel.classList.toggle("is-wav2vec2", isWav2Vec2(activeModel));
    previewModel.hidden = false;
  }

  function setSending(on) {
    isSending = on;
    sendBtn.disabled = on || !pendingBlob;
    sendSpinner.hidden = !on;
    sendLabel.textContent = on ? "Transcribing…" : "Send";
    recBtn.disabled = on;
    fileInput.disabled = on;
    redoBtn.disabled = on;
    clearBtn.disabled = on;
    modelTabs.forEach((btn) => { btn.disabled = on; });
  }

  function clearPending() {
    captureId++;                 // drop any capture that is still being prepared
    pendingBlob = null;
    pendingSource = "recording";
    pendingDuration = 0;
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
    if (previewPlayer) previewPlayer.reset();
    previewLabel.textContent = "Ready to send";
    preview.hidden = true;
    previewModel.hidden = true;
    sendBtn.disabled = true;
  }

  function setPending(blob, name, source, duration) {
    pendingBlob = blob;
    pendingName = name;
    pendingSource = source;
    pendingDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob);
    if (!previewPlayer) previewPlayer = createPlayer(audioPreview);
    previewPlayer.setSource(previewUrl, pendingDuration);
    preview.hidden = false;
    sendBtn.disabled = isSending || !blob;
    updatePendingCopy();
  }

  // Drops the clip (and its result) and returns the panel to its start state.
  function resetToReady() {
    clearPending();
    resultCard.hidden = true;
    resultModel.hidden = true;
    recState.textContent = "Ready to record";
    recHint.textContent = "Press the microphone and start speaking in Amharic.";
    recTimer.textContent = "0:00";
  }

  // ---- recording ----------------------------------------------------------
  function supportedMime() {
    if (!window.MediaRecorder) return "";
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const m of candidates) {
      if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
    }
    return "";
  }

  function extForMime(mime, fallback) {
    if (!mime) return fallback;
    if (mime.includes("webm")) return "webm";
    if (mime.includes("ogg")) return "ogg";
    if (mime.includes("mp4")) return "m4a";
    if (mime.includes("wav")) return "wav";
    return fallback;
  }

  function tick() {
    recTimer.textContent = fmtClock((Date.now() - startedAt) / 1000);
  }

  async function startRecording() {
    if (isRecording || isSending) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("This browser cannot record audio. Use the file upload below instead.", true);
      return;
    }
    if (!window.MediaRecorder) {
      showToast("Recording is not supported in this browser. Upload a file instead.", true);
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (err) {
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      showToast(
        denied
          ? "Microphone access was blocked. Allow the microphone for this page, or upload a file."
          : `Could not start the microphone: ${err.message || err}`,
        true
      );
      return;
    }

    clearPending();
    resultCard.hidden = true;
    resultModel.hidden = true;

    const mime = supportedMime();
    try {
      mediaRecorder = mime ? new MediaRecorder(mediaStream, { mimeType: mime })
                           : new MediaRecorder(mediaStream);
    } catch (err) {
      showToast(`Could not start recording: ${err.message || err}`, true);
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
      return;
    }

    chunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const type = mediaRecorder.mimeType || mime || "audio/webm";
      const original = new Blob(chunks, { type });
      if (original.size === 0) {
        showToast("Nothing was recorded. Try again and speak a little louder.", true);
        recState.textContent = "Ready to record";
        recHint.textContent = "Press the microphone and start speaking in Amharic.";
        return;
      }

      recState.textContent = "Preparing the recording…";
      recHint.textContent = "One moment — making a clip you can replay before sending.";
      const id = ++captureId;

      // Browser recordings are Opus in WebM/OGG, which some players refuse.
      // A decoded WAV copy plays everywhere, seeks correctly and carries a real
      // duration, so the preview always matches the clip that was just recorded.
      const prepared = await prepareCapture(original);
      if (id !== captureId) return;   // a newer capture or a reset won the race

      if (prepared) {
        setPending(prepared.blob, "recording.wav", "recording", prepared.duration);
      } else {
        setPending(original, `recording.${extForMime(type, "webm")}`, "recording", 0);
      }
    };

    try {
      mediaRecorder.start();
    } catch (err) {
      // A stream can refuse to record (device unplugged, track ended, codec
      // unavailable). Release the microphone instead of leaving it open.
      showToast(`Could not start recording: ${err.message || err}`, true);
      stopWave();
      if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
        mediaStream = null;
      }
      mediaRecorder = null;
      recState.textContent = "Ready to record";
      recHint.textContent = "Press the microphone and start speaking in Amharic.";
      return;
    }

    isRecording = true;
    recBtn.classList.add("is-recording");
    recBtn.setAttribute("aria-label", "Stop recording");
    recState.textContent = "Recording… speak now";
    recHint.textContent = "Press the button again when you are done.";
    startedAt = Date.now();
    recTimer.textContent = "0:00";
    timerId = setInterval(tick, 200);
    try {
      await startWave(mediaStream);
    } catch (_) {
      // Recording still works without the optional visualization.
    }
  }

  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    clearInterval(timerId);
    timerId = null;
    recBtn.classList.remove("is-recording");
    recBtn.setAttribute("aria-label", "Start recording");
    recState.textContent = "Finishing…";
    stopWave();
    try {
      if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    } catch (_) { /* ignore */ }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
  }

  recBtn.addEventListener("click", () => {
    if (isRecording) stopRecording();
    else startRecording();
  });

  redoBtn.addEventListener("click", () => {
    if (isSending) return;
    // startRecording clears the old clip only once the mic is actually ready,
    // so a denied permission prompt does not throw the clip away.
    startRecording();
  });

  clearBtn.addEventListener("click", () => {
    if (isSending) return;
    resetToReady();
  });

  // ---- file upload --------------------------------------------------------
  function takeFile(file) {
    if (!file) return;
    if (isSending) return;
    clearPending();
    resultCard.hidden = true;
    resultModel.hidden = true;
    setPending(file, file.name, "upload", 0);
    showTab("new");
  }

  fileInput.addEventListener("change", (e) => takeFile(e.target.files && e.target.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("is-over");
    })
  );
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) takeFile(f);
  });

  // ---- result -------------------------------------------------------------
  function metricsChips(d) {
    return [
      `<span class="chip">Audio <b>${fmtSecs(d.duration_s)}</b></span>`,
      `<span class="chip">Took <b>${fmtSecs(d.processing_s)}</b></span>`,
      `<span class="chip">RTF <b>${Number(d.rtf).toFixed(2)}</b></span>`,
      `<span class="chip">RAM <b>${Math.round(d.peak_rss_mb)} MB</b></span>`,
    ].join("");
  }

  async function send() {
    if (!pendingBlob || isSending) return;

    const blob = pendingBlob;
    const name = pendingName;
    const source = pendingSource;
    const modelKey = activeModel;
    const state = modelStates[modelKey];

    setSending(true);
    try {
      if (state && !state.ready) {
        recState.textContent = `Loading ${modelLabel(modelKey)}…`;
        await ensureModelLoaded(modelKey);
      }

      recState.textContent = `Transcribing with ${modelLabel(modelKey)} on the CPU…`;

      const fd = new FormData();
      fd.append("audio", blob, name);
      fd.append("source", source);
      fd.append("model", modelKey);

      const r = await fetch("/api/transcribe", { method: "POST", body: fd });
      let data = null;
      try { data = await r.json(); } catch (_) { /* non-JSON */ }

      if (!r.ok) {
        throw new Error((data && data.error) || `Server error (${r.status})`);
      }
      if (!data) throw new Error("The server returned an unreadable reply.");

      const text = (data.text || "").trim();
      const doneKey = data.model || modelKey;
      const doneLabel = data.model_label || modelLabel(doneKey);
      resultText.value = text || "(no speech detected)";
      resultText.style.height = "auto";
      resultText.style.height = Math.max(96, resultText.scrollHeight) + "px";
      resultMeta.innerHTML = metricsChips(data);
      resultModel.textContent = doneLabel;
      resultModel.classList.toggle("is-wav2vec2", isWav2Vec2(doneKey));
      resultModel.hidden = false;
      resultCard.hidden = false;

      // The clip is deliberately kept: the user can switch model tabs and
      // press Send again to compare the same recording with the other model.
      previewLabel.textContent = "Clip kept";
      previewModel.textContent = doneLabel;
      previewModel.classList.toggle("is-wav2vec2", isWav2Vec2(doneKey));
      recState.textContent = `Done with ${doneLabel}. Clip kept — switch model and press Send to compare.`;
      recHint.textContent = "The same recording can be sent again with the other model, or record another.";

      // Auto-copy the generated text so it's ready to paste.
      const copyText = (resultText.value || "").trim();
      if (copyText) {
        try {
          await navigator.clipboard.writeText(copyText);
          showToast("Text copied to clipboard.");
        } catch (_) {
          try { resultText.focus(); resultText.select(); document.execCommand("copy"); showToast("Text copied to clipboard."); }
          catch (_) { /* clipboard unavailable */ }
        }
      }

      loadHistory();
      resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      showToast(err.message || "Transcription failed.", true);
      recState.textContent = "Transcription failed. You can try Send again.";
    } finally {
      setSending(false);
    }
  }

  sendBtn.addEventListener("click", send);

  copyBtn.addEventListener("click", async () => {
    const text = (resultText.value || "").trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard.");
    } catch (_) {
      try { resultText.focus(); resultText.select(); } catch (_) { /* ignore */ }
      showToast("Selected — press Ctrl+C to copy.");
    }
  });

  resultText.addEventListener("input", () => {
    resultText.style.height = "auto";
    resultText.style.height = Math.max(96, resultText.scrollHeight) + "px";
  });

  // ---- history ------------------------------------------------------------
  const TRASH = '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7m-8 0 .8 12.1A1.5 1.5 0 0 0 9.3 20.5h5.4a1.5 1.5 0 0 0 1.5-1.4L17 7"/></svg>';

  function makeModelBadge(key) {
    const badge = document.createElement("span");
    badge.className = "model-badge" + (isWav2Vec2(key) ? " is-wav2vec2" : "");
    badge.textContent = modelLabel(key);
    return badge;
  }

  function renderHistory() {
    historyList.innerHTML = "";

    const items = historyItems.filter((it) => {
      const key = it.model || DEFAULT_MODEL;
      return historyFilterValue === "all" || key === historyFilterValue;
    });

    if (!items.length) {
      historyEmpty.hidden = false;
      clearAllBtn.hidden = !historyItems.length;
      historyCount.hidden = !historyItems.length;
      historyCount.textContent = String(historyItems.length);
      if (historyItems.length) {
        emptyTitle.textContent = "No transcriptions for this model";
        emptySub.textContent = "Switch the filter to All, or record something with this model.";
      } else {
        emptyTitle.textContent = "No transcriptions yet";
        emptySub.textContent = "Record something in the New tab and it will appear here, along with the audio.";
      }
      return;
    }

    historyEmpty.hidden = true;
    clearAllBtn.hidden = false;
    historyCount.hidden = false;
    historyCount.textContent = String(historyItems.length);

    const frag = document.createDocumentFragment();

    items.forEach((it) => {
      const key = it.model || DEFAULT_MODEL;
      const el = document.createElement("article");
      el.className = "hist-item";
      el.dataset.id = it.id;
      el.dataset.model = key;

      const top = document.createElement("div");
      top.className = "hist-top";
      const when = document.createElement("span");
      when.className = "hist-when";
      when.textContent = fmtWhen(it.created_at);
      const meta = document.createElement("div");
      meta.className = "hist-meta";
      meta.append(makeModelBadge(key));
      const dur = document.createElement("span");
      dur.className = "chip";
      dur.textContent = `${fmtSecs(it.duration_s)} · RTF ${Number(it.rtf || 0).toFixed(2)}`;
      meta.append(dur);
      top.append(when, meta);

      const text = document.createElement("p");
      const body = (it.text || "").trim();
      text.className = "hist-text" + (body ? "" : " is-empty");
      text.textContent = body || "(no speech detected)";

      const bottom = document.createElement("div");
      bottom.className = "hist-bottom";

      if (it.audio_file) {
        const host = document.createElement("div");
        host.className = "player player-compact";
        bottom.append(host);
        createPlayer(host, { lazy: true }).setSource(
          `/api/history/${encodeURIComponent(it.id)}/audio`,
          it.duration_s
        );
      }

      const actions = document.createElement("div");
      actions.className = "hist-actions";

      const del = document.createElement("button");
      del.className = "icon-btn danger";
      del.type = "button";
      del.title = "Delete this entry";
      del.setAttribute("aria-label", "Delete this entry");
      del.innerHTML = TRASH;
      del.addEventListener("click", async () => {
        if (!confirm("Delete this transcription and its audio?")) return;
        try {
          const r = await fetch(`/api/history/${encodeURIComponent(it.id)}`, { method: "DELETE" });
          if (!r.ok) throw new Error("Delete failed.");
          el.remove();
          loadHistory();
          showToast("Deleted.");
        } catch (err) {
          showToast(err.message || "Could not delete.", true);
        }
      });

      actions.append(del);
      bottom.append(actions);

      el.append(top, text, bottom);
      frag.append(el);
    });

    historyList.append(frag);
  }

  async function loadHistory() {
    try {
      const r = await fetch("/api/history");
      const items = await r.json();
      historyItems = Array.isArray(items) ? items : [];
      renderHistory();
    } catch (_) {
      /* leave whatever is on screen */
    }
  }

  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      historyFilterValue = chip.dataset.filter;
      filterChips.forEach((c) => {
        const on = c === chip;
        c.classList.toggle("is-active", on);
        c.setAttribute("aria-selected", String(on));
      });
      renderHistory();
    });
  });

  clearAllBtn.addEventListener("click", async () => {
    if (!confirm("Delete ALL transcriptions and their audio? This cannot be undone.")) return;
    try {
      const r = await fetch("/api/history", { method: "DELETE" });
      if (!r.ok) throw new Error("Could not clear history.");
      loadHistory();
      showToast("History cleared.");
    } catch (err) {
      showToast(err.message || "Could not clear history.", true);
    }
  });

  // ---- boot ---------------------------------------------------------------
  previewPlayer = createPlayer(audioPreview);
  drawIdleWave();
  loadStatus();
  loadHistory();
})();
