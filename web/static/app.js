/* Amharic STT - front end
   Record with the microphone, send to the local model, browse history.
   No external libraries: works fully offline. */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---- elements -----------------------------------------------------------
  const tabNew = $("tabNew");
  const tabHistory = $("tabHistory");
  const paneNew = $("paneNew");
  const paneHistory = $("paneHistory");

  const recBtn = $("recBtn");
  const recTimer = $("recTimer");
  const recState = $("recState");
  const recHint = $("recHint");
  const preview = $("preview");
  const audioPreview = $("audioPreview");
  const redoBtn = $("redoBtn");
  const sendBtn = $("sendBtn");
  const sendLabel = sendBtn.querySelector(".btn-label");
  const sendSpinner = sendBtn.querySelector(".spinner");

  const drop = $("drop");
  const fileInput = $("fileInput");

  const resultCard = $("resultCard");
  const resultText = $("resultText");
  const resultMeta = $("resultMeta");
  const copyBtn = $("copyBtn");

  const historyList = $("historyList");
  const historyEmpty = $("historyEmpty");
  const historyCount = $("historyCount");
  const clearAllBtn = $("clearAllBtn");

  const statusDot = $("statusDot");
  const statusText = $("statusText");

  const toast = $("toast");
  const toastText = $("toastText");

  // ---- state --------------------------------------------------------------
  let mediaRecorder = null;
  let mediaStream = null;
  let chunks = [];
  let pendingBlob = null;      // recorded or selected audio waiting to be sent
  let pendingName = "recording.webm";
  let pendingSource = "recording";
  let timerId = null;
  let startedAt = 0;
  let isRecording = false;
  let isSending = false;

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

  function setSending(on) {
    isSending = on;
    sendBtn.disabled = on || !pendingBlob;
    sendSpinner.hidden = !on;
    sendLabel.textContent = on ? "Transcribing…" : "Send";
    recBtn.disabled = on;
    fileInput.disabled = on;
  }

  function clearPending() {
    pendingBlob = null;
    pendingSource = "recording";
    if (audioPreview.src) {
      URL.revokeObjectURL(audioPreview.src);
      audioPreview.removeAttribute("src");
      audioPreview.load();
    }
    preview.hidden = true;
    sendBtn.disabled = true;
  }

  function setPending(blob, name, source, hint) {
    pendingBlob = blob;
    pendingName = name;
    pendingSource = source;
    audioPreview.src = URL.createObjectURL(blob);
    preview.hidden = false;
    sendBtn.disabled = isSending || !blob;
    if (hint) recState.textContent = hint;
  }

  // ---- status -------------------------------------------------------------
  async function loadStatus() {
    try {
      const r = await fetch("/api/status");
      const d = await r.json();
      if (d.error) {
        statusDot.className = "dot err";
        statusText.textContent = "Model unavailable";
        statusText.title = d.error;
      } else if (d.ready) {
        statusDot.className = "dot ok";
        statusText.textContent = `Model ready${d.load_seconds ? ` · ${d.load_seconds.toFixed(2)}s` : ""}`;
        statusText.title = `${d.model}\n${d.model_dir}`;
      } else {
        statusDot.className = "dot load";
        statusText.textContent = "Loading model…";
        setTimeout(loadStatus, 1200);
      }
    } catch (e) {
      statusDot.className = "dot err";
      statusText.textContent = "Server not reachable";
    }
  }

  // ---- tabs ---------------------------------------------------------------
  function showTab(which) {
    const isNew = which === "new";
    tabNew.classList.toggle("is-active", isNew);
    tabHistory.classList.toggle("is-active", !isNew);
    tabNew.setAttribute("aria-selected", String(isNew));
    tabHistory.setAttribute("aria-selected", String(!isNew));
    paneNew.classList.toggle("is-active", isNew);
    paneHistory.classList.toggle("is-active", !isNew);
    if (!isNew) loadHistory();
  }

  tabNew.addEventListener("click", () => showTab("new"));
  tabHistory.addEventListener("click", () => showTab("history"));

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
    mediaRecorder.onstop = () => {
      const type = mediaRecorder.mimeType || mime || "audio/webm";
      const blob = new Blob(chunks, { type });
      if (blob.size === 0) {
        showToast("Nothing was recorded. Try again and speak a little louder.", true);
        recState.textContent = "Ready to record";
        return;
      }
      const ext = extForMime(type, "webm");
      setPending(blob, `recording.${ext}`, "recording", "Recording captured. Press Send to transcribe.");
      recHint.textContent = "Listen back, re-record if needed, then press Send.";
    };

    mediaRecorder.start();
    isRecording = true;
    recBtn.classList.add("is-recording");
    recBtn.setAttribute("aria-label", "Stop recording");
    recState.textContent = "Recording… speak now";
    recHint.textContent = "Press the button again when you are done.";
    startedAt = Date.now();
    recTimer.textContent = "0:00";
    timerId = setInterval(tick, 200);
  }

  function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    clearInterval(timerId);
    timerId = null;
    recBtn.classList.remove("is-recording");
    recBtn.setAttribute("aria-label", "Start recording");
    recState.textContent = "Finishing…";
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
    clearPending();
    resultCard.hidden = true;
    recState.textContent = "Ready to record";
    recHint.textContent = "Press the microphone and start speaking in Amharic.";
    recTimer.textContent = "0:00";
  });

  // ---- file upload --------------------------------------------------------
  function takeFile(file) {
    if (!file) return;
    if (isSending) return;
    clearPending();
    resultCard.hidden = true;
    setPending(file, file.name, "upload",
      `Selected "${file.name}". Press Send to transcribe.`);
    recHint.textContent = "Check the preview, then press Send.";
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

  // ---- send ---------------------------------------------------------------
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

    setSending(true);
    recState.textContent = "Transcribing on the CPU…";

    const fd = new FormData();
    fd.append("audio", blob, name);
    fd.append("source", source);

    try {
      const r = await fetch("/api/transcribe", { method: "POST", body: fd });
      let data = null;
      try { data = await r.json(); } catch (_) { /* non-JSON */ }

      if (!r.ok) {
        throw new Error((data && data.error) || `Server error (${r.status})`);
      }
      if (!data) throw new Error("The server returned an unreadable reply.");

      const text = (data.text || "").trim();
      resultText.textContent = text || "(no speech detected)";
      resultMeta.innerHTML = metricsChips(data);
      resultCard.hidden = false;
      recState.textContent = "Done. Transcription added to History.";
      recHint.textContent = "Press the microphone to record another, or open History.";

      clearPending();
      recTimer.textContent = "0:00";
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
    const text = resultText.textContent || "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard.");
    } catch (_) {
      const range = document.createRange();
      range.selectNodeContents(resultText);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      showToast("Selected — press Ctrl+C to copy.");
    }
  });

  // ---- history ------------------------------------------------------------
  const TRASH = '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7m-8 0 .8 12.1A1.5 1.5 0 0 0 9.3 20.5h5.4a1.5 1.5 0 0 0 1.5-1.4L17 7"/></svg>';

  function renderHistory(items) {
    historyList.innerHTML = "";

    if (!items.length) {
      historyEmpty.hidden = false;
      clearAllBtn.hidden = true;
      historyCount.hidden = true;
      return;
    }

    historyEmpty.hidden = true;
    clearAllBtn.hidden = false;
    historyCount.hidden = false;
    historyCount.textContent = String(items.length);

    const frag = document.createDocumentFragment();

    items.forEach((it) => {
      const el = document.createElement("article");
      el.className = "hist-item";
      el.dataset.id = it.id;

      const top = document.createElement("div");
      top.className = "hist-top";
      const when = document.createElement("span");
      when.className = "hist-when";
      when.textContent = fmtWhen(it.created_at);
      const dur = document.createElement("span");
      dur.className = "chip";
      dur.textContent = `${fmtSecs(it.duration_s)} · RTF ${Number(it.rtf || 0).toFixed(2)}`;
      top.append(when, dur);

      const text = document.createElement("p");
      const body = (it.text || "").trim();
      text.className = "hist-text" + (body ? "" : " is-empty");
      text.textContent = body || "(no speech detected)";

      const bottom = document.createElement("div");
      bottom.className = "hist-bottom";

      if (it.audio_file) {
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = `/api/history/${encodeURIComponent(it.id)}/audio`;
        bottom.append(audio);
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
      renderHistory(Array.isArray(items) ? items : []);
    } catch (_) {
      /* leave whatever is on screen */
    }
  }

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
  loadStatus();
  loadHistory();
})();
