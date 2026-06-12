/**
 * Local AI Voice Clone Assistant — Frontend logic
 *
 * Manages:
 *  - WebSocket connection to /ws/chat
 *  - Orb state machine (idle | thinking | speaking)
 *  - Terminal-style chat transcript
 *  - Voice Studio panel (list, add, remove, preview)
 *  - System status polling
 */

'use strict';

// ===================================================================
// DOM references
// ===================================================================

const $ = id => document.getElementById(id);

const orbWrap        = $('orbWrap');
const orbLabel       = $('orbLabel');
const chatMessages   = $('chatMessages');
let   chatEmpty      = $('chatEmpty');
const chatInput      = $('chatInput');
const sendBtn        = $('sendBtn');
const micBtn         = $('micBtn');
const voiceSelect    = $('voiceSelect');
const toggleSpeak    = $('toggleSpeak');

// Corner status elements
const dotLlm     = $('dotLlm');
const dotTts     = $('dotTts');
const statLlm    = $('statLlm');
const statTts    = $('statTts');
const statVoice  = $('statVoice');
const statStatus = $('statStatus');
const statModel  = $('statModel');

// Voice Studio
const btnStudio      = $('btnVoiceStudio');
const btnCloseStudio = $('btnCloseStudio');
const voiceStudio    = $('voiceStudio');
const overlay        = $('overlay');
const studioVoices   = $('studioVoices');
const voiceName      = $('voiceName');
const voiceFile      = $('voiceFile');
const fileDrop       = $('fileDrop');
const fileLabel      = $('fileLabel');
const voiceTranscript= $('voiceTranscript');
const btnAddVoice    = $('btnAddVoice');
const addStatus      = $('addStatus');

// ===================================================================
// Orb state machine
// ===================================================================

const ORB_STATES = { IDLE: 'idle', THINKING: 'thinking', SPEAKING: 'speaking' };
let orbState = ORB_STATES.IDLE;

function setOrb(state) {
  orbWrap.classList.remove('thinking', 'speaking', 'listening');
  if (state !== ORB_STATES.IDLE) orbWrap.classList.add(state);
  orbLabel.textContent = state.toUpperCase();
  orbState = state;
  if (window.VoiceOrb) VoiceOrb.setState(state);
}

// ===================================================================
// Audio playback — cloned voice streams from the server as base64 WAV
// chunks (one per sentence). We decode + schedule them gaplessly with
// the Web Audio API so playback happens in the BROWSER, not the server.
// ===================================================================

let audioCtx        = null;
let playbackAnalyser = null;        // taps the cloned-voice output for the visualizer
let nextStartTime   = 0;            // when the next chunk should start (ctx clock)
let audioChain      = Promise.resolve();  // serialises decode+schedule in order
let audioEnqueued   = 0;            // chunks received but not finished playing
let audioStreamEnded = false;       // server has sent the last chunk for this turn
let expectAudio     = false;        // did we ask for spoken output this turn?
let audioTurn       = 0;            // bumped each turn to discard stale chunks
let activeSources   = [];

const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    playbackAnalyser = audioCtx.createAnalyser();
    playbackAnalyser.fftSize = 256;
    playbackAnalyser.smoothingTimeConstant = 0.82;
    playbackAnalyser.connect(audioCtx.destination);
  }
  if (audioCtx.state === 'suspended') audioCtx.resume();
  return audioCtx;
}

// Audio arrives as binary WebSocket frames (16-bit WAV bytes) — no base64
// decode needed; the ArrayBuffer goes straight to decodeAudioData.
function enqueueAudio(arrayBuf) {
  const turn = audioTurn;
  audioEnqueued++;
  audioChain = audioChain.then(() => playChunk(arrayBuf, turn));
}

async function playChunk(arrayBuf, turn) {
  if (turn !== audioTurn) { audioEnqueued--; maybeFinishTurn(); return; }
  const ctx = ensureAudioCtx();
  let buffer;
  try {
    buffer = await ctx.decodeAudioData(arrayBuf);
  } catch (e) {
    audioEnqueued--; maybeFinishTurn(); return;
  }
  if (turn !== audioTurn) { audioEnqueued--; maybeFinishTurn(); return; }

  const src = ctx.createBufferSource();
  src.buffer = buffer;

  // Route through a gain node and apply short fades at both edges; schedule
  // the next chunk to overlap this one's tail by `fade` for a crossfade.
  // This removes the clicks/discontinuities at chunk boundaries.
  const gain = ctx.createGain();
  src.connect(gain);
  gain.connect(playbackAnalyser);

  const startAt = Math.max(ctx.currentTime, nextStartTime);
  const dur  = buffer.duration;
  const fade = Math.min(0.018, dur / 3);
  const g = gain.gain;
  g.setValueAtTime(0.0001, startAt);
  g.exponentialRampToValueAtTime(1.0, startAt + fade);
  g.setValueAtTime(1.0, Math.max(startAt + fade, startAt + dur - fade));
  g.exponentialRampToValueAtTime(0.0001, startAt + dur);

  src.start(startAt);
  nextStartTime = startAt + dur - fade;
  activeSources.push(src);
  if (orbState !== ORB_STATES.SPEAKING) setOrb(ORB_STATES.SPEAKING);
  startViz(playbackAnalyser, 'voice');

  src.onended = () => {
    activeSources = activeSources.filter(s => s !== src);
    if (turn === audioTurn) audioEnqueued--;
    maybeFinishTurn();
  };
}

function stopAllAudio() {
  activeSources.forEach(s => { try { s.stop(); } catch (_) {} });
  activeSources = [];
  audioEnqueued = 0;
  audioChain = Promise.resolve();
  // Small startup lead so synthesis jitter doesn't cause an early underrun.
  if (audioCtx) nextStartTime = audioCtx.currentTime + 0.08;
}

function maybeFinishTurn() {
  if (audioStreamEnded && audioEnqueued === 0) {
    stopViz();
    setOrb(ORB_STATES.IDLE);
    setInputEnabled(true);
  }
}

// ===================================================================
// Audio visualizer — radial spectrum around the orb, driven by an
// AnalyserNode tapping either the cloned-voice output or the live mic.
// ===================================================================

const vizCanvas = $('vizCanvas');
const vizCtx    = vizCanvas ? vizCanvas.getContext('2d') : null;
const VIZ_SIZE  = 320;
let vizAnalyser = null;
let vizRAF      = null;
let vizMode     = 'voice';
const vizData   = new Uint8Array(128);

function sizeViz() {
  if (!vizCanvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  vizCanvas.width  = VIZ_SIZE * dpr;
  vizCanvas.height = VIZ_SIZE * dpr;
  vizCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function startViz(analyser, mode) {
  if (REDUCED_MOTION || !vizCtx) return;
  vizAnalyser = analyser;
  vizMode = mode;
  if (!vizRAF) vizRAF = requestAnimationFrame(drawViz);
}

function stopViz() {
  vizAnalyser = null;
  if (vizRAF) { cancelAnimationFrame(vizRAF); vizRAF = null; }
  if (vizCtx) vizCtx.clearRect(0, 0, VIZ_SIZE, VIZ_SIZE);
  if (window.VoiceOrb) VoiceOrb.setLevel(0);
}

function lerp(a, b, t) { return a + (b - a) * t; }

function drawViz() {
  if (!vizAnalyser) { vizRAF = null; return; }
  vizRAF = requestAnimationFrame(drawViz);

  vizAnalyser.getByteFrequencyData(vizData);
  const bins = vizAnalyser.frequencyBinCount;       // 128
  const cx = VIZ_SIZE / 2, cy = VIZ_SIZE / 2;
  const bars = 80, innerR = 88, maxLen = 54;
  vizCtx.clearRect(0, 0, VIZ_SIZE, VIZ_SIZE);
  vizCtx.lineCap = 'round';
  vizCtx.lineWidth = 2.6;

  let sum = 0;
  const half = bars / 2;
  for (let i = 0; i < bars; i++) {
    // Mirror the spectrum across the vertical axis for symmetry.
    const m   = i < half ? i : bars - i;
    const bin = Math.floor((m / half) * (bins * 0.72));
    const v   = vizData[bin] / 255;
    sum += v;

    const len = 3 + v * v * maxLen;
    const ang = (i / bars) * Math.PI * 2 - Math.PI / 2;
    const cos = Math.cos(ang), sin = Math.sin(ang);
    const x1 = cx + cos * innerR,        y1 = cy + sin * innerR;
    const x2 = cx + cos * (innerR + len), y2 = cy + sin * (innerR + len);

    let r, g, b;
    if (vizMode === 'mic') {
      r = 255; g = 59; b = 92;
    } else {
      const t = m / half;                 // cyan → purple around the ring
      r = Math.round(lerp(0,   168, t));
      g = Math.round(lerp(245,  85, t));
      b = Math.round(lerp(255, 247, t));
    }
    vizCtx.strokeStyle = `rgba(${r},${g},${b},${0.30 + v * 0.7})`;
    vizCtx.beginPath();
    vizCtx.moveTo(x1, y1);
    vizCtx.lineTo(x2, y2);
    vizCtx.stroke();
  }

  // Feed overall loudness to the plasma orb.
  const amp = sum / bars;
  if (window.VoiceOrb) VoiceOrb.setLevel(Math.min(1, amp * 1.6));
}

// ===================================================================
// WebSocket
// ===================================================================

let ws = null;
let reconnectTimer = null;

function connectWS() {
  clearTimeout(reconnectTimer);
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
  ws.binaryType = 'arraybuffer';   // cloned-voice audio arrives as raw WAV bytes

  ws.addEventListener('open',    ()    => console.log('[WS] connected'));
  ws.addEventListener('error',   ()    => ws.close());
  ws.addEventListener('close',   ()    => {
    // Reset UI so user isn't stuck with a disabled input or wrong orb state.
    stopAllAudio();
    audioStreamEnded = true;
    setOrb(ORB_STATES.IDLE);
    setInputEnabled(true);
    activeBubble = null;
    activeBubbleText = '';
    reconnectTimer = setTimeout(connectWS, 2500);
  });
  ws.addEventListener('message', evt  => {
    if (evt.data instanceof ArrayBuffer) {
      enqueueAudio(evt.data);          // binary frame = one audio chunk
    } else {
      handleMsg(JSON.parse(evt.data));
    }
  });
}

// -- Incoming message handler --------------------------------------

let activeBubble     = null;   // DOM element of the in-progress assistant message
let activeBubbleText = '';     // accumulated text for the active message

function handleMsg(msg) {
  switch (msg.type) {

    case 'token': {
      if (!activeBubble) {
        activeBubble = createBubble('assistant');
        activeBubbleText = '';
        setOrb(ORB_STATES.THINKING);
      }
      activeBubbleText += (activeBubbleText ? ' ' : '') + msg.text.trim();
      const body = activeBubble.querySelector('.msg-text');
      body.innerHTML = escHtml(activeBubbleText) + '<span class="cursor"></span>';
      scrollBottom();
      break;
    }

    case 'done': {
      if (activeBubble) {
        activeBubble.querySelector('.msg-text').textContent = msg.full;
        activeBubble = null;
        activeBubbleText = '';
      }
      scrollBottom();
      // If no spoken output is coming, the turn is over now. Otherwise wait
      // for the audio stream to end and all scheduled chunks to finish.
      if (!expectAudio) {
        setOrb(ORB_STATES.IDLE);
        setInputEnabled(true);
      }
      break;
    }

    case 'speaking': {
      if (!msg.state) {
        // Server finished sending audio chunks; finish once playback drains.
        audioStreamEnded = true;
        maybeFinishTurn();
      }
      break;
    }

    case 'error': {
      addSystemMsg(msg.msg);
      stopAllAudio();
      audioStreamEnded = true;
      setOrb(ORB_STATES.IDLE);
      setInputEnabled(true);
      break;
    }
  }
}

// ===================================================================
// Send message
// ===================================================================

function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addSystemMsg('Not connected — retrying…');
    return;
  }

  // -- Start a fresh turn: discard any audio still playing/queued ------
  ensureAudioCtx();                 // unlock audio on this user gesture
  audioTurn++;
  stopAllAudio();
  audioStreamEnded = false;
  expectAudio = toggleSpeak.checked && !!voiceSelect.value;

  chatInput.value = '';
  setInputEnabled(false);
  setOrb(ORB_STATES.THINKING);

  activeBubble = null;
  activeBubbleText = '';

  createBubble('user').querySelector('.msg-text').textContent = text;
  scrollBottom();

  ws.send(JSON.stringify({
    text,
    voice_id: voiceSelect.value,
    speak:    toggleSpeak.checked,
  }));
}

// ===================================================================
// Voice input — record mic, transcribe locally via Whisper, then send
// ===================================================================

let mediaRecorder = null;
let recChunks     = [];
let recording     = false;
let micSource     = null;
let micAnalyser   = null;

async function toggleRecord() {
  if (recording) { stopRecord(); return; }
  if (chatInput.disabled) return;          // busy with a response
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Tap the mic into the visualizer (analyser only — never to the
    // speakers, which would cause feedback).
    const ctx = ensureAudioCtx();
    micSource = ctx.createMediaStreamSource(stream);
    micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = 256;
    micAnalyser.smoothingTimeConstant = 0.82;
    micSource.connect(micAnalyser);

    mediaRecorder = new MediaRecorder(stream);
    recChunks = [];
    mediaRecorder.ondataavailable = e => { if (e.data.size) recChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      if (micSource) { try { micSource.disconnect(); } catch (_) {} micSource = null; }
      stopViz();
      orbWrap.classList.remove('listening');
      const blob = new Blob(recChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      await transcribeAndSend(blob);
    };
    mediaRecorder.start();
    recording = true;
    micBtn.classList.add('recording');
    orbWrap.classList.add('listening');
    orbLabel.textContent = 'LISTENING';
    if (window.VoiceOrb) VoiceOrb.setState('listening');
    startViz(micAnalyser, 'mic');
  } catch (e) {
    toast(`Mic access failed: ${e.message}`, 'error');
  }
}

function stopRecord() {
  if (mediaRecorder && recording) {
    recording = false;
    micBtn.classList.remove('recording');
    mediaRecorder.stop();
  }
}

async function transcribeAndSend(blob) {
  setOrb(ORB_STATES.THINKING);
  orbLabel.textContent = 'TRANSCRIBING';
  const ext = (blob.type || '').includes('ogg') ? 'ogg' : 'webm';
  const fd = new FormData();
  fd.append('file', blob, `mic.${ext}`);
  try {
    const res  = await fetch('/api/transcribe', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Transcription failed');
    const text = (data.text || '').trim();
    if (text) {
      chatInput.value = text;
      sendMessage();
    } else {
      addSystemMsg('No speech detected.');
      setOrb(ORB_STATES.IDLE);
    }
  } catch (e) {
    addSystemMsg(`Transcribe error: ${e.message}`);
    setOrb(ORB_STATES.IDLE);
  }
}

// ===================================================================
// Chat UI helpers
// ===================================================================

function timestamp() {
  const d = new Date();
  return `(${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')})`;
}

function createBubble(role) {
  if (chatEmpty) { chatEmpty.remove(); chatEmpty = null; }

  const el = document.createElement('p');
  el.className = `msg msg--${role}`;

  const prefix = document.createElement('span');
  prefix.className = 'msg-prefix';
  prefix.textContent = role === 'user' ? 'USER:' : 'AI:';

  const body = document.createElement('span');
  body.className = 'msg-text';
  body.textContent = '';

  const ts = document.createElement('span');
  ts.className = 'msg-ts';
  ts.textContent = ' ' + timestamp();

  el.appendChild(prefix);
  el.appendChild(document.createTextNode(' '));
  el.appendChild(body);
  el.appendChild(ts);

  chatMessages.appendChild(el);
  return el;
}

// -- Toast notifications -------------------------------------------
const toastStack = $('toastStack');

function toast(text, kind = '') {
  if (!toastStack) return;
  const el = document.createElement('div');
  el.className = `toast${kind ? ' toast--' + kind : ''}`;
  const ico = document.createElement('span');
  ico.className = 'toast-ico';
  const tx = document.createElement('span');
  tx.textContent = text;
  el.append(ico, tx);
  toastStack.appendChild(el);

  const ttl = kind === 'error' ? 5200 : 3200;
  setTimeout(() => {
    el.classList.add('out');
    setTimeout(() => el.remove(), 320);
  }, ttl);
}

// Kept for call sites that report problems; routes to an error toast.
function addSystemMsg(text) {
  toast(text, 'error');
}

function scrollBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setInputEnabled(on) {
  chatInput.disabled = !on;
  sendBtn.disabled   = !on;
  micBtn.disabled    = !on;
  if (on) chatInput.focus();
}

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ===================================================================
// System status
// ===================================================================

async function pollStatus() {
  try {
    const data = await fetch('/api/status').then(r => r.json());

    // LLM status
    const llmOk = data.ollama;
    dotLlm.className = `stat-dot ${llmOk ? 'ok' : 'err'}`;
    statLlm.textContent = llmOk ? 'ONLINE' : 'OFFLINE';
    statLlm.className = `stat-val${llmOk ? '' : ' err'}`;

    // TTS status
    const ttsOk = data.tts === 'ready';
    dotTts.className = `stat-dot ${ttsOk ? 'ok' : ''}`;
    statTts.textContent = data.tts ? data.tts.toUpperCase() : '—';
    statTts.className = `stat-val${ttsOk ? '' : ' warn'}`;

    // Overall STATUS (top-right)
    statStatus.textContent = llmOk ? 'ONLINE [o]' : 'OFFLINE [x]';
    statStatus.className = `stat-val${llmOk ? '' : ' err'}`;

    // Model name
    statModel.textContent = (data.model || '—').toUpperCase();

  } catch (_) {
    dotLlm.className = 'stat-dot err';
    statLlm.textContent = 'UNREACHABLE';
    statLlm.className = 'stat-val err';
    statStatus.textContent = 'OFFLINE [x]';
    statStatus.className = 'stat-val err';
  }
}

// ===================================================================
// Voice management
// ===================================================================

async function loadVoices() {
  try {
    const voices = await fetch('/api/voices').then(r => r.json());
    rebuildVoiceSelect(voices);
    rebuildVoiceCards(voices);
    updateVoiceCorner(voices);
  } catch (_) {}
}

function updateVoiceCorner(voices) {
  const current = voices.find(v => v.id === voiceSelect.value);
  statVoice.textContent = current ? current.name.toUpperCase() : (voices.length ? voices[0].name.toUpperCase() : '—');
}

function rebuildVoiceSelect(voices) {
  const prev = voiceSelect.value;
  voiceSelect.innerHTML = '<option value="">None</option>';
  voices.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.id;
    opt.textContent = v.status === 'ready' ? v.name : `${v.name} ⚠`;
    voiceSelect.appendChild(opt);
  });
  if ([...voiceSelect.options].some(o => o.value === prev)) {
    voiceSelect.value = prev;
  } else if (voices.length > 0) {
    voiceSelect.value = voices[0].id;
  }
}

function rebuildVoiceCards(voices) {
  studioVoices.innerHTML = '';
  if (voices.length === 0) {
    studioVoices.innerHTML = '<p class="studio-empty">No voices yet — add one below.</p>';
    return;
  }
  voices.forEach(v => {
    const card = document.createElement('div');
    card.className = 'voice-card';
    card.innerHTML = `
      <div class="voice-card-info">
        <div class="voice-card-name">${escHtml(v.name)}</div>
        <span class="voice-badge ${v.status}">${v.status.toUpperCase()}</span>
      </div>
      <div class="voice-card-actions">
        <button class="voice-btn play"   data-id="${v.id}" title="Preview voice">&#9654;</button>
        <button class="voice-btn remove" data-id="${v.id}" title="Remove voice">&#x2715;</button>
      </div>
    `;
    card.querySelector('.play').addEventListener('click',   () => previewVoice(v.id));
    card.querySelector('.remove').addEventListener('click', () => removeVoice(v.id, v.name));
    studioVoices.appendChild(card);
  });
}

async function previewVoice(id) {
  const audio = new Audio(`/api/voices/${id}/preview`);
  audio.play().catch(e => addSystemMsg(`Preview error: ${e.message}`));
}

async function removeVoice(id, name) {
  if (!confirm(`Remove voice "${name}"?`)) return;
  try {
    const res = await fetch(`/api/voices/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error((await res.json()).detail);
    await loadVoices();
  } catch (e) {
    addSystemMsg(`Remove failed: ${e.message}`);
  }
}

// Preload voice on selection change for faster first synthesis
voiceSelect.addEventListener('change', () => {
  const id = voiceSelect.value;
  if (id) {
    fetch(`/api/voices/${id}/preload`, { method: 'POST' }).catch(() => {});
    statVoice.textContent = voiceSelect.options[voiceSelect.selectedIndex]?.text?.replace(' ⚠', '').toUpperCase() || '—';
  }
});

// -- Add voice form ------------------------------------------------

voiceFile.addEventListener('change', () => {
  if (voiceFile.files[0]) {
    fileLabel.textContent = voiceFile.files[0].name;
    fileDrop.classList.add('active');
  }
});

fileDrop.addEventListener('dragover',  e => { e.preventDefault(); fileDrop.classList.add('active'); });
fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('active'));
fileDrop.addEventListener('drop', e => {
  e.preventDefault();
  fileDrop.classList.remove('active');
  if (e.dataTransfer.files[0]) {
    const dt = new DataTransfer();
    dt.items.add(e.dataTransfer.files[0]);
    voiceFile.files = dt.files;
    fileLabel.textContent = e.dataTransfer.files[0].name;
    fileDrop.classList.add('active');
  }
});

function showAddStatus(msg, cls = '') {
  addStatus.hidden = false;
  addStatus.textContent = msg;
  addStatus.className = `add-status ${cls}`;
}

btnAddVoice.addEventListener('click', async () => {
  const name = voiceName.value.trim();
  const file = voiceFile.files[0];
  if (!name || !file) {
    showAddStatus('NAME AND AUDIO FILE REQUIRED', 'error');
    return;
  }

  const fd = new FormData();
  fd.append('name', name);
  fd.append('file', file);
  fd.append('transcript', voiceTranscript.value.trim());

  btnAddVoice.disabled = true;
  showAddStatus('PROCESSING — THIS MAY TAKE A MINUTE...');

  try {
    const res  = await fetch('/api/voices', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Unknown error');
    showAddStatus('VOICE READY!', 'success');
    toast(`Voice "${name}" is ready`, 'ok');
    voiceName.value = '';
    voiceFile.value = '';
    fileLabel.textContent = 'Drop audio or click to browse';
    fileDrop.classList.remove('active');
    voiceTranscript.value = '';
    await loadVoices();
    setTimeout(() => { addStatus.hidden = true; }, 3000);
  } catch (e) {
    showAddStatus(`ERROR: ${e.message}`, 'error');
  } finally {
    btnAddVoice.disabled = false;
  }
});

// ===================================================================
// Voice Studio panel
// ===================================================================

function openStudio() {
  voiceStudio.classList.add('open');
  voiceStudio.setAttribute('aria-hidden', 'false');
  overlay.classList.add('active');
  overlay.setAttribute('aria-hidden', 'false');
}
function closeStudio() {
  voiceStudio.classList.remove('open');
  voiceStudio.setAttribute('aria-hidden', 'true');
  overlay.classList.remove('active');
  overlay.setAttribute('aria-hidden', 'true');
}

btnStudio.addEventListener('click',      openStudio);
btnCloseStudio.addEventListener('click', closeStudio);
overlay.addEventListener('click',        closeStudio);

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && voiceStudio.classList.contains('open')) closeStudio();
});

// ===================================================================
// Input events
// ===================================================================

// Send button (and Enter) sends typed text; mic button records voice.
sendBtn.addEventListener('click', sendMessage);
micBtn.addEventListener('click', toggleRecord);
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ===================================================================
// Boot
// ===================================================================

window.addEventListener('resize', sizeViz);

(async function init() {
  sizeViz();
  if (window.VoiceOrb) VoiceOrb.init($('orbGl'));
  connectWS();
  await loadVoices();
  await pollStatus();
  setInterval(pollStatus, 10_000);
  chatInput.focus();
})();
