/* ─────────────────────────────────────────────────────────────────────────────
   Muses Frontend — app.js
   Handles: file upload, document listing, SSE chat, citation rendering, toasts
───────────────────────────────────────────────────────────────────────────── */

const API = '';  // Same-origin; FastAPI serves this file

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone          = document.getElementById('dropZone');
const fileInput         = document.getElementById('fileInput');
const btnUpload         = document.getElementById('btnUpload');
const uploadProgressList= document.getElementById('uploadProgressList');
const docsList          = document.getElementById('docsList');
const docsEmpty         = document.getElementById('docsEmpty');
const btnRefresh        = document.getElementById('btnRefresh');
const emptyState        = document.getElementById('emptyState');
const messages          = document.getElementById('messages');
const queryInput        = document.getElementById('queryInput');
const btnSend           = document.getElementById('btnSend');
const toastContainer    = document.getElementById('toastContainer');

// Suggestion chips
['chip1','chip2','chip3'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', () => {
    queryInput.value = el.textContent;
    queryInput.dispatchEvent(new Event('input'));
    submitQuery();
  });
});

// ── State ─────────────────────────────────────────────────────────────────────
let isStreaming = false;

// ── Toasts ───────────────────────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  const icons = { success: '✓', error: '✕', info: '✦' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type]}</span><span>${message}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('fade-out');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  }, duration);
}

// ── Auto-resize textarea ──────────────────────────────────────────────────────
queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + 'px';
  btnSend.disabled = isStreaming || queryInput.value.trim() === '';
});

queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!btnSend.disabled) submitQuery();
  }
});

// ── Upload — drag & drop ──────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles([...e.dataTransfer.files]);
});
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
btnUpload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFiles([...fileInput.files]);
  fileInput.value = '';
});

// ── Upload — handle files ─────────────────────────────────────────────────────
async function handleFiles(files) {
  if (!files.length) return;

  // Build FormData
  const formData = new FormData();
  files.forEach(f => formData.append('files', f));

  // Create progress items
  const items = files.map(f => createProgressItem(f.name));

  try {
    items.forEach(it => setItemStatus(it, 'loading'));

    const res = await fetch(`${API}/api/ingest`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const data = await res.json();

    data.results.forEach((result, i) => {
      const it = items[i];
      if (!it) return;
      if (result.status === 'success') {
        setItemStatus(it, 'success', `${result.chunks} chunks`);
        showToast(`"${result.filename}" ingested (${result.chunks} chunks)`, 'success');
      } else {
        setItemStatus(it, 'error', 'Failed');
        showToast(`Failed: ${result.filename} — ${result.message}`, 'error');
      }
    });

    // Refresh document list after short delay
    setTimeout(loadDocuments, 600);

  } catch (err) {
    items.forEach(it => setItemStatus(it, 'error', 'Error'));
    showToast(`Upload failed: ${err.message}`, 'error');
  }
}

function createProgressItem(filename) {
  const el = document.createElement('div');
  el.className = 'upload-item';
  el.innerHTML = `
    <div class="upload-item-name" title="${filename}">${filename}</div>
    <span class="upload-item-status pending">Pending</span>
  `;
  uploadProgressList.appendChild(el);
  return el;
}

function setItemStatus(el, status, label) {
  const statusEl = el.querySelector('.upload-item-status');
  const spinnerEl = el.querySelector('.upload-spinner');
  if (spinnerEl) spinnerEl.remove();

  statusEl.className = `upload-item-status ${status}`;

  if (status === 'loading') {
    const spinner = document.createElement('div');
    spinner.className = 'upload-spinner';
    el.insertBefore(spinner, statusEl);
    statusEl.textContent = 'Ingesting…';
  } else if (status === 'success') {
    statusEl.textContent = label || '✓ Done';
    setTimeout(() => el.remove(), 4000);
  } else if (status === 'error') {
    statusEl.textContent = label || '✕ Error';
    setTimeout(() => el.remove(), 5000);
  }
}

// ── Document list ─────────────────────────────────────────────────────────────
async function loadDocuments() {
  btnRefresh.classList.add('spinning');
  try {
    const res  = await fetch(`${API}/api/documents`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderDocuments(data.documents || []);
  } catch (err) {
    showToast('Could not load documents: ' + err.message, 'error');
  } finally {
    btnRefresh.classList.remove('spinning');
  }
}

function renderDocuments(docs) {
  // Clear current items (keep empty message in DOM)
  [...docsList.querySelectorAll('.doc-item')].forEach(el => el.remove());

  if (!docs.length) {
    docsEmpty.style.display = '';
    return;
  }
  docsEmpty.style.display = 'none';

  docs.forEach(doc => {
    const el = document.createElement('div');
    el.className = 'doc-item';
    const ext  = doc.filename.split('.').pop().toUpperCase();
    const name = doc.filename;
    const downloadUrl = `${API}/api/documents/${encodeURIComponent(name)}/download`;

    el.innerHTML = `
      <div class="doc-info">
        <div class="doc-name" title="${name}">${name}</div>
        <div class="doc-meta">${ext} · ${doc.chunks} chunks</div>
      </div>
      <a class="btn-download" href="${downloadUrl}" download="${name}"
         title="Download ${name}" aria-label="Download ${name}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
      </a>
    `;
    docsList.appendChild(el);
  });
}

btnRefresh.addEventListener('click', loadDocuments);

// ── Chat ──────────────────────────────────────────────────────────────────────
function submitQuery() {
  const query = queryInput.value.trim();
  if (!query || isStreaming) return;

  emptyState.classList.add('hidden');

  appendUserMessage(query);
  queryInput.value = '';
  queryInput.style.height = 'auto';
  btnSend.disabled = true;

  streamAnswer(query);
}

btnSend.addEventListener('click', submitQuery);

function appendUserMessage(text) {
  const el = document.createElement('div');
  el.className = 'message user';
  el.innerHTML = `
    <div class="message-label">You</div>
    <div class="message-bubble">${escapeHtml(text)}</div>
  `;
  messages.appendChild(el);
  scrollToBottom();
}

function appendAssistantMessage() {
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="message-label">Muses</div>
    <div class="message-bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messages.appendChild(el);
  scrollToBottom();
  return el;
}

async function streamAnswer(query) {
  isStreaming = true;
  btnSend.classList.add('loading');
  btnSend.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <line x1="12" y1="2" x2="12" y2="6"/>
      <line x1="12" y1="18" x2="12" y2="22"/>
      <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/>
      <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
      <line x1="2" y1="12" x2="6" y2="12"/>
      <line x1="18" y1="12" x2="22" y2="12"/>
    </svg>`;

  const msgEl = appendAssistantMessage();
  const bubble = msgEl.querySelector('.message-bubble');

  try {
    const res = await fetch(`${API}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query }),
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';
    let   text    = '';
    let   typingRemoved = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }

        if (evt.token !== undefined) {
          // Remove typing indicator on first token
          if (!typingRemoved) {
            bubble.innerHTML = '';
            typingRemoved = true;
          }
          text += evt.token;
          bubble.innerHTML = formatResponse(text);
          scrollToBottom();
        }

        if (evt.done) {
          // Render citations
          if (evt.citations && evt.citations.length > 0) {
            const citEl = buildCitationsCard(evt.citations);
            msgEl.appendChild(citEl);
            scrollToBottom();
          }
        }
      }
    }
  } catch (err) {
    bubble.innerHTML = `<span style="color:var(--error)">Error: ${escapeHtml(err.message)}</span>`;
    showToast('Chat error: ' + err.message, 'error');
  } finally {
    isStreaming = false;
    btnSend.classList.remove('loading');
    btnSend.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <line x1="22" y1="2" x2="11" y2="13"/>
        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
      </svg>`;
    btnSend.disabled = queryInput.value.trim() === '';
  }
}

// ── Citations card ────────────────────────────────────────────────────────────
function buildCitationsCard(citations) {
  const card = document.createElement('div');
  card.className = 'citations-card';

  const toggle = document.createElement('div');
  toggle.className = 'citations-toggle';
  toggle.innerHTML = `
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2.5">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
    ${citations.length} Source${citations.length > 1 ? 's' : ''}
  `;

  const body = document.createElement('div');
  body.className = 'citations-body';

  citations.forEach(c => {
    const item = document.createElement('div');
    item.className = 'citation-item';
    const filename = c.source.split('/').pop().split('\\').pop();
    const downloadUrl = `${API}/api/documents/${encodeURIComponent(filename)}/download`;
    item.innerHTML = `
      <div class="citation-text">
        <div class="citation-source" title="${filename}">${filename}</div>
        <div class="citation-details">Page ${c.page || '—'} · ${c.heading || 'General'}</div>
      </div>
      <a class="btn-cite-download" href="${downloadUrl}" download="${filename}">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download
      </a>
    `;
    body.appendChild(item);
  });

  toggle.addEventListener('click', () => {
    const open = toggle.classList.toggle('open');
    body.classList.toggle('open', open);
  });

  card.appendChild(toggle);
  card.appendChild(body);
  return card;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Minimal markdown-like renderer for the AI response:
 * - **bold**, *italic*, `code`, bullet lists
 */
function formatResponse(text) {
  const escaped = escapeHtml(text);
  return escaped
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code style="font-family:var(--mono);font-size:12px;background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:4px;">$1</code>')
    // Bullet points
    .replace(/^- (.+)$/gm, '<div style="display:flex;gap:8px;margin:2px 0"><span style="color:var(--purple);flex-shrink:0">•</span><span>$1</span></div>')
    // Line breaks
    .replace(/\n/g, '<br>');
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadDocuments();
