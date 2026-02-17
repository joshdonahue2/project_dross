/* ============================================================
   DROSS // Ether Script
   ============================================================ */

marked.setOptions({ gfm: true, breaks: true });

const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
let ws;
let reconnectTimer = null;

// DOM Selectors
const els = {
    orb: document.getElementById('nexus-orb'),
    stateLabel: document.getElementById('nexus-state'),
    activityLabel: document.getElementById('nexus-activity'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    liveStream: document.getElementById('live-stream'),
    missionGoal: document.getElementById('mission-goal'),
    missionSubtasks: document.getElementById('mission-subtasks'),
    memCountBadge: document.getElementById('mem-count-badge'),
    systemInfo: document.getElementById('system-info'),
    ollamaHealth: document.getElementById('ollama-health'),
    toolsGrid: document.getElementById('tools-grid'),
    journalFeed: document.getElementById('journal-feed'),
    fleetGrid: document.getElementById('fleet-grid'),
    filesList: document.getElementById('files-list')
};

function connect() {
    ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

    ws.onopen = () => {
        setOrbState('idle');
        els.stateLabel.innerText = "Core Connected";
        addStreamLine("System uplink established.", "sys");
        fetchStatus();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWsMessage(data);
    };

    ws.onclose = () => {
        setOrbState('idle');
        els.stateLabel.innerText = "Connection Lost";
        setTimeout(connect, 2000);
    };
}

function handleWsMessage(data) {
    switch (data.type) {
        case 'status':
            setOrbState(data.status);
            if (data.status === 'thinking') els.activityLabel.innerText = "Processing neural patterns...";
            else if (data.status === 'speaking') els.activityLabel.innerText = "Transmitting response...";
            else if (data.status === 'idle') els.activityLabel.innerText = "Awaiting interaction...";
            break;

        case 'tool_start':
            setOrbState('thinking');
            const args = JSON.stringify(data.args);
            els.activityLabel.innerText = `Executing: ${data.tool}`;
            addStreamLine(`⚡ Running ${data.tool}(${args.substring(0, 30)}${args.length > 30 ? '...' : ''})`, "tool");
            break;

        case 'response':
            addChatMessage('assistant', data.content);
            setOrbState('idle');
            els.activityLabel.innerText = "Task complete.";
            break;

        case 'log':
            addStreamLine(data.content);
            break;

        case 'refresh_status':
            fetchStatus();
            break;

        case 'refresh_journal':
            if (document.getElementById('view-journal').classList.contains('active')) fetchJournal();
            break;
    }
}

function setOrbState(state) {
    els.orb.className = 'orb ' + state;
    if (state === 'thinking') els.stateLabel.innerText = "Nexus Reasoning";
    else if (state === 'speaking') els.stateLabel.innerText = "Nexus Speaking";
    else els.stateLabel.innerText = "Nexus Core";
}

function addChatMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    if (role === 'assistant') {
        div.innerHTML = marked.parse(text);
    } else {
        div.innerText = text;
    }
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function addStreamLine(text, type = "") {
    const div = document.createElement('div');
    div.className = `stream-line ${type}`;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
    div.innerText = `[${time}] ${text}`;
    els.liveStream.prepend(div);
    if (els.liveStream.children.length > 50) els.liveStream.removeChild(els.liveStream.lastChild);
}

async function sendMessage() {
    const text = els.chatInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    addChatMessage('user', text);
    ws.send(text);
    els.chatInput.value = "";
    setOrbState('thinking');
}

els.sendBtn.onclick = sendMessage;
els.chatInput.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };

// View Switching
function switchView(viewId) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(v => v.classList.remove('active'));

    document.getElementById(`view-${viewId}`).classList.add('active');
    document.getElementById(`nav-${viewId}`).classList.add('active');

    if (viewId === 'knowledge') loadKnowledge();
    if (viewId === 'system') fetchSystemInfo();
    if (viewId === 'tools') fetchTools();
    if (viewId === 'journal') fetchJournal();
    if (viewId === 'fleet') fetchFleet();
    if (viewId === 'files') fetchFiles();
}

async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        els.memCountBadge.innerText = data.memory_count || 0;

        if (data.goal && data.goal.goal && data.goal.goal !== "Idle") {
            els.missionGoal.innerText = data.goal.goal;
            renderSubtasks(data.goal.subtasks || []);
        } else {
            els.missionGoal.innerText = "Awaiting Mission";
            els.missionSubtasks.innerHTML = "";
        }

        // Update fleet if view is active
        if (document.getElementById('view-fleet').classList.contains('active')) renderFleet(data.subagents);
    } catch (e) { console.error(e); }
}

function renderSubtasks(tasks) {
    els.missionSubtasks.innerHTML = "";
    tasks.forEach(t => {
        const div = document.createElement('div');
        div.className = 'task-item' + (t.status === 'completed' ? ' done' : '');
        div.innerText = t.description;
        els.missionSubtasks.appendChild(div);
    });
}

async function fetchTools() {
    try {
        const res = await fetch('/api/tools');
        const data = await res.json();
        if (!data.tools) return;

        els.toolsGrid.innerHTML = data.tools.map(tool => `
            <div class="tool-card glass">
                <h3>${tool.name}</h3>
                <p>${tool.description}</p>
                <div class="tool-meta" style="margin-top:10px; font-size:0.7rem; color:var(--text-dim); font-family:var(--font-mono);">
                    Params: ${Object.keys(tool.parameters.properties).join(', ') || 'none'}
                </div>
            </div>
        `).join('');
    } catch (e) { console.error(e); }
}

async function fetchJournal() {
    try {
        const res = await fetch('/api/journal');
        const data = await res.json();
        if (!data.entries) return;

        els.journalFeed.innerHTML = data.entries.slice().reverse().map(entry => {
            let content = entry.entry;
            let outcome = '';
            try {
                const p = JSON.parse(content);
                content = p.lessons || p.entry || content;
                if (p.outcome) outcome = `<div class="outcome-badge" style="color:var(--${p.outcome === 'success' ? 'accent-blue' : 'fail'})">${p.outcome}</div>`;
            } catch (e) {}

            return `
                <div class="journal-card glass">
                    <div class="timestamp">${new Date(entry.timestamp).toLocaleString()}</div>
                    <div class="entry">${content}</div>
                    ${outcome}
                </div>
            `;
        }).join('');
    } catch (e) { console.error(e); }
}

async function fetchFleet() {
    fetchStatus(); // Fleet is rendered within fetchStatus for live updates
}

function renderFleet(agents) {
    if (!agents || agents.length === 0) {
        els.fleetGrid.innerHTML = '<div class="msg sys" style="grid-column: 1/-1;">No subagents currently deployed.</div>';
        return;
    }

    els.fleetGrid.innerHTML = agents.map(a => `
        <div class="fleet-card glass">
            <h3>AGENT ${a.id.substring(0, 8)}</h3>
            <p>${a.goal}</p>
            <div class="status ${a.status}">${a.status}</div>
            <div style="margin-top:10px; font-size:0.75rem; color:var(--text-dim);">
                Runtime: ${a.runtime_seconds}s
            </div>
        </div>
    `).join('');
}

async function fetchFiles() {
    try {
        const res = await fetch('/api/files');
        const data = await res.json();
        if (!data.files) return;

        els.filesList.innerHTML = data.files.map(f => `
            <div class="file-row">
                <div class="name">
                    <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
                    ${f.path}
                </div>
                <div class="size">${(f.size / 1024).toFixed(1)} KB</div>
            </div>
        `).join('');
    } catch (e) { console.error(e); }
}

async function fetchSystemInfo() {
    try {
        const res = await fetch('/api/system_info');
        const data = await res.json();
        els.systemInfo.innerHTML = Object.entries(data).map(([k, v]) => `
            <div class="info-row">
                <span class="label">${k.replace(/_/g, ' ')}</span>
                <span class="val">${v}</span>
            </div>
        `).join('');

        const res2 = await fetch('/api/status');
        const data2 = await res2.json();
        els.ollamaHealth.innerHTML = Object.entries(data2.ollama_health || {}).map(([host, ok]) => `
            <div class="info-row">
                <span class="label">${host.replace('https://', '').replace('http://', '')}</span>
                <span class="val ${ok ? 'ok' : 'fail'}">${ok ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
        `).join('');
    } catch (e) {}
}

// Knowledge Graph
let network = null;
async function loadKnowledge() {
    const container = document.getElementById('network-viz');
    if (!container) return;
    try {
        const res = await fetch('/api/memory/graph');
        const data = await res.json();

        const options = {
            nodes: {
                shape: 'dot',
                size: 16,
                font: { color: '#fff', size: 12, face: 'Plus Jakarta Sans' },
                borderWidth: 2,
                color: { background: '#14141e', border: '#4facfe', highlight: { background: '#4facfe', border: '#fff' } }
            },
            edges: {
                color: { color: 'rgba(255,255,255,0.1)', highlight: 'rgba(255,255,255,0.4)' },
                arrows: { to: { enabled: true, scaleFactor: 0.5 } }
            },
            physics: { barnesHut: { gravitationalConstant: -2000 } }
        };

        if (network) network.destroy();
        network = new vis.Network(container, { nodes: new vis.DataSet(data.nodes), edges: new vis.DataSet(data.edges) }, options);
    } catch (e) {}
}

// Global Actions
window.wipeMemory = async () => {
    if (confirm("Purge all neural nodes?")) {
        await fetch('/api/memory/clear', { method: 'POST' });
        fetchStatus();
    }
};

window.fullReset = async () => {
    if (confirm("Total system reset? All mission data will be lost.")) {
        await fetch('/api/reset', { method: 'POST' });
        location.reload();
    }
};

window.switchView = switchView;

// Init
connect();
setInterval(fetchStatus, 10000);
