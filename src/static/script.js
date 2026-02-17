/* ============================================================
   DROSS Command Center v3.2 - Script
   ============================================================ */

// Configure Marked for security and line breaks
marked.setOptions({
    gfm: true,
    breaks: true,
    sanitize: false, // We'll trust the agent output but ideally sanitize if it were a public app
    headerIds: false,
    mangle: false
});

const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";

let ws;
let reconnectDelay = 1000;
let reconnectTimer = null;

function connectWebSocket() {
    ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

    ws.onopen = () => {
        reconnectDelay = 1000;
        clearTimeout(reconnectTimer);
        addLog("SYSTEM: Uplink established.", "success");
        updateStatus("ONLINE");
        if (els.connectionDot) els.connectionDot.className = "status-indicator online";
        fetchStatus();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "response") {
            addMessage("assistant", data.content);
            updateStatus("IDLE");
        } else if (data.type === "status") {
            updateStatus(data.status.toUpperCase());
        } else if (data.type === "log") {
            let type = "system";
            if (data.content.toLowerCase().includes("error")) type = "error";
            if (data.content.includes("⚡")) type = "success";
            addLog(data.content, type);

            if (data.content.includes("Heartbeat:")) {
                const thought = data.content.replace(/.*Heartbeat:\s*(Autonomy:\s*)?/, "").trim();
                if (thought) addThought(thought);
            }
        } else if (data.type === "refresh_status") {
            fetchStatus();
        } else if (data.type === "refresh_journal") {
            if (document.getElementById('tab-journal').classList.contains('active')) {
                loadJournal();
            }
        }
    };

    ws.onclose = () => {
        addLog(`SYSTEM: Uplink lost. Reconnecting in ${reconnectDelay / 1000}s...`, "error");
        updateStatus("OFFLINE");
        if (els.connectionDot) els.connectionDot.className = "status-indicator";
        reconnectTimer = setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 2, 30000);
            connectWebSocket();
        }, reconnectDelay);
    };
}

// DOM Elements
const els = {
    chatWindow: document.getElementById("chat-window"),
    userInput: document.getElementById("user-input"),
    sendBtn: document.getElementById("send-btn"),
    statusPill: document.getElementById("agent-status-pill"),
    statusText: document.getElementById("agent-status-text"),
    goalTitle: document.getElementById("goal-title"),
    subtaskList: document.getElementById("subtask-list"),
    logConsole: document.getElementById("log-console"),
    memoryCount: document.getElementById("memory-count"),
    connectionDot: document.getElementById("connection-dot"),
    thoughtStream: document.getElementById("proactive-stream"),
    journalFeed: document.getElementById("journal-entries"),
    ollamaHealth: document.getElementById("ollama-health-grid"),
    systemInfo: document.getElementById("system-info-box")
};

connectWebSocket();

// --- Core UI Functions ---

function updateStatus(status) {
    els.statusText.innerText = status;
    els.statusPill.className = "status-pill";

    if (status === "THINKING") {
        els.statusPill.style.color = "var(--yellow)";
        els.statusPill.style.borderColor = "rgba(255, 215, 0, 0.2)";
    } else if (status === "SPEAKING") {
        els.statusPill.style.color = "var(--cyan)";
        els.statusPill.style.borderColor = "rgba(0, 243, 255, 0.2)";
    } else if (status === "OFFLINE") {
        els.statusPill.style.color = "var(--red)";
        els.statusPill.style.borderColor = "rgba(255, 49, 49, 0.2)";
    } else {
        els.statusPill.style.color = "var(--green)";
        els.statusPill.style.borderColor = "rgba(0, 255, 136, 0.1)";
    }
}

function addMessage(role, text) {
    const div = document.createElement("div");
    div.classList.add("message", role);

    if (role === "assistant") {
        div.innerHTML = marked.parse(text);
    } else {
        div.innerText = text;
    }

    els.chatWindow.appendChild(div);
    els.chatWindow.scrollTop = els.chatWindow.scrollHeight;
}

function addLog(text, type = "system") {
    const div = document.createElement("div");
    div.classList.add("log-line", type);
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    div.innerText = `[${time}] ${text}`;
    els.logConsole.appendChild(div);
    els.logConsole.scrollTop = els.logConsole.scrollHeight;

    while (els.logConsole.children.length > 200) els.logConsole.removeChild(els.logConsole.firstChild);
}

function addThought(text) {
    const div = document.createElement("div");
    div.classList.add("thought-entry");
    div.innerText = text;
    els.thoughtStream.prepend(div);
    if (els.thoughtStream.children.length > 20) els.thoughtStream.removeChild(els.thoughtStream.lastChild);
}

function sendMessage() {
    const text = els.userInput.value.trim();
    if (!text) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addLog("SYSTEM: Cannot send — uplink not connected.", "error");
        return;
    }
    addMessage("user", text);
    ws.send(text);
    els.userInput.value = "";
    updateStatus("THINKING");
}

els.userInput.addEventListener("keypress", (e) => { if (e.key === "Enter") sendMessage(); });
els.sendBtn.addEventListener("click", sendMessage);

// --- Data Fetching ---

function renderSubtasks(tasks) {
    els.subtaskList.innerHTML = "";
    if (tasks.length === 0) {
        els.subtaskList.innerHTML = "<div class='empty-state'>No active subtasks</div>";
        return;
    }
    tasks.forEach(t => {
        const div = document.createElement("div");
        div.classList.add("subtask-item");
        if (t.status === "completed") div.classList.add("completed");
        div.innerText = t.description;
        els.subtaskList.appendChild(div);
    });
}

async function loadJournal() {
    try {
        const res = await fetch('/api/journal');
        const data = await res.json();
        const container = els.journalFeed;

        if (!data.entries || data.entries.length === 0) {
            container.innerHTML = '<div class="empty-state">No entries found.</div>';
            return;
        }

        container.innerHTML = '';
        data.entries.slice().reverse().forEach(entry => {
            const div = document.createElement('div');
            div.className = 'entry-card';

            const time = new Date(entry.timestamp).toLocaleString();
            let content = entry.entry;
            let outcome = '';

            try {
                const p = JSON.parse(content);
                content = p.lessons || content;
                if (p.outcome) outcome = `<div class="outcome-tag" style="color:var(--${getOutcomeColor(p.outcome)})">${p.outcome}</div>`;
                if (p.what_failed) content += `<br><br><span style="color:var(--red)">ERR:</span> ${p.what_failed}`;
            } catch (e) { }

            div.innerHTML = `
                <div class="entry-time">${time}</div>
                <div class="entry-body">${content}</div>
                ${outcome}
            `;
            container.appendChild(div);
        });
    } catch (e) {
        console.error(e);
    }
}

function getOutcomeColor(o) {
    if (o === 'success') return 'green';
    if (o === 'failure') return 'red';
    return 'yellow';
}

async function fetchSystemInfo() {
    try {
        const res = await fetch("/api/system_info");
        const data = await res.json();
        if (data.error) return;

        els.systemInfo.innerHTML = "";
        Object.entries(data).forEach(([key, val]) => {
            const div = document.createElement("div");
            div.className = "info-item";
            div.innerHTML = `<span class="label">${key.replace(/_/g, ' ')}</span><span class="value">${val}</span>`;
            els.systemInfo.appendChild(div);
        });
    } catch (e) { console.error(e); }
}

function renderOllamaHealth(healthMap) {
    els.ollamaHealth.innerHTML = "";
    const entries = Object.entries(healthMap);
    if (entries.length === 0) {
        els.ollamaHealth.innerHTML = "<div class='empty-state'>No hosts configured</div>";
        return;
    }
    entries.forEach(([host, ok]) => {
        const div = document.createElement("div");
        div.className = "status-item";
        const cleanHost = host.replace('https://', '').replace('http://', '');
        div.innerHTML = `
            <span class="label">${cleanHost}</span>
            <span class="value ${ok ? 'online' : 'offline'}">${ok ? 'ACTIVE' : 'OFFLINE'}</span>
        `;
        els.ollamaHealth.appendChild(div);
    });
}

// --- Navigation ---

function switchMainView(viewId) {
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

    const targetView = document.getElementById(`view-${viewId}`);
    if (targetView) targetView.classList.add('active');

    const targetBtn = document.getElementById(`btn-${viewId}`);
    if (targetBtn) targetBtn.classList.add('active');

    if (viewId === 'graph') loadGraph();
    if (viewId === 'settings') fetchSystemInfo();
}

function switchRightTab(tabId) {
    document.querySelectorAll('.panel-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    const targetTab = document.getElementById(`tab-${tabId}`);
    if (targetTab) targetTab.classList.add('active');

    const targetBtn = document.getElementById(`tab-btn-${tabId}`);
    if (targetBtn) targetBtn.classList.add('active');

    if (tabId === 'journal') loadJournal();
}

window.switchMainView = switchMainView;
window.switchRightTab = switchRightTab;

window.wipeMemory = () => {
    if (confirm("PURGE ALL MEMORY?")) fetch('/api/memory/clear', { method: 'POST' }).then(() => fetchStatus());
};

window.fullReset = () => {
    if (confirm("TOTAL SYSTEM RESET? All memories and goals will be lost.")) {
        fetch('/api/reset', { method: 'POST' }).then(r => r.json()).then(data => {
            addLog("SYSTEM: Full reset complete.", "success");
            fetchStatus();
        });
    }
};

// --- Graph ---
let network = null;
let graphData = { nodes: new vis.DataSet(), edges: new vis.DataSet() };
let lastMemoryCount = -1;

function loadGraph(force = false) {
    const container = document.getElementById('network-container');
    if (!container) return;

    fetch('/api/memory/graph').then(r => r.json()).then(data => {
        if (!data.nodes) return;

        if (!network) {
            graphData.nodes.add(data.nodes);
            graphData.edges.add(data.edges);
            const options = {
                nodes: {
                    shape: 'dot',
                    font: { color: '#888', size: 11, face: 'JetBrains Mono' },
                    borderWidth: 2,
                    shadow: true
                },
                edges: {
                    color: { color: '#222', highlight: '#444' },
                    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
                    smooth: { type: 'continuous' }
                },
                physics: {
                    stabilization: false,
                    barnesHut: { gravitationalConstant: -3000, springLength: 120 }
                }
            };
            network = new vis.Network(container, graphData, options);
        } else if (force) {
            const existingNodeIds = new Set(graphData.nodes.getIds());
            const newNodes = data.nodes.filter(n => !existingNodeIds.has(n.id));
            if (newNodes.length > 0) graphData.nodes.add(newNodes);

            const existingEdges = graphData.edges.get();
            const existingEdgeKeys = new Set(existingEdges.map(e => `${e.from}|${e.to}|${e.label}`));
            const newEdges = data.edges.filter(e => !existingEdgeKeys.has(`${e.from}|${e.to}|${e.label}`));
            if (newEdges.length > 0) graphData.edges.add(newEdges);
        }
    });
}

async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();

        if (els.memoryCount) {
            const count = data.memory_count || 0;
            els.memoryCount.innerText = count;
            if (count > lastMemoryCount) {
                if (lastMemoryCount !== -1 && document.getElementById('view-graph').classList.contains('active')) {
                    loadGraph(true);
                }
                lastMemoryCount = count;
            }
        }

        if (data.ollama_health) renderOllamaHealth(data.ollama_health);

        if (data.goal && data.goal.goal && data.goal.goal !== "Idle") {
            els.goalTitle.innerText = data.goal.goal;
            els.goalTitle.style.color = "var(--text-main)";
            renderSubtasks(data.goal.subtasks || []);
        } else {
            els.goalTitle.innerText = "No active goal";
            els.goalTitle.style.color = "var(--text-dim)";
            els.subtaskList.innerHTML = "<div class='empty-state'>System Idle</div>";
        }
    } catch (e) {
        console.error(e);
    }
}

setInterval(fetchStatus, 10000);
fetchStatus();
