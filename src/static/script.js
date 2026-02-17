/* ============================================================
   DROSS Command Center v3.2 - Script
   ============================================================ */

// Build WebSocket URL dynamically so it works on any host/port, not just localhost:8001
const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";

let ws;
let reconnectDelay = 1000;
let reconnectTimer = null;

function connectWebSocket() {
    ws = new WebSocket(`${wsProto}//${window.location.host}/ws`);

    ws.onopen = () => {
        reconnectDelay = 1000; // Reset backoff on successful connection
        clearTimeout(reconnectTimer);
        addLog("SYSTEM: Uplink established.");
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
            addLog(data.content);
            // Specialized handling — server sends "⚡ Heartbeat: ..." prefix
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
        addLog(`SYSTEM: Uplink lost. Reconnecting in ${reconnectDelay / 1000}s...`);
        updateStatus("OFFLINE");
        if (els.connectionDot) els.connectionDot.className = "status-indicator";
        // Exponential backoff capped at 30s
        reconnectTimer = setTimeout(() => {
            reconnectDelay = Math.min(reconnectDelay * 2, 30000);
            connectWebSocket();
        }, reconnectDelay);
    };

    ws.onerror = () => {
        // onclose fires after onerror, so just swallow this to avoid duplicate handling
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
    journalFeed: document.getElementById("journal-entries")
};

// Kick off connection (defined above, called here so `els` is ready)
connectWebSocket();

// --- Core UI Functions ---

function updateStatus(status) {
    els.statusText.innerText = status;
    els.statusPill.className = "status-pill"; // Reset

    if (status === "THINKING") {
        els.statusPill.style.borderColor = "var(--yellow)";
        els.statusPill.style.color = "var(--yellow)";
        els.statusPill.querySelector('.dot').style.background = "var(--yellow)";
    } else if (status === "SPEAKING") {
        els.statusPill.style.borderColor = "var(--cyan)";
        els.statusPill.style.color = "var(--cyan)";
        els.statusPill.querySelector('.dot').style.background = "var(--cyan)";
    } else if (status === "OFFLINE") {
        els.statusPill.style.borderColor = "var(--red)";
        els.statusPill.style.color = "var(--red)";
        els.statusPill.querySelector('.dot').style.background = "var(--red)";
    } else {
        els.statusPill.style.borderColor = "#333";
        els.statusPill.style.color = "var(--green)";
        els.statusPill.querySelector('.dot').style.background = "var(--green)";
    }
}

function addMessage(role, text) {
    const div = document.createElement("div");
    div.classList.add("message", role);
    div.innerHTML = text.replace(/\n/g, "<br>");
    els.chatWindow.appendChild(div);
    els.chatWindow.scrollTop = els.chatWindow.scrollHeight;
}

function addLog(text) {
    const div = document.createElement("div");
    div.classList.add("log-line");
    // Timestamp
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
    // Prepend to show newest at top
    els.thoughtStream.prepend(div);
    if (els.thoughtStream.children.length > 20) els.thoughtStream.removeChild(els.thoughtStream.lastChild);
}

function sendMessage() {
    const text = els.userInput.value.trim();
    if (!text) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addLog("SYSTEM: Cannot send — uplink not connected.");
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

// fetchStatus moved to the bottom with graph reactive logic

function renderSubtasks(tasks) {
    els.subtaskList.innerHTML = "";
    if (tasks.length === 0) {
        els.subtaskList.innerHTML = "<div class='empty-state'>No subtasks</div>";
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

// Poll
setInterval(fetchStatus, 5000);

// --- Journal ---

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
        data.entries.reverse().forEach(entry => {
            const div = document.createElement('div');
            div.className = 'entry-card';

            const time = new Date(entry.timestamp).toLocaleString();
            let content = entry.entry;
            let outcome = '';

            try {
                const p = JSON.parse(content);
                content = p.lessons || content;
                if (p.outcome) outcome = `<div class="outcome-tag" style="color:var(--${getOutcomeColor(p.outcome)})">${p.outcome}</div>`;
                if (p.what_failed) content += `<br><br>FAILED: ${p.what_failed}`;
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

// --- Navigation ---

function switchMainView(viewId) {
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

    document.getElementById(`view-${viewId}`).classList.add('active');
    // Find button that triggered this? Simplified: just visually activate correct one based on viewId
    // (Skipping simpler logic for brevity, assuming user clicks buttons)

    // Hardcoded active state logic for icons
    const btns = document.querySelectorAll('.nav-btn');
    if (viewId === 'dashboard') btns[0].classList.add('active');
    if (viewId === 'graph') {
        btns[1].classList.add('active');
        loadGraph();
    }
    if (viewId === 'settings') btns[2].classList.add('active');
}

function switchRightTab(tabId) {
    document.querySelectorAll('.panel-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    document.getElementById(`tab-${tabId}`).classList.add('active');

    // Find button
    const tabs = document.querySelectorAll('.tab-btn');
    if (tabId === 'mission') tabs[0].classList.add('active');
    if (tabId === 'logs') tabs[1].classList.add('active');
    if (tabId === 'journal') {
        tabs[2].classList.add('active');
        loadJournal();
    }
}

// Expose global
window.switchMainView = switchMainView;
window.switchRightTab = switchRightTab;
window.wipeMemory = () => {
    if (confirm("WIPE MEMORY? (This only clears short/long term facts)")) fetch('/api/memory/clear', { method: 'POST' }).then(() => fetchStatus());
};

window.fullReset = () => {
    if (confirm("FULL SYSTEM RESET? This will wipe ALL memories, goals, and stacks. Logs and Journal will persist.")) {
        fetch('/api/reset', { method: 'POST' }).then(r => r.json()).then(data => {
            alert(data.status);
            fetchStatus();
            loadJournal();
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
                nodes: { shape: 'dot', size: 6, font: { color: '#888', size: 10, face: 'JetBrains Mono' }, borderWidth: 1, color: { background: '#333', border: '#555' } },
                edges: {
                    color: { color: '#444' },
                    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
                    font: { size: 8, align: 'middle', color: '#666' },
                    smooth: { type: 'continuous' }
                },
                physics: { stabilization: false, barnesHut: { gravitationalConstant: -2000, centralGravity: 0.3, springLength: 100 } }
            };
            network = new vis.Network(container, graphData, options);
        } else if (force) {
            // Smart update: add any nodes not already in the dataset
            const existingNodeIds = new Set(graphData.nodes.getIds());
            const newNodes = data.nodes.filter(n => !existingNodeIds.has(n.id));
            if (newNodes.length > 0) graphData.nodes.add(newNodes);

            // Edges from the API have no id field — deduplicate by composite from+to+label key
            const existingEdges = graphData.edges.get();
            const existingEdgeKeys = new Set(existingEdges.map(e => `${e.from}|${e.to}|${e.label}`));
            const newEdges = data.edges.filter(e => !existingEdgeKeys.has(`${e.from}|${e.to}|${e.label}`));
            if (newEdges.length > 0) graphData.edges.add(newEdges);
        }
    });
}

// In fetchStatus, detect memory changes
async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();

        // Stats
        if (els.memoryCount) {
            const count = data.memory_count || 0;
            els.memoryCount.innerText = count;

            // Auto-refresh graph if count increased and view is active
            if (count > lastMemoryCount) {
                if (lastMemoryCount !== -1 && document.getElementById('view-graph').classList.contains('active')) {
                    loadGraph(true);
                }
                lastMemoryCount = count;
            }
        }

        // Goal
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
