document.getElementById('startBtn').addEventListener('click', () => {
    fetch('/start').then(() => document.getElementById('agentStatus').innerText = "Status: Running");
});

document.getElementById('stopBtn').addEventListener('click', () => {
    fetch('/stop').then(() => document.getElementById('agentStatus').innerText = "Status: Stopped");
});

function updateUI() {
    fetch('/results')
        .then(response => response.json())
        .then(data => {
            document.getElementById('stats').innerText = "Total Processed: " + data.total_processed;
            const log = document.getElementById('log');
            
            // Render latest results
            log.innerHTML = data.recent_actions.reverse().map(a => 
                `<div class="log-entry">
                    <span style="color: #9b59b6">[ID: ${a.item_id}]</span> 
                    Prob: ${a.probability} -> 
                    <b style="color: ${a.decision.includes('Call') ? '#2ecc71' : '#e67e22'}">${a.decision}</b>
                </div>`
            ).join('');
        });
}

// Refresh UI every second
setInterval(updateUI, 1000);