// Batch Controls
document.getElementById('startBtn').addEventListener('click', () => {
    fetch('/start').then(() => document.getElementById('agentStatus').innerText = "Status: Running");
});

document.getElementById('stopBtn').addEventListener('click', () => {
    fetch('/stop').then(() => document.getElementById('agentStatus').innerText = "Status: Stopped");
});

// Manual Prediction Control
document.getElementById('predictBtn').addEventListener('click', async () => {
    const data = {
        age: parseInt(document.getElementById('ageInput').value),
        job: document.getElementById('jobInput').value,
        marital: document.getElementById('maritalInput').value,
        education: document.getElementById('educationInput').value,
        default: document.getElementById('defaultInput').value,
        housing: document.getElementById('housingInput').value,
        loan: document.getElementById('loanInput').value
    };

    const display = document.getElementById('manualResult');
    display.innerText = "Thinking...";

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        display.innerHTML = `
            <div>Decision: <span style="color:blue">${result.decision}</span></div>
            <div>Probability: ${(result.probability * 100).toFixed(2)}%</div>
        `;
        
        updateUI(); // Refresh the log immediately
    } catch (error) {
        display.innerText = "Error contacting agent.";
    }
});

function updateUI() {
    fetch('/results')
        .then(response => response.json())
        .then(data => {
            document.getElementById('stats').innerText = "Total Processed: " + data.total_processed;
            const log = document.getElementById('log');
            log.innerHTML = data.recent_actions.reverse().map(a => 
                `<div class="log-entry">
                    <span style="color: #9b59b6">[${a.item_id}]</span> 
                    Prob: ${a.probability} -> <b>${a.decision}</b>
                </div>`
            ).join('');
        });
}

setInterval(updateUI, 2000);
updateUI();