import threading
import time
from flask import Flask, jsonify
from shared import ScoringAgentRunner

app = Flask(__name__)

# --- CONFIGURATION & INIT ---
# Host (Web) odlučuje o putanji do podataka i inicijalizaciji
CSV_PATH = "bank-full.csv"
scoring_agent = ScoringAgentRunner(CSV_PATH)

# Thread-safe memorija za rezultate (Simulacija baze podataka u memoriji)
processed_results = []
results_lock = threading.Lock()

def agent_worker_loop():
    """
    Background scheduling (Pravilo #4 i #8).
    Host upravlja 'živahnošću' agenta.
    """
    print("Agent Background Worker pokrenut...")
    
    while True:
        try:
            # SENSE/THINK/ACT se dešava unutar Shared sloja (Step)
            # Web layer ne zna ŠTA agent radi, samo dobija rezultat
            result = scoring_agent.step()
            
            if result:
                # Rule #6: Rezultat je standardizovan DTO (TickResult)
                with results_lock:
                    processed_results.append(result.to_dict())
                
                print(f"Tick Izvršen: ID {result.item_id} | Odluka: {result.decision}")
                
                # Rule #4: Host kontroliše brzinu (1 sekunda pauze)
                time.sleep(1) 
            else:
                # Rule #3: No-work izlaz (nema više poruka/redova)
                print("Nema više podataka. Agent prelazi u stanje mirovanja...")
                time.sleep(10)
                
        except Exception as e:
            print(f"Kritična greška u radu agenta: {e}")
            time.sleep(5)

# --- WEB API (Host Layer) ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "agent": "BankScoringAgent"})

@app.route('/results', methods=['GET'])
def get_results():
    """
    Web layer samo mapira podatke iz memorije u JSON.
    Nema domenske logike ovdje.
    """
    with results_lock:
        return jsonify({
            "total_processed": len(processed_results),
            "recent_actions": processed_results[-15:] # Posljednjih 15 rezultata
        })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Primjer dodatne obrade podataka za UI."""
    with results_lock:
        calls = [r for r in processed_results if "Call" in r['decision']]
        skips = [r for r in processed_results if "Skip" in r['decision']]
        return jsonify({
            "count_high_potential": len(calls),
            "count_low_potential": len(skips)
        })

if __name__ == "__main__":
    # 1. Pokretanje agenta u pozadini (Worker)
    # Daemon=True osigurava da se thread gasi kada se ugasi glavni program
    worker = threading.Thread(target=agent_worker_loop, daemon=True)
    worker.start()

    # 2. Pokretanje Flask servera (Transport/Host)
    # Isključujemo debug=True jer Flask debug mode pokreće threadove duplo
    print("Pokrećem Web Server na http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)