import threading
import time
from flask import Flask, jsonify
from shared import ScoringAgentRunner, RetrainAgentRunner, BankClassifier

app = Flask(__name__)

CSV_PATH = "bank-full.csv"

# Shared Brain
shared_classifier = BankClassifier()
if not shared_classifier.load_model():
    # Inicijalno učenje ako nema modela
    import pandas as pd
    initial_df = pd.read_csv(CSV_PATH, sep=';').head(1000)
    shared_classifier.train(initial_df)
    shared_classifier.save_model()

# Instanciranje oba agenta
scoring_agent = ScoringAgentRunner(CSV_PATH, shared_classifier)
retrain_agent = RetrainAgentRunner(shared_classifier, CSV_PATH)

processed_results = []
results_lock = threading.Lock()

def agent_worker_loop():
    while True:
        try:
            # 1. SCORING STEP
            result = scoring_agent.step()
            
            if result:
                with results_lock:
                    processed_results.append(result.to_dict())
                
                # Javi Retrain agentu da je obrađen jedan podatak (Sense update za Retrain)
                retrain_agent.increment_counter()
                
                # 2. RETRAIN STEP (Provjera da li treba učiti)
                learn_result = retrain_agent.step()
                if learn_result:
                    print(f"!!! AGENT JE NAUČIO NEŠTO NOVO: {learn_result.status} !!!")

                time.sleep(0.5) 
            else:
                time.sleep(5)
                
        except Exception as e:
            print(f"Greška: {e}")
            time.sleep(2)

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
    worker = threading.Thread(target=agent_worker_loop, daemon=True)
    worker.start()
    app.run(host='0.0.0.0', port=5000, debug=False)