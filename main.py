import threading
import time
import os
import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory, request
from shared import ScoringAgentRunner, RetrainAgentRunner, BankClassifier

# --- FLASK CONFIGURATION (Flat Folder Mode) ---
# template_folder='.' allows index.html to be in the same directory
app = Flask(__name__, template_folder='.', static_folder='.')

# --- AGENT STATE & SHARED DATA ---
CSV_PATH = "bank-full.csv"
is_running = False
processed_results = []
results_lock = threading.Lock()

# --- AGENT CORE INITIALIZATION ---
shared_classifier = BankClassifier()

# Initial setup check
if not shared_classifier.load_model():
    print("--- Initializing: No model found. Training first version... ---")
    try:
        # Note: Bank dataset usually uses ';' separator
        initial_df = pd.read_csv(CSV_PATH, sep=';').head(1000)
        shared_classifier.train(initial_df)
        shared_classifier.save_model()
    except Exception as e:
        print(f"Initial training failed: {e}. Check if {CSV_PATH} exists.")

# Instantiate Runners
scoring_agent = ScoringAgentRunner(CSV_PATH, shared_classifier)
retrain_agent = RetrainAgentRunner(shared_classifier, CSV_PATH)

# --- THE AGENT WORKER LOOP (Sense -> Think -> Act) ---
def agent_worker_loop():
    global is_running
    print("Agent Worker Thread: Started and waiting for signal...")
    
    while True:
        if is_running:
            try:
                # 1. SCORING STEP (Sense & Think & Act)
                result = scoring_agent.step()
                
                if result:
                    # Update Web State
                    with results_lock:
                        processed_results.append(result.to_dict())
                    
                    # 2. RETRAIN LOGIC (Sense Update)
                    retrain_agent.increment_counter()
                    
                    # 3. LEARN STEP (Check if retrain is needed)
                    learn_result = retrain_agent.step()
                    if learn_result:
                        print(f"!!! LEARN COMPONENT: {learn_result.status} !!!")

                    # Control the speed of simulation (0.5 seconds per tick)
                    time.sleep(0.5) 
                else:
                    # Dataset finished
                    print("Agent: Finished processing all rows.")
                    is_running = False
            except Exception as e:
                print(f"Error in agent tick: {e}")
                time.sleep(2)
        else:
            # Idle wait while agent is 'Stopped'
            time.sleep(1)

# --- WEB ROUTES (Host Layer) ---

@app.route('/')
def index():
    """Serves the main UI."""
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serves CSS and JS files from the same folder."""
    return send_from_directory('.', filename)

@app.route('/start')
def start_agent():
    global is_running
    is_running = True
    return jsonify({"status": "Agent started"})

@app.route('/stop')
def stop_agent():
    global is_running
    is_running = False
    return jsonify({"status": "Agent stopped"})

@app.route('/results', methods=['GET'])
def get_results():
    """Web layer only reads and returns the results list."""
    with results_lock:
        return jsonify({
            "total_processed": len(processed_results),
            "recent_actions": processed_results[-15:] # Send last 15 for the UI log
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend JSON
        user_data = request.json 
        
        # Validate minimal input (expecting 'age' and 'duration')
        if not user_data or 'age' not in user_data or 'duration' not in user_data:
            return jsonify({"error": "Missing age or duration"}), 400

        # Delegate to Agent (The 'Brain')
        result = scoring_agent.predict_single(user_data)
        
        # Optionally log this to our history
        with results_lock:
            processed_results.append(result.to_dict())

        return jsonify(result.to_dict())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"is_running": is_running, "data_loaded": len(processed_results)})

# --- START THE SYSTEM ---
if __name__ == "__main__":
    # Start the Agent in its own thread
    worker = threading.Thread(target=agent_worker_loop, daemon=True)
    worker.start()
    
    # Start the Web Server
    print("Starting Web Host at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)