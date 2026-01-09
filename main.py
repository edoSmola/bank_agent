import threading
import time
import os
import json
import pandas as pd
from collections import deque
from flask import Flask, jsonify, render_template, send_from_directory, request
from shared import BankClassifier
from agents import QueueService, ScoringService, TrainingService, ScoringAgentRunner, RetrainAgentRunner, SystemSettings, ReviewService

# --- FLASK CONFIGURATION (Flat Folder Mode) ---
# template_folder='.' allows index.html to be in the same directory
app = Flask(__name__, template_folder='.', static_folder='.')

# --- AGENT STATE & SHARED DATA ---
CSV_PATH = "bank-full.csv"
is_running = False
processed_results = deque(maxlen=30)
results_lock = threading.Lock()

# --- AGENT CORE INITIALIZATION ---
shared_classifier = BankClassifier()

# Initial setup check (ensure we have a model available)
if not shared_classifier.load_model():
    print("--- Initializing: No model found. Training first version... ---")
    try:
        initial_df = pd.read_csv(CSV_PATH, sep=',').head(1000)
        shared_classifier.train(initial_df)
        shared_classifier.save_model()

        # Persist the initial training rows as labeled experiences so the
        # agent does not re-process them at runtime. Keep only the feature
        # columns used by the classifier plus the target `y`.
        experiences_path = 'experiences.csv'
        feature_keys = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
        cols_to_write = [c for c in feature_keys + ['y'] if c in initial_df.columns]

        try:
            if cols_to_write:
                # If experiences file already exists and appears to already
                # contain at least as many rows as initial_df, skip appending
                if os.path.exists(experiences_path):
                    try:
                        existing = pd.read_csv(experiences_path, sep=',')
                        if len(existing) >= len(initial_df):
                            appended = 0
                        else:
                            # append only missing count (best-effort)
                            missing_n = len(initial_df) - len(existing)
                            to_append = initial_df.loc[:missing_n-1, cols_to_write]
                            to_append.to_csv(experiences_path, mode='a', header=False, index=False)
                            appended = len(to_append)
                    except Exception:
                        # If read fails for any reason, fallback to appending nothing
                        appended = 0
                else:
                    # write full file with header
                    initial_df.loc[:, cols_to_write].to_csv(experiences_path, mode='w', header=True, index=False)
                    appended = len(initial_df)
            else:
                appended = 0
        except Exception as e:
            print(f"Warning: failed to persist experiences: {e}")
            appended = 0

        # Advance queue state so the initial rows are skipped during processing
        # by setting next_index to at least the number of initial rows used for training.
        state_path = 'queue_state.json'
        desired_next = len(initial_df)
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as sf:
                    try:
                        state = json.load(sf)
                    except Exception:
                        state = {"next_index": 0}
            else:
                state = {"next_index": 0}

            # Do not move backwards if a later state already progressed further
            state['next_index'] = max(state.get('next_index', 0), desired_next)
            with open(state_path, 'w', encoding='utf-8') as sf:
                json.dump(state, sf)
            next_index = state['next_index']
        except Exception as e:
            print(f"Warning: failed to write {state_path}: {e}")
            next_index = desired_next

        print(f"Initial training completed on {len(initial_df)} rows. Experiences appended: {appended}. queue next_index={next_index}.")
    except Exception as e:
        print(f"Initial training failed: {e}. Check if {CSV_PATH} exists.")

# Application services
queue_service = QueueService(CSV_PATH)
# Load settings from file (creates settings.json with defaults if missing)
settings = SystemSettings.load()
scoring_service = ScoringService(shared_classifier, queue_service, settings=settings)
# Review service collects gold-labeled experiences (used by TrainingService)
review_service = ReviewService()
scoring_service.review_service = review_service
training_service = TrainingService(shared_classifier, CSV_PATH)

# Instantiate Runners using services
scoring_agent = ScoringAgentRunner(queue_service, scoring_service)
retrain_agent = RetrainAgentRunner(training_service, settings)

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
            "recent_actions": list(processed_results) # Send last 30 (deque already capped)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend JSON
        user_data = request.json 
        
        # Validate minimal input (expecting the feature set used by the model)
        required = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
        if not user_data or any(k not in user_data for k in required):
            return jsonify({"error": f"Missing required fields. Required: {required}"}), 400

        # Delegate to ScoringService (The 'Brain')
        result = scoring_service.predict_single(user_data)
        
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