import os
import json
import threading
import uuid
import pandas as pd
from dataclasses import dataclass, asdict
from core import TickResult
from shared import BankClassifier


class BankDecision:
    CALL = "High Potential (Call)"
    SKIP = "Low Potential (Skip)"
    PENDING = "Pending Review"


class BankRules:
    @staticmethod
    def decide(probability, threshold):
        if probability > threshold:
            return BankDecision.CALL
        return BankDecision.SKIP


class ModelVersion:
    def __init__(self, version_id, path, timestamp=None, active=False):
        self.version_id = version_id
        self.path = path
        self.timestamp = timestamp
        self.active = active

    def to_dict(self):
        return {
            "version_id": self.version_id,
            "path": self.path,
            "timestamp": self.timestamp,
            "active": self.active,
        }


@dataclass(frozen=True)
class SystemSettings:
    retrain_interval: int = 20
    threshold: float = 0.6

    @classmethod
    def load(cls, path: str = 'settings.json', create_if_missing: bool = True):
        """Load settings from JSON file. If missing, create with defaults when allowed."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
            # Validate keys and provide defaults for missing ones
            retrain = int(data.get('retrain_interval', cls.retrain_interval))
            threshold = float(data.get('threshold', cls.threshold))
            return cls(retrain_interval=retrain, threshold=threshold)

        if not create_if_missing:
            raise FileNotFoundError(path)

        inst = cls()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(asdict(inst), f, indent=2)
        except Exception:
            # If persisting fails, still return in-memory defaults
            pass
        return inst


class QueueService:
    """Lightweight queue backed by CSV and a small state file for idempotency."""
    def __init__(self, csv_path, state_path='queue_state.json'):
        self.csv_path = csv_path
        self.state_path = state_path
        self.dataset = pd.read_csv(csv_path, sep=',')
        # state: {"next_index": int}
        if os.path.exists(state_path):
            with open(state_path, 'r', encoding='utf-8') as f:
                self.state = json.load(f)
        else:
            self.state = {"next_index": 0}
            self._persist()

    def _persist(self):
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(self.state, f)

    def dequeue_next(self):
        ni = self.state.get('next_index', 0)
        if ni >= len(self.dataset):
            return None
        row = self.dataset.iloc[ni]
        row_dict = row.to_dict()
        return int(ni), row_dict

    def mark_processed(self):
        self.state['next_index'] = self.state.get('next_index', 0) + 1
        self._persist()


class ScoringService:
    def __init__(self, classifier: BankClassifier, queue_service: QueueService, predictions_path='predictions.json', settings: SystemSettings = None):
        self.classifier = classifier
        self.queue = queue_service
        self.predictions_path = predictions_path
        if settings is None:
            raise ValueError("SystemSettings must be provided to ScoringService")
        self.settings = settings
        self.review_service = None
        # ensure predictions file exists
        if not os.path.exists(self.predictions_path):
            with open(self.predictions_path, 'w', encoding='utf-8') as f:
                json.dump([], f)



    def predict_single(self, data_dict):
        p_yes = self.classifier.predict(data_dict)
        decision = BankRules.decide(p_yes, self.settings.threshold)
        # Return a TickResult but do NOT persist/advance the queue for manual predictions
        return TickResult(
            item_id="MANUAL_INPUT",
            probability=float(round(p_yes, 4)),
            decision=decision,
            status=f"Predicted via Web (Model v{self.classifier.version})"
        )


class ReviewService:
    """Collects gold-labeled experiences for learning.

    Experiences are appended to a CSV with the same feature columns
    plus the target `y`. TrainingService will consume these.
    """
    def __init__(self, experiences_path='experiences.csv', feature_keys=None):
        self.experiences_path = experiences_path
        self.feature_keys = feature_keys or ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
        # ensure file exists with header
        if not os.path.exists(self.experiences_path):
            with open(self.experiences_path, 'w', encoding='utf-8') as f:
                header = ','.join(self.feature_keys + ['y'])
                f.write(header + '\n')

    def record_experience(self, item_id, row_dict):
        # write only the expected feature columns and y if present
        values = []
        for k in self.feature_keys:
            v = row_dict.get(k, '')
            # escape commas/newlines simplistically
            values.append(str(v).replace('\n', ' ').replace('\r', ' ').replace(',', ' '))
        # true label
        y = row_dict.get('y', '')
        values.append(str(y))
        line = ','.join(values) + '\n'
        with open(self.experiences_path, 'a', encoding='utf-8') as f:
            f.write(line)


class TrainingService:
    """Non-blocking training service. Starts training jobs in background threads and tracks versions."""
    def __init__(self, classifier: BankClassifier, csv_path, models_state='models.json', models_dir='models'):
        self.classifier = classifier
        self.csv_path = csv_path
        self.jobs = {}  # job_id -> status
        self.models_state = models_state
        self.models_dir = models_dir
        self.experiences_path = 'experiences.csv'
        os.makedirs(self.models_dir, exist_ok=True)
        if os.path.exists(self.models_state):
            with open(self.models_state, 'r', encoding='utf-8') as f:
                self.models = json.load(f)
        else:
            self.models = []
            self._persist_models()

        self._lock = threading.Lock()

    def _persist_models(self):
        with open(self.models_state, 'w', encoding='utf-8') as f:
            json.dump(self.models, f)

    def should_retrain(self, processed_since_last_train, interval):
        return processed_since_last_train >= interval

    def start_training_async(self, sample_n=1000):
        # Only start a training job if we have collected experiences (gold labels)
        try:
            if not os.path.exists(self.experiences_path):
                return None
            exp_df = pd.read_csv(self.experiences_path, sep=',')
            if exp_df.empty:
                return None
        except Exception:
            return None

        job_id = str(uuid.uuid4())
        self.jobs[job_id] = 'running'
        thread = threading.Thread(target=self._train_worker, args=(job_id, sample_n), daemon=True)
        thread.start()
        return job_id

    def get_job_status(self, job_id):
        return self.jobs.get(job_id)

    def _train_worker(self, job_id, sample_n):
        try:
            # Train only on collected experiences (gold labels) with NO SAMPLING.
            if not os.path.exists(self.experiences_path):
                self.jobs[job_id] = 'error: no experiences'
                return
            exp_df = pd.read_csv(self.experiences_path, sep=',')
            if exp_df.empty:
                self.jobs[job_id] = 'error: no experiences'
                return

            # Train on the entire experiences dataset (no sampling)
            self.classifier.train(exp_df)
            # save model file
            version = self.classifier.version
            model_path = os.path.join(self.models_dir, f'bank_model_{version}.joblib')
            self.classifier.save_model(model_path)

            # register ModelVersion
            mv = {
                'version_id': version,
                'path': model_path,
                'timestamp': self.classifier.version,
                'active': True
            }
            with self._lock:
                # deactivate others
                for m in self.models:
                    m['active'] = False
                self.models.append(mv)
                self._persist_models()


            self.jobs[job_id] = 'done'
        except Exception as e:
            self.jobs[job_id] = f'error: {e}'


class ScoringAgentRunner:
    """Runner implements Sense->Think->Act for scoring using services."""
    def __init__(self, queue_service, scoring_service):
        self.queue = queue_service
        self.scoring = scoring_service

    # --- SENSE ---
    def sense(self):
        """Read a single queued item from the queue. Returns (item_id, row) or None."""
        item = self.queue.dequeue_next()
        if item is None:
            return None
        return item

    # --- THINK ---
    def think(self, row):
        """Compute model probability and decision based on rules/settings."""
        p_yes = self.scoring.classifier.predict(row)
        decision = BankRules.decide(p_yes, self.scoring.settings.threshold)
        return p_yes, decision

    # --- ACT ---
    def act(self, item_id, row, probability, decision):
        """Persist prediction and update queue state. Returns TickResult."""
        # --- MAKE (pure) ---
        result = TickResult(
            item_id=int(item_id),
            probability=float(round(probability, 4)),
            decision=decision,
            status=f"Processed (Model v{self.scoring.classifier.version})"
        )

        # --- PERSIST (I/O) ---
        try:
            with open(self.scoring.predictions_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.append(result.to_dict())
                f.seek(0)
                json.dump(data, f)
                f.truncate()
        except Exception:
            # If persistence fails, still return the result so the agent can continue.
            pass

        # Record experience if available
        try:
            if self.scoring.review_service is not None and row is not None and 'y' in row:
                try:
                    self.scoring.review_service.record_experience(result.item_id, row)
                except Exception:
                    pass
        except Exception:
            pass

        # Advance queue progress
        try:
            self.scoring.queue.mark_processed()
        except Exception:
            pass

        return result

    def step(self):
        # SENSE
        sensed = self.sense()
        if sensed is None:
            return None
        item_id, row = sensed

        # THINK
        probability, decision = self.think(row)

        # ACT
        result = self.act(item_id, row, probability, decision)
        return result


class RetrainAgentRunner:
    """Runner decides whether to schedule retraining and triggers TrainingService."""
    def __init__(self, training_service, settings):
        self.training = training_service
        self.settings = settings
        self.processed_since_last_train = 0
        self._last_job = None

    def increment_counter(self):
        self.processed_since_last_train += 1

    def step(self):
        # SENSE
        processed = self.processed_since_last_train

        # THINK
        should = self.training.should_retrain(processed, self.settings.retrain_interval)
        if not should:
            return None

        # ACT
        job_id = self.training.start_training_async()
        if job_id is None:
            return None

        self._last_job = job_id
        # LEARN: reset counter as part of learning step
        self.processed_since_last_train = 0

        return TickResult(item_id=0, probability=1.0, decision="MODEL_TRAIN_STARTED", status=f"Training job {job_id} started")
