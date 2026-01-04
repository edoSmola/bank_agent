import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from core import SoftwareAgent, TickResult
import joblib
import os
import datetime

# ==========================================
# 1. DOMAIN LAYER
# Entiteti, Enum-i i čistokrvna pravila
# ==========================================

class BankDecision:
    CALL = "High Potential (Call)"
    SKIP = "Low Potential (Skip)"
    PENDING = "Pending Review"

class BankRules:
    @staticmethod
    def decide(probability, threshold):
        """Domensko pravilo za odluku (Rule #8)"""
        if probability > threshold:
            return BankDecision.CALL
        return BankDecision.SKIP

# ==========================================
# 2. ML LAYER (Infrastructure/Service)
# "Crna kutija" za model
# ==========================================

class BankClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.encoder = LabelEncoder()
        self.is_trained = False
        self.version = "1.0.0"

    def train(self, df):
        # Priprema podataka (Sve kolone ili specifične)
        # Za demo koristimo age i duration
        X = df[['age', 'duration']]
        y = self.encoder.fit_transform(df['y'])
        self.model.fit(X, y)
        
        # Update version on every train
        self.version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.is_trained = True
        print(f"--- LEARN: Model treniran. Nova verzija: {self.version} ---")

    def predict(self, row_dict):
        if not self.is_trained: return 0.0
        X_single = pd.DataFrame([row_dict])[['age', 'duration']]
        return self.model.predict_proba(X_single)[0][1]
    
    def save_model(self, filename="bank_model.joblib"):
        joblib.dump({"model": self.model, "version": self.version, "encoder": self.encoder}, filename)

    def load_model(self, filename="bank_model.joblib"):
        if os.path.exists(filename):
            data = joblib.load(filename)
            self.model = data["model"]
            self.version = data["version"]
            self.encoder = data.get("encoder", LabelEncoder())
            self.is_trained = True
            return True
        return False

class ScoringAgentRunner(SoftwareAgent):
    def __init__(self, csv_path, classifier):
        self.dataset = pd.read_csv(csv_path, sep=',')
        self.current_row = 0
        self.classifier = classifier
        self.threshold = 0.6

    def step(self):
        if self.current_row >= len(self.dataset):
            return None 

        row = self.dataset.iloc[self.current_row]
        row_dict = row.to_dict()
        current_id = self.current_row
        self.current_row += 1

        # THINK
        p_yes = self.classifier.predict(row_dict)
        decision = BankRules.decide(p_yes, self.threshold)

        # ACT
        return TickResult(
            item_id=int(current_id),
            probability=float(round(p_yes, 4)),
            decision=decision,
            status=f"Processed (Model v{self.classifier.version})"
        )
    
    def predict_single(self, data_dict):
        # THINK
        p_yes = self.classifier.predict(data_dict)
        decision = BankRules.decide(p_yes, self.threshold)

        # ACT (Return result for the Web layer to display)
        return TickResult(
            item_id="MANUAL_INPUT",
            probability=float(round(p_yes, 4)),
            decision=decision,
            status=f"Predicted via Web (Model v{self.classifier.version})"
        )

class RetrainAgentRunner(SoftwareAgent):
    """
    IMPLEMENTACIJA LEARN DIJELA
    """
    def __init__(self, classifier, csv_path):
        self.classifier = classifier
        self.csv_path = csv_path
        self.processed_since_last_train = 0
        self.retrain_interval = 20 # Svakih 20 predviđanja radimo "re-learning"

    def increment_counter(self):
        self.processed_since_last_train += 1

    def step(self):
        # --- SENSE ---
        # Provjeri da li je ispunjen uslov za učenje (npr. skupili smo dovoljno novih podataka)
        if self.processed_since_last_train < self.retrain_interval:
            return None # Nema posla (Pravilo #3)

        # --- THINK ---
        print(f"RetrainAgent: Detektovano {self.processed_since_last_train} novih stavki. Pokrećem učenje...")

        # --- ACT / LEARN ---
        # U realnosti bi ovdje učitavali samo NOVE podatke iz baze
        full_df = pd.read_csv(self.csv_path, sep=',')
        
        # Simuliramo učenje na "svježim" podacima (uzimamo nasumični uzorak kao nove gold labele)
        fresh_data = full_df.sample(n=1000) 
        
        self.classifier.train(fresh_data)
        self.classifier.save_model()
        
        # Resetujemo stanje agenta
        self.processed_since_last_train = 0
        
        return TickResult(
            item_id=0,
            probability=1.0,
            decision="MODEL_UPDATED",
            status=f"New version: {self.classifier.version}"
        )