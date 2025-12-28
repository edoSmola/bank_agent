import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from core import SoftwareAgent, TickResult
import joblib
import os

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
        # Priprema podataka
        X = df[['age', 'duration']]
        y = self.encoder.fit_transform(df['y'])
        self.model.fit(X, y)
        self.is_trained = True
        print(f"Model v{self.version} treniran.")

    def predict(self, row_dict):
        if not self.is_trained: return 0.0
        X_single = pd.DataFrame([row_dict])[['age', 'duration']]
        return self.model.predict_proba(X_single)[0][1]
    
    def save_model(self, filename="bank_model.joblib"):
        """Saves the brain to disk (Infrastructure logic)"""
        joblib.dump({"model": self.model, "version": self.version}, filename)
        print(f"Model spremljen na lokaciju: {filename}")

    def load_model(self, filename="bank_model.joblib"):
        """Loads the brain from disk"""
        if os.path.exists(filename):
            data = joblib.load(filename)
            self.model = data["model"]
            self.version = data["version"]
            self.is_trained = True
            print(f"Model v{self.version} uspješno učitan.")
            return True
        return False

# ==========================================
# 3. APPLICATION LAYER (Runners)
# Ovdje se spaja Sense -> Think -> Act
# ==========================================

class ScoringAgentRunner(SoftwareAgent):
    def __init__(self, csv_path):
        # SENSE: Inicijalizacija izvora podataka
        self.dataset = pd.read_csv(csv_path, sep=',')
        self.current_row = 0
        
        # Servisi
        self.classifier = BankClassifier()
        if not self.classifier.load_model():
            print("Nema spremljenog modela. Treniram novi...")
            self.classifier.train(self.dataset.head(500))
            self.classifier.save_model() # Odmah spremi početno znanje
        
        # Postavke (Infrastruktura/Domain)
        self.threshold = 0.6

    def step(self):
        """Jedan Tick/Step agenta (Pravilo #1)"""
        
        # --- SENSE ---
        # Uzmi stanje svijeta (jedan red iz CSV/DB)
        if self.current_row >= len(self.dataset):
            return None # Pravilo #3: No-work izlaz

        row = self.dataset.iloc[self.current_row]
        row_dict = row.to_dict()
        current_id = self.current_row
        self.current_row += 1

        # --- THINK ---
        # 1. Izračunaj vjerovatnoću (ML)
        p_yes = self.classifier.predict(row_dict)
        # 2. Donesi domensku odluku (Domain Rules)
        decision = BankRules.decide(p_yes, self.threshold)

        # --- ACT ---
        # Vrati standardizovani rezultat (Pravilo #6)
        # U realnom sistemu ovdje bi išao i self.db.save(...)
        return TickResult(
            item_id=int(current_id),
            probability=float(round(p_yes, 4)),
            decision=decision,
            status="Processed"
        )

class RetrainAgentRunner(SoftwareAgent):
    """
    Dodatni agent za 'LEARN' dio (Pravilo #8b)
    """
    def __init__(self, classifier, main_dataset):
        self.classifier = classifier
        self.dataset = main_dataset
        self.new_gold_labels = 0
        self.retrain_threshold = 50 # Svakih 50 novih labela

    def step(self):
        # SENSE: Provjeri brojače ili postavke
        should_retrain = self.new_gold_labels >= self.retrain_threshold
        
        # THINK: Da li je potrebno učenje?
        if not should_retrain:
            return None
            
        # ACT/LEARN: Pokreni trening
        print("RetrainAgent: Pokrećem učenje novog modela...")
        self.classifier.train(self.dataset.sample(600))
        self.new_gold_labels = 0
        
        return TickResult(0, 0, "Model Retrained", "Success")