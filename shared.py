import pandas as pd
import threading
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import datetime


class BankClassifier:
    def __init__(self):
        self.pipeline = None
        self.encoder = LabelEncoder()
        self.is_trained = False
        self.version = "1.0.0"
        self._lock = threading.RLock()

    def train(self, df):
        with self._lock:
            feature_keys = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
            cat_cols = ['job', 'marital', 'default', 'housing', 'loan']
            edu_order = [
                'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                'high.school', 'professional.course', 'university.degree'
            ]

            X = df[feature_keys].copy()
            y = self.encoder.fit_transform(df['y'])

            edu_encoder = OrdinalEncoder(categories=[edu_order], handle_unknown='use_encoded_value', unknown_value=-1)

            preprocessor = ColumnTransformer([
                ('edu', edu_encoder, ['education']),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ], remainder='passthrough')

            self.pipeline = Pipeline([
                ('pre', preprocessor),
                ('clf', RandomForestClassifier(n_estimators=100))
            ])

            self.pipeline.fit(X, y)

            self.version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.is_trained = True
            print(f"--- LEARN: Model treniran. Nova verzija: {self.version} ---")

    def predict(self, row_dict):
        if not self.is_trained:
            return 0.0

        feature_keys = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
        row = {k: row_dict.get(k, None) for k in feature_keys}
        X_single_df = pd.DataFrame([row])

        with self._lock:
            # Try to get probability estimates
            try:
                probas = self.pipeline.predict_proba(X_single_df)
            except Exception:
                # If predict_proba unavailable, fall back to predict
                try:
                    pred = self.pipeline.predict(X_single_df)[0]
                    return float(pred)
                except Exception:
                    return 0.0

            # Determine encoder mapping for 'yes'
            try:
                encoded_yes = self.encoder.transform(['yes'])[0]
            except Exception:
                encoded_yes = 1

            # Try to find classes from the classifier step
            try:
                clf = self.pipeline.named_steps.get('clf')
                classes = list(getattr(clf, 'classes_', []))
            except Exception:
                classes = list(range(probas.shape[1]))

            # Handle single-class models
            if probas.shape[1] == 1:
                if len(classes) == 1 and classes[0] == encoded_yes:
                    return 1.0
                return 0.0

            # Multiple classes: map to encoded 'yes' if present
            if encoded_yes in classes:
                idx = classes.index(encoded_yes)
                return float(probas[0][idx])

            # Fallback to last column (convention: positive class)
            return float(probas[0][-1])

    def save_model(self, filename="data/bank_model.joblib"):
        with self._lock:
            # ensure target directory exists
            dirpath = os.path.dirname(filename)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            joblib.dump({"pipeline": self.pipeline, "version": self.version, "encoder": self.encoder}, filename)

    def load_model(self, filename="data/bank_model.joblib"):
        if os.path.exists(filename):
            with self._lock:
                data = joblib.load(filename)
                self.pipeline = data.get("pipeline")
                self.version = data.get("version", self.version)
                self.encoder = data.get("encoder", LabelEncoder())
                self.is_trained = True
            return True
        return False