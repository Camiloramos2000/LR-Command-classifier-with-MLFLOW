from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Transformer import SentenceTransformerVectorizer
from Data import Data
import time


# ==========================================================
# 🧠 MAIN CLASS: Logistic Regression Model
# ==========================================================
class ModelLR:
    def __init__(self, dataset_path="commands_dataset.csv", parameters=None):
        self.name = "LogisticRegression"
        self.model = LogisticRegression(**parameters)
        self.dataset_path = dataset_path
        self.data = Data(self.dataset_path)
        self.pipeline = None
        self.parameters = parameters

    # ------------------------------------------------------
    # ⚙️ PIPELINE BUILDING
    # ------------------------------------------------------
    def build_pipeline(self):
        print("\n🔧 Building the model pipeline...\n")
        vectorizer = SentenceTransformerVectorizer()
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', self.model),
        ])
        print("✅ Pipeline successfully created.\n")

    # ------------------------------------------------------
    # 🏋️ MODEL TRAINING
    # ------------------------------------------------------
    def train(self):
        print("\n" + "=" * 80)
        print("🏋️  MODEL TRAINING STARTED")
        print("=" * 80 + "\n")

        time_start = time.time()

        # Load and process data
        texts_cleaned, labels_encoded = self.data.load_data()
        if texts_cleaned is None or labels_encoded is None:
            print("❌ Failed to load or process the dataset. Training aborted.")
            return

        # Dataset split
        X_train, X_test, y_train, y_test = train_test_split(
            texts_cleaned, labels_encoded, test_size=0.2, random_state=42
        )

        # Build and train the pipeline
        self.build_pipeline()
        print("🚀 Training Logistic Regression model...\n")
        self.pipeline.fit(X_train, y_train)
        print("✅ Model successfully trained.\n")

        # Evaluation
        print("🔍 Evaluating the model on the test set...")
        y_pred = self.pipeline.predict(X_test)
        time_end = time.time()
        elapsed_time = time_end - time_start
        print("✅ Evaluation completed.\n")

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Console results
        print("-" * 80)
        print("📊 TRAINING RESULTS")
        print("-" * 80)
        print(f"⏱️  Training time: {elapsed_time:.4f} seconds")
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"🧩 Precision (weighted): {precision:.4f}")
        print(f"🔁 Recall (weighted): {recall:.4f}")
        print(f"🏆 F1 Score (weighted): {f1:.4f}")
        print("-" * 80 + "\n")

        return self.pipeline, accuracy, precision, recall, f1, elapsed_time

    # ------------------------------------------------------
    # 🔮 MODEL TEST / PREDICTION
    # ------------------------------------------------------
    def test(self, text):
        print("\n" + "=" * 80)
        print("🔮  MODEL TEST STARTED")
        print("=" * 80 + "\n")

        time_start = time.time()

        # Validations
        if self.pipeline is None:
            print("❌ The model has not been trained yet. Please train it before testing.\n")
            return None

        if not isinstance(text, str) or not text.strip():
            print("❌ Invalid input. Please provide a valid text.\n")
            return None

        print("🧠 Processing and making prediction...\n")

        # Prediction
        prediction = self.pipeline.predict([text])
        prediction_decoded = self.data.encoder.inverse_transform(prediction)
        probabilities = self.pipeline.predict_proba([text])
        prob_max = probabilities.max()

        time_end = time.time()
        elapsed_time = time_end - time_start

        # Results
        print("-" * 80)
        print("📊 PREDICTION RESULTS")
        print("-" * 80)
        print(f"🗣️  Input text: {text}")
        print(f"🧪 Predicted class: {prediction_decoded[0]}")
        print(f"📈 Maximum probability: {prob_max:.4f}")
        print(f"⏱️  Inference time: {elapsed_time:.4f} seconds")
        print("-" * 80 + "\n")

        return prediction_decoded, prob_max, elapsed_time
