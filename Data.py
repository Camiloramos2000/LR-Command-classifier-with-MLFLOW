import unicodedata
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class Data:
    
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.encoder = LabelEncoder()
        self.X = None
        self.y = None


    def load_data(self):
        try:
            # --- 1️⃣ Verify path existence ---
            if not self.csv_path:
                raise ValueError("❌ No CSV file path was provided.")

            # --- 2️⃣ Load dataset ---
            print(f"📂 Loading dataset from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)

            # --- 3️⃣ Verify required columns ---
            required_cols = {"text", "command", "intent"}
            if not required_cols.issubset(df.columns):
                raise KeyError(f"❌ Missing required columns in dataset. Expected: {required_cols}")

            # --- 4️⃣ Create combined label column ---
            df["label"] = df["command"].astype(str) + " " + df["intent"].astype(str)
            df = df.drop(columns=["command", "intent"])

            self.X = df["text"].values
            self.y = df["label"].values

            # --- 5️⃣ Clean text and encode labels ---
            texts_cleaned = [self.clean_text(text) for text in self.X]
            labels_encoded = self.encoder.fit_transform(self.y)

            print("✅ Dataset successfully loaded and processed.")
            return texts_cleaned, labels_encoded

        except FileNotFoundError:
            print(f"❌ CSV file not found at path: {self.csv_path}")
            return None, None

        except pd.errors.EmptyDataError:
            print("⚠️ The CSV file is empty or corrupted.")
            return None, None

        except KeyError as e:
            print(str(e))
            return None, None

        except Exception as e:
            print(f"⚠️ Unexpected error while loading data: {e}")
            return None, None


    def clean_text(self, text):
        try:
            if not isinstance(text, str):
                text = str(text)

            # Lowercase
            text = text.lower()
            # Remove accents
            text = ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
            # Remove symbols, numbers, and punctuation
            text = re.sub(r'[^a-zA-ZñÑ\s]', '', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        
        except Exception as e:
            print(f"⚠️ Error cleaning text: {e}")
            return ""
