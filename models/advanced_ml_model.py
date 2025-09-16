"""
Advanced ML Model for TruthGuard AI
Based on 99% accuracy fake news detection model
Integrates TF-IDF + Logistic Regression approach
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os
from pathlib import Path

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        SPACY_AVAILABLE = False
        print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy")

try:
    from nltk.corpus import stopwords
    import nltk
    try:
        stop_words = set(stopwords.words('english'))
        NLTK_AVAILABLE = True
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    stop_words = set()

class AdvancedMLModel:
    """
    Advanced Machine Learning model for fake news detection
    Based on TF-IDF vectorization and Logistic Regression
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_path = Path(__file__).parent / "saved_models"
        self.model_path.mkdir(exist_ok=True)
        
    def clean_text_spacy(self, text):
        """Advanced text cleaning using spaCy"""
        if not SPACY_AVAILABLE:
            return self.clean_text_basic(text)
        
        try:
            doc = nlp(str(text).lower())
            cleaned_tokens = [
                token.lemma_ for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
            return " ".join(cleaned_tokens)
        except:
            return self.clean_text_basic(text)
    
    def clean_text_basic(self, text):
        """Basic text cleaning fallback"""
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords if available
        if NLTK_AVAILABLE:
            words = text.split()
            text = ' '.join([word for word in words if word not in stop_words])
        
        return text
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Create cleaned text column
        print("Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text_spacy)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def train_model(self, df, test_size=0.2, max_features=5000):
        """Train the TF-IDF + Logistic Regression model"""
        print("Preprocessing data...")
        df = self.preprocess_data(df.copy())
        
        # Prepare features and labels
        X_text = df['cleaned_text']
        y = df['label']
        
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(X_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return accuracy, classification_report(y_test, y_pred, output_dict=True)
    
    def predict_single(self, text):
        """Predict if a single text is fake or real"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Clean the text
        cleaned_text = self.clean_text_spacy(text)
        
        if not cleaned_text:
            return {
                'prediction': 'uncertain',
                'probability': 0.5,
                'confidence': 'low',
                'reason': 'Text too short or empty after cleaning'
            }
        
        # Vectorize
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probabilities = self.model.predict_proba(text_vector)[0]
        
        # Get confidence
        max_prob = max(probabilities)
        confidence_level = 'high' if max_prob > 0.8 else 'medium' if max_prob > 0.6 else 'low'
        
        return {
            'prediction': 'real' if prediction == 1 else 'fake',
            'probability': float(max_prob),
            'probabilities': {
                'fake': float(probabilities[0]),
                'real': float(probabilities[1])
            },
            'confidence': confidence_level,
            'cleaned_text': cleaned_text
        }
    
    def predict_batch(self, texts):
        """Predict for multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'prediction': 'error',
                    'probability': 0.0,
                    'confidence': 'low',
                    'reason': str(e)
                })
        return results
    
    def save_model(self, filename=None):
        """Save the trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filename is None:
            filename = "advanced_ml_model"
        
        model_file = self.model_path / f"{filename}_model.pkl"
        vectorizer_file = self.model_path / f"{filename}_vectorizer.pkl"
        
        # Save model
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {model_file}")
        print(f"Vectorizer saved to {vectorizer_file}")
    
    def load_model(self, filename=None):
        """Load a pre-trained model and vectorizer"""
        if filename is None:
            filename = "advanced_ml_model"
        
        model_file = self.model_path / f"{filename}_model.pkl"
        vectorizer_file = self.model_path / f"{filename}_vectorizer.pkl"
        
        if not model_file.exists() or not vectorizer_file.exists():
            raise FileNotFoundError(f"Model files not found: {model_file}, {vectorizer_file}")
        
        # Load model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {model_file}")
    
    def get_feature_importance(self, top_n=20):
        """Get most important features for fake vs real classification"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Get top features for fake news (negative coefficients)
        fake_indices = np.argsort(coefficients)[:top_n]
        fake_features = [(feature_names[i], coefficients[i]) for i in fake_indices]
        
        # Get top features for real news (positive coefficients) 
        real_indices = np.argsort(coefficients)[-top_n:][::-1]
        real_features = [(feature_names[i], coefficients[i]) for i in real_indices]
        
        return {
            'fake_indicators': fake_features,
            'real_indicators': real_features
        }


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    dummy_data = {
        'text': [
            "This is a legitimate news article about politics.",
            "BREAKING: SHOCKING truth they don't want you to know!!!",
            "Scientists discover new treatment for common disease.",
            "FAKE NEWS: Government hiding alien contact!!!"
        ],
        'label': [1, 0, 1, 0]  # 1 = real, 0 = fake
    }
    
    df = pd.DataFrame(dummy_data)
    
    # Initialize and train model
    model = AdvancedMLModel()
    accuracy, report = model.train_model(df)
    
    # Test prediction
    test_text = "This is a breaking news story about politics."
    result = model.predict_single(test_text)
    print(f"\nTest prediction: {result}")