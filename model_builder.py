# model_builder.py - Fixed and Optimized
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_DIR = './models'
DATA_PATH = './data/Final_DataSet.xlsx'  # Update this path

class TextEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                              padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:,0,:].cpu().numpy()

def load_and_preprocess_data():
    try:
        if DATA_PATH.endswith('.xlsx'):
            df = pd.read_excel(DATA_PATH, engine='openpyxl')
        else:
            df = pd.read_csv(DATA_PATH)
        
        # Ensure required columns exist
        required_cols = ['Company', 'Project Type', 'Data Snippet (regarding wellness of the project)']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from dataset")
        
        # Generate text embeddings
        embedder = TextEmbedder()
        print("Generating text embeddings...")
        embeddings = []
        for text in tqdm(df['Data Snippet (regarding wellness of the project)'].astype(str)):
            embeddings.append(embedder.embed_text(text))
        embeddings = np.vstack(embeddings)
        
        # Create embedding columns more efficiently
        embedding_cols = {f'embed_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
        df = pd.concat([df, pd.DataFrame(embedding_cols)], axis=1)
        
        # Ensure score columns exist
        score_cols = [
            'Sustainability Score',
            'Community Engagement Score',
            'Ethical Business Score', 
            'Public Engagement Score',
            'Total Impact Score'
        ]
        
        for col in score_cols:
            if col not in df.columns:
                raise ValueError(f"Score column '{col}' missing from dataset")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_models(df):
    if df is None:
        print("No data available for training")
        return None
    
    # Prepare features and targets
    categorical_features = ['Company', 'Project Type']
    numeric_features = [col for col in df.columns if col.startswith('embed_')]
    target_cols = [
        'Sustainability Score',
        'Community Engagement Score',
        'Ethical Business Score',
        'Public Engagement Score',
        'Total Impact Score'
    ]
    
    # Create preprocessor - compatible with older scikit-learn versions
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])
    
    models = {}
    for target in target_cols:
        try:
            print(f"\nTraining model for {target}...")
            
            # Split data
            X = df[categorical_features + numeric_features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Create and train pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1))
            ])
            
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            print(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
            # Save model
            models[target] = pipeline
            os.makedirs(MODEL_DIR, exist_ok=True)
            filename = target.replace(' ', '_').lower() + '_model.pkl'
            joblib.dump(pipeline, os.path.join(MODEL_DIR, filename))
            print(f"Saved model for {target}")
            
        except Exception as e:
            print(f"Error training {target} model: {str(e)}")
    
    return models

if __name__ == '__main__':
    print("Starting model training process...")
    data = load_and_preprocess_data()
    if data is not None:
        print(f"\nData loaded with {len(data)} records")
        print("Columns:", data.columns.tolist())
        
        models = train_models(data)
        if models:
            print(f"\nSuccessfully trained {len(models)} models")
        else:
            print("\nModel training failed")
    else:
        print("\nCould not load data for training")
