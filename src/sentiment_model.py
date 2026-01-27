import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    print("Loading preprocessed data...")
    input_path = os.path.join('output', 'preprocessed_data.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}. Run preprocess.py first.")
        
    df = pd.read_csv(input_path)
    
    # Handle missing values (if any empty tweets were created during cleaning)
    df = df.dropna(subset=['cleaned_text', 'airline_sentiment'])
    
    # Define features (X) and target (y)
    X = df['cleaned_text']
    y = df['airline_sentiment']
    
    # Split data into Training and Test sets (80% train, 20% test)
    print("Splitting data...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, df['tweet_id'], test_size=0.2, random_state=42
    )
    
    # 1. Vectorization (TF-IDF)
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Save the vectorizer (Requirement: output/tfidf_vectorizer.pkl)
    with open(os.path.join('output', 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # 2. Train Model (Logistic Regression)
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Save the model (Requirement: output/sentiment_model.pkl)
    with open(os.path.join('output', 'sentiment_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    # 3. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_score_macro": f1_score(y_test, y_pred, average='macro')
    }
    
    # Save metrics (Requirement: output/sentiment_metrics.json)
    with open(os.path.join('output', 'sentiment_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 4. Save Predictions (Requirement: output/sentiment_predictions.csv)
    print("Saving predictions...")
    predictions_df = pd.DataFrame({
        'tweet_id': ids_test,
        'predicted_sentiment': y_pred
    })
    predictions_df.to_csv(os.path.join('output', 'sentiment_predictions.csv'), index=False)
    
    print("Success! Sentiment model trained and artifacts saved to output/ folder.")

if __name__ == "__main__":
    main()