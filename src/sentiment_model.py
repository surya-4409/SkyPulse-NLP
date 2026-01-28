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
    
    # --- PATH FIX ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_path = os.path.join(project_root, 'output', 'preprocessed_data.csv')
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    # ----------------
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}. Run preprocess.py first.")
        
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['cleaned_text', 'airline_sentiment'])
    
    X = df['cleaned_text']
    y = df['airline_sentiment']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, df['tweet_id'], test_size=0.2, random_state=42
    )
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save vectorizer
    with open(os.path.join(output_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Save model
    with open(os.path.join(output_dir, 'sentiment_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_score_macro": f1_score(y_test, y_pred, average='macro')
    }
    
    with open(os.path.join(output_dir, 'sentiment_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Saving predictions...")
    predictions_df = pd.DataFrame({
        'tweet_id': ids_test,
        'predicted_sentiment': y_pred
    })
    predictions_df.to_csv(os.path.join(output_dir, 'sentiment_predictions.csv'), index=False)
    
    print("Success! Sentiment model trained and artifacts saved.")

if __name__ == "__main__":
    main()