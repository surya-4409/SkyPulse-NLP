import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    """
    Standard text cleaning: lowercase, remove URLs/special chars, remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

def main():
    print("Loading data...")
    
    # --- PATH FIX: Calculate paths relative to this script's location ---
    # This ensures it works whether run from root or src/
    current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src
    project_root = os.path.dirname(current_dir)              # .../ (parent of src)
    
    # Check for Tweets.csv (Capital T) or tweets.csv (lowercase t)
    input_path_primary = os.path.join(project_root, 'data', 'Tweets.csv')
    input_path_fallback = os.path.join(project_root, 'data', 'tweets.csv')
    
    if os.path.exists(input_path_primary):
        input_path = input_path_primary
    elif os.path.exists(input_path_fallback):
        input_path = input_path_fallback
    else:
        # If neither exists, raise error
        raise FileNotFoundError(f"Input file not found. Checked:\n1. {input_path_primary}\n2. {input_path_fallback}")

    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'preprocessed_data.csv')
    # ------------------------------------------------------------------

    # Load dataset
    df = pd.read_csv(input_path)
    
    if 'tweet_id' not in df.columns:
        print("Warning: 'tweet_id' column not found. Creating IDs from index.")
        df['tweet_id'] = df.index
        
    print("Preprocessing text (this might take a moment)...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    final_df = df[['tweet_id', 'cleaned_text', 'airline_sentiment']] 
    
    final_df.to_csv(output_path, index=False)
    print(f"Success! Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    main()