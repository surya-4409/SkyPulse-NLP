import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Ensure NLTK data is downloaded (useful for local testing)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    """
    1. Lowercase
    2. Remove URLs
    3. Remove punctuation and numbers
    4. Tokenize & Remove Stopwords
    5. Lemmatize (convert words to base form, e.g., 'running' -> 'run')
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs (http/https/www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize (split into words)
    tokens = text.split()
    
    # 5. Remove Stopwords & Lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # We keep words that are NOT in the stop_words list
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(clean_tokens)

def main():
    print("Loading data...")
    # Define paths
    input_path = os.path.join('data', 'tweets.csv') # Ensure your file is named tweets.csv
    output_path = os.path.join('output', 'preprocessed_data.csv')
    
    # Check if data exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}. Please check the 'data' folder.")

    # Load dataset
    df = pd.read_csv(input_path)
    
    # The dataset usually comes with 'tweet_id' and 'text'. 
    # We need to make sure we map them correctly.
    # If the CSV has 'tweet_id', use it. If not, use the index as ID.
    if 'tweet_id' not in df.columns:
        print("Warning: 'tweet_id' column not found. Creating IDs from index.")
        df['tweet_id'] = df.index
        
    print("Preprocessing text (this might take a moment)...")
    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Filter only required columns
    final_df = df[['tweet_id', 'cleaned_text', 'airline_sentiment']] 
    # Note: We keep 'airline_sentiment' because we need it for training in Step 3!
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Success! Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    main()