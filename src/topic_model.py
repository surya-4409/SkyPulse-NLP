import pandas as pd
import pickle
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn

def main():
    print("Loading preprocessed data for Topic Modeling...")
    input_path = os.path.join('output', 'preprocessed_data.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
        
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['cleaned_text'])
    
    # 1. Vectorization (Bag of Words)
    print("Vectorizing text...")
    vectorizer = CountVectorizer(max_features=5000, max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(df['cleaned_text'])
    
    # 2. Train LDA Model
    print("Training LDA model (this may take a minute)...")
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(dtm)
    
    # Save the model
    with open(os.path.join('output', 'lda_model.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
        
    # 3. Extract Topics
    print("Extracting top words per topic...")
    topics_data = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for index, topic in enumerate(lda_model.components_):
        top_words_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_data[f"topic_{index}"] = top_words
        
    # Save topics
    with open(os.path.join('output', 'topics.json'), 'w') as f:
        json.dump(topics_data, f, indent=4)
        
    # 4. Generate Visualization
    print("Generating interactive visualization...")
    
    # --- FIX: Monkey patch the vectorizer to support pyLDAvis ---
    # This aliases the new method name back to the old one pyLDAvis expects
    vectorizer.get_feature_names = vectorizer.get_feature_names_out
    # ------------------------------------------------------------
    
    panel = pyLDAvis.sklearn.prepare(lda_model, dtm, vectorizer, mds='tsne')
    
    # Save visualization
    output_html = os.path.join('output', 'lda_visualization.html')
    pyLDAvis.save_html(panel, output_html)
    
    print("Success! Topic model and visualization saved to output/ folder.")

if __name__ == "__main__":
    main()