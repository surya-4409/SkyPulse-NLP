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
    
    # --- PATH FIX ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    input_path = os.path.join(project_root, 'output', 'preprocessed_data.csv')
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    # ----------------

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
        
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['cleaned_text'])
    
    print("Vectorizing text...")
    vectorizer = CountVectorizer(max_features=5000, max_df=0.9, min_df=2)
    dtm = vectorizer.fit_transform(df['cleaned_text'])
    
    print("Training LDA model (this may take a minute)...")
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(dtm)
    
    with open(os.path.join(output_dir, 'lda_model.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
        
    print("Extracting top words per topic...")
    topics_data = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for index, topic in enumerate(lda_model.components_):
        top_words_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_indices]
        topics_data[f"topic_{index}"] = top_words
        
    with open(os.path.join(output_dir, 'topics.json'), 'w') as f:
        json.dump(topics_data, f, indent=4)
        
    print("Generating interactive visualization...")
    # Monkey patch for pyLDAvis compatibility
    vectorizer.get_feature_names = vectorizer.get_feature_names_out
    
    panel = pyLDAvis.sklearn.prepare(lda_model, dtm, vectorizer, mds='tsne')
    
    output_html = os.path.join(output_dir, 'lda_visualization.html')
    pyLDAvis.save_html(panel, output_html)
    
    print("Success! Topic model and visualization saved.")

if __name__ == "__main__":
    main()