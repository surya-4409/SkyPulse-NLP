
# ✈️ SkyPulse: Airline Sentiment & Topic Analysis  

## 📌 Project Overview
SkyPulse is a comprehensive Natural Language Processing (NLP) pipeline designed to analyze public sentiment towards US Airlines. The application processes raw tweets to classify sentiment (Positive, Negative, Neutral) and utilizes Latent Dirichlet Allocation (LDA) to discover underlying topics of conversation, such as baggage issues or flight delays.

The entire application is containerized using **Docker** to ensure reproducibility and ease of deployment.

## 🚀 Features
* **Text Preprocessing Pipeline:** Cleans raw text by removing URLs, punctuation, and stopwords, followed by lemmatization.
* **Sentiment Classification:** A Logistic Regression model (trained on TF-IDF features) predicting tweet sentiment with ~80% accuracy.
* **Topic Modeling:** An unsupervised LDA model that extracts key themes from customer feedback.
* **Interactive Dashboard:** A Streamlit-based web interface for visualizing dataset statistics, model metrics, and an interactive LDA topic map.

## 📂 Repository Structure
```text
./
├── data/                   # Contains the input dataset (tweets.csv)
├── output/                 # Generated artifacts (models, metrics, visualizations)
├── src/                    # Source code modules
│   ├── preprocess.py       # Data cleaning logic
│   ├── sentiment_model.py  # Sentiment classification training
│   └── topic_model.py      # LDA topic modeling
├── app.py                  # Streamlit Dashboard application
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
└── README.md               # Project documentation

```

## 🛠️ Setup & Installation

### Prerequisites

* Docker Desktop installed and running.

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/surya-4409/SkyPulse-NLP.git
cd skypulse-nlp

```


2. **Add the Data:**
Ensure the `tweets.csv` file is placed inside the `data/` directory.
3. **Run with Docker Compose:**
This command builds the image and starts the application.
```bash
docker-compose up --build

```


4. **Access the Dashboard:**
Open your browser and navigate to: [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)

## 📊 Methodology & Findings

### 1. Sentiment Analysis

* **Model:** Logistic Regression with TF-IDF Vectorization (Max Features: 5000).
* **Performance:** The model achieved an accuracy of approximately **80%**.
* **Observation:** The dataset is heavily imbalanced towards negative sentiment. The model performs well on negative tweets but occasionally struggles to distinguish between "Neutral" and "Positive" when the text is short or sarcastic.

### 2. Topic Modeling (LDA)

* **Technique:** Latent Dirichlet Allocation (LDA) with CountVectorizer.
* **Key Topics Identified:**
* **Topic 0:** General flight delays and gate issues.
* **Topic 1:** Customer service interactions (thanks, service, help).
* **Topic 2:** Baggage and lost luggage complaints.
* **Topic 3:** Late flights and cancellations.


* **Insight:** The interactive PyLDAvis visualization clearly separates "service-related" topics from "logistics-related" topics (baggage, delays).

## 🛡️ Requirements Checklist

* [x] Dockerized application (Dockerfile + docker-compose.yml)
* [x] Data preprocessing script (`src/preprocess.py`)
* [x] TF-IDF Vectorizer & Sentiment Model (`src/sentiment_model.py`)
* [x] Evaluation Metrics saved as JSON
* [x] LDA Topic Modeling & Visualization (`src/topic_model.py`)
* [x] Interactive Streamlit Dashboard (`app.py`)

