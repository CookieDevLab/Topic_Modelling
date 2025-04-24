### BBC News Topic Classifier 📰🤖

A Python-based machine learning project that classifies BBC news articles into categories using Natural Language Processing techniques. It uses TF-IDF for feature extraction and a Multinomial Naive Bayes classifier for prediction.

### 🔍 Overview

This project demonstrates text classification with a real-world dataset of BBC news articles. The process includes:

- Text preprocessing and cleaning
- TF-IDF vectorization for numerical representation
- Training and evaluation of a Naive Bayes model
- Performance reporting with precision, recall, F1-score

### ⚙️ Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas` for data handling
  - `scikit-learn` for ML modeling and metrics
  - `TfidfVectorizer` for feature extraction
- **Model**: Multinomial Naive Bayes

### 📁 Dataset

The dataset consists of two CSV files:

- `BBC News Train.csv` — Training data
- `BBC News Test.csv` — Testing data

Each file contains:
- `Text`: The news article content
- `Category`: The category label (e.g., tech, politics, business)

> **Note:** Make sure to update the file paths in the code as per your local directory structure.

### 🧠 Model

- **Vectorizer**: `TfidfVectorizer` (with stopword removal and `max_df=0.7`)
- **Classifier**: `MultinomialNB` from `sklearn.naive_bayes`

### 🛠️ Requirements

Install dependencies using:

```bash
pip install pandas scikit-learn
```

###🚀 How to Run
```bash
python topic_classifier.py
```
Ensure that the dataset files are correctly placed and the file paths in the script are updated accordingly.

###📈 Sample Output
```bash
Validation Accuracy: 0.965
Precision: 0.96
Recall: 0.965
F1-score: 0.962
```

###📊 Future Improvements
- Support for more complex models (e.g., SVM, LSTM)
- Web-based demo using Flask or Streamlit
- Improved preprocessing (lemmatization, named entity removal)

###🙌 Acknowledgements
Dataset sourced from [BBC News Dataset](https://www.kaggle.com/competitions/learn-ai-bbc/)
Inspired by academic and real-world applications of text classification
Shoutout to scikit-learn for making ML accessible and efficient 🚀


