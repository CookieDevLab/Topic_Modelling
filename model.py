import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#Please update the path of the dataset according to where you save them 
train_data = pd.read_csv('C:\Python\Debug\BBC News Train.csv')
test_data = pd.read_csv('C:\Python\Debug\BBC News Test.csv')

train_data['Text'] = train_data['Text'].str.lower()
test_data['Text'] = test_data['Text'].str.lower()

X_train, X_val, y_train, y_val = train_test_split(train_data['Text'], train_data['Category'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test_data['Text'])  

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

val_preds = model.predict(X_val_tfidf)
test_preds = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {accuracy}")

precision = precision_score(y_val, val_preds, average='weighted')
recall = recall_score(y_val, val_preds, average='weighted')
f1 = f1_score(y_val, val_preds, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
