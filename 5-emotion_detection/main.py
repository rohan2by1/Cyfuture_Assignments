import pandas as pd
import numpy as np
import neattext as nt
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Preprocess
df['clean_text'] = df['text'].apply(nfx.remove_stopwords)
df['clean_text'] = df['clean_text'].apply(str.lower)
df['clean_text'] = df['clean_text'].apply(nfx.remove_special_characters)

# 3. Split
X = df['clean_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Model Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# 5. Train
model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Optional: Visualize counts
sns.countplot(x='emotion', data=df)
plt.title("Emotion Distribution")
plt.show()

# 7. Save Model
joblib.dump(model, 'emotion_detector.pkl')

# 8. Load and Demo
loaded_model = joblib.load('emotion_detector.pkl')
sample_texts = ["I am so angry right now!", "This makes me so happy!", "I feel sad and lonely."]
for txt in sample_texts:
    # Preprocess: clean for consistency
    cleaned = nfx.remove_special_characters(nfx.remove_stopwords(txt.lower()))
    pred = loaded_model.predict([cleaned])[0]
    print(f"Input: {txt} | Predicted Emotion: {pred}")