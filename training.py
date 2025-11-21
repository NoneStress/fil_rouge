import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv("./data/data.csv")
print("Data chargee avec succes ✔")

# Preprocess data with TF-IDF (better than CountVectorizer for text classification)
vectorizer = TfidfVectorizer(
    max_features=150,  # Increased feature space
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams for better context
    lowercase=True,
    stop_words='french',
    min_df=1,  # Minimum document frequency
    max_df=0.8  # Maximum document frequency
)
X = vectorizer.fit_transform(df['description'])
y = df['category']

# Split data with stratification to maintain category distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with optimized alpha for better generalization
model = MultinomialNB(alpha=0.05)  # Lower alpha for more confidence
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# # Test confidence scores on sample data
# print("\n🔍 Sample Confidence Tests:")
# test_samples = ['Electricity bill', 'Doctor visit', 'Supermarket groceries', 'Uber ride']
# for sample in test_samples:
#     X_test_sample = vectorizer.transform([sample])
#     pred = model.predict(X_test_sample)[0]
#     prob = model.predict_proba(X_test_sample)[0].max() * 100
#     print(f"   '{sample}' → {pred} ({prob:.1f}% confidence)")

# # Save model and vectorizer
# joblib.dump(model, 'expense_categorizer_model.pkl')
# joblib.dump(vectorizer, 'vectorizer.pkl')

# print(f"\n✅ Model trained and saved successfully!")
# print(f"📈 Training samples: {len(df)}")
# print(f"🏷️  Categories: {', '.join(df['category'].unique())}")