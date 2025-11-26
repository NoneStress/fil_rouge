import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

french_stopwords = [
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "ça", "car",
    "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "de", "des", "du",
    "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "droite", "du", "d'une",
    "d'", "elle", "elles", "en", "encore", "essai", "est", "et", "étaient", "étais", "était",
    "étant", "etc", "été", "être", "eu", "eue", "eues", "eux", "faire", "fait", "fois", "font",
    "force", "haut", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur",
    "leurs", "ma", "maintenant", "mais", "mes", "même", "moi", "mon", "mot", "ni", "nommés",
    "nos", "notre", "nous", "nouveaux", "ou", "où", "par", "parce", "parole", "pas", "personnes",
    "peu", "peut", "plupart", "pour", "pourquoi", "qu", "que", "quel", "quelle", "quelles",
    "quels", "qui", "sa", "sans", "sera", "serai", "seraient", "serait", "seras", "serez",
    "seriez", "serions", "serons", "seront", "ses", "seulement", "si", "sien", "son", "sont",
    "sous", "soyez", "soyons", "suis", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton",
    "tous", "tout", "toute", "toutes", "très", "trop", "tu", "un", "une", "valeur", "voie",
    "voient", "vont", "vos", "votre", "vous", "vu"
]

# Tester avec un autre model eventuellement
# Hugging Face Transformers (ou bibliothèques basées sur les “transformers” / embeddings modernes)

# Si tu cherches des représentations plus “riches” que TF-IDF — c.-à-d. des embeddings contextuels : pour capturer le sens, les synonymes, les relations sémantiques — Transformers offre des modèles pré-entraînés (BERT, RoBERTa, etc.) très puissants pour classification, similarité, embedding de phrases/documents. 
# ActiveTech Systems
# +2
# textpulse
# +2

# Cela peut donner de bien meilleurs résultats que TF-IDF + modèle classique, surtout si la sémantique / le contexte a de l’importance (dans ton cas, les descriptions immobilières, sujets NLP, etc.).


df = pd.read_csv("./data/data.csv")
print("Data chargee avec succes ✔")

# Preprocess data with TF-IDF (better than CountVectorizer for text classification)
vectorizer = TfidfVectorizer(
    max_features=150,  # Increased feature space
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams for better context
    lowercase=True,
    stop_words= french_stopwords,
    min_df=1,  # Minimum document frequency
    max_df=0.8  # Maximum document frequency
)
X = vectorizer.fit_transform(df['description'])
y = df['categories']

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