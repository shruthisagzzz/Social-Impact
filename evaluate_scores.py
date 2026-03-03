import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from textblob import TextBlob

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define keyword sets for scoring
sustainability_keywords = {'sustainability', 'climate', 'environment', 'carbon', 'renewable', 'green', 'ecology'}
ethical_keywords = {'ethics', 'compliance', 'integrity', 'transparency', 'fairness', 'accountability'}
community_keywords = {'community', 'volunteer', 'donation', 'charity', 'support', 'upliftment'}
public_keywords = {'public', 'engagement', 'awareness', 'campaign', 'communication', 'participation'}

# ----------- Core Functions -----------

def preprocess_text(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range: [-1, 1]

def extract_entities(text):
    doc = nlp(text)
    return [ent.label_ for ent in doc.ents]

def get_dominant_topic(bow, lda_model):
    topics = lda_model.get_document_topics(bow)
    return max(topics, key=lambda x: x[1])[0] if topics else None

def compute_scores(row):
    tokens = set(row['processed_text'])
    sentiment = row['sentiment_score']

    # Raw keyword matching scores
    sustainability_score = len(sustainability_keywords & tokens) * 10
    ethical_score = len(ethical_keywords & tokens) * 10
    community_score = len(community_keywords & tokens) * 10
    public_score = len(public_keywords & tokens) * 10

    # Add sentiment adjustment
    sentiment_adjustment = sentiment * 10  # scale to match 0-100 range
    sustainability_score += sentiment_adjustment
    ethical_score += sentiment_adjustment
    community_score += sentiment_adjustment
    public_score += sentiment_adjustment

    # Clamp scores between 0 and 100
    return pd.Series([
        min(max(sustainability_score, 0), 100),
        min(max(ethical_score, 0), 100),
        min(max(community_score, 0), 100),
        min(max(public_score, 0), 100)
    ], index=['Sustainability_Score', 'Ethical_Business_Score', 'Community_Engagement_Score', 'Public_Engagement_Score'])

# ----------- Main Function -----------

def evaluate_social_impact(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    text_column = 'text'  # Replace with actual column name if different

    # Preprocessing
    df['processed_text'] = df[text_column].astype(str).apply(preprocess_text)
    df['sentiment_score'] = df[text_column].astype(str).apply(get_sentiment)
    df['entities'] = df[text_column].astype(str).apply(extract_entities)

    # Topic Modeling
    dictionary = corpora.Dictionary(df['processed_text'])
    corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
    lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=10)
    df['dominant_topic'] = [get_dominant_topic(bow, lda_model) for bow in corpus]

    # Compute Scores
    df[['Sustainability_Score', 'Ethical_Business_Score',
        'Community_Engagement_Score', 'Public_Engagement_Score']] = df.apply(compute_scores, axis=1)

    # Export enhanced dataset
    df.to_csv(output_csv_path, index=False)
    print(f"Evaluation complete. Output saved to {output_csv_path}")