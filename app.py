from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import re
import pandas as pd
import nltk
import lime.lime_tabular
import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

app = Flask(__name__)
CORS(app)
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
model = joblib.load(r"C:\Users\neera\Downloads\11_new\old save\Test 1\xgb_model_with_adv_features.pkl")
vectorizer = joblib.load(r"C:\Users\neera\Downloads\11_new\old save\Test 1\count_vectorizer.pkl")
scaler = joblib.load(r"C:\Users\neera\Downloads\11_new\old save\Test 1\feature_scaler.pkl")

spam_trigger_words = set([
    'win', 'free', 'urgent', 'money', 'offer', 'click', 'limited', 'buy now',
    'act now', 'congratulations', 'selected', 'reward', 'risk-free', 'guaranteed',
    'trial', 'claim', 'exclusive', 'winner', 'credit card', 'storage', 'upgrade',
    'account', 'verify', 'login', 'reset', 'security alert', 'suspend'
])
suspicious_domains = {'phishy.com', 'scamalert.net', 'bad-domain.org'}

# === Feature Engineering ===
def count_uppercase_words(text): return sum(1 for word in text.split() if word.isupper())
def count_exclamations(text): return text.count('!')
def count_links(text): return len(re.findall(r'https?://\S+|www\.\S+', text))
def count_special_chars(text): return len(re.findall(r'[^A-Za-z0-9\s]', text))
def starts_with_greeting(text): return int(text.strip().lower().split()[0] in ['hi', 'hello', 'dear']) if text.strip() else 0
def has_reply_keywords(text): return int(any(k in text.lower() for k in ['regards', 'thank you', 'sincerely']))
def contains_spam_words(text): return int(any(w in text.lower() for w in spam_trigger_words))
def suspicious_word_ratio(text): 
    words = text.lower().split()
    return sum(1 for word in words if any(spam in word for spam in spam_trigger_words)) / len(words) if words else 0
def urgency_score(text): return int(any(k in text.lower() for k in ['immediately', 'asap', 'urgent', 'now', 'act now']))
def detect_phishing_pattern(text):
    matches = re.findall(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)">(.*?)</a>', text, re.IGNORECASE)
    return int(any(display.lower() not in url.lower() for url, display in matches if display and url))
def domain_threat_check(text):
    domains = re.findall(r'https?://([^/\s]+)', text)
    return int(any(domain in suspicious_domains for domain in domains))
def ner_count(text):
    doc = nlp(text)
    return len([ent for ent in doc.ents])
def sentiment_polarity(text): return TextBlob(text).sentiment.polarity

def extract_features(email_text):
    return {
        'Length': len(email_text),
        'num_words': len(word_tokenize(email_text)),
        'num_sentence': len(sent_tokenize(email_text)),
        'avg_word_length': np.mean([len(w) for w in email_text.split()]) if email_text else 0,
        'readability_score': len(email_text.split()) / (len(sent_tokenize(email_text)) + 1),
        'num_uppercase_words': count_uppercase_words(email_text),
        'num_exclamations': count_exclamations(email_text),
        'num_links': count_links(email_text),
        'num_special_chars': count_special_chars(email_text),
        'starts_with_greeting': starts_with_greeting(email_text),
        'has_reply_keywords': has_reply_keywords(email_text),
        'contains_spam_words': contains_spam_words(email_text),
        'suspicious_word_ratio': suspicious_word_ratio(email_text),
        'urgency_score': urgency_score(email_text),
        'phishing_pattern': detect_phishing_pattern(email_text),
        'domain_threat': domain_threat_check(email_text),
        'ner_count': ner_count(email_text),
        'sentiment_polarity': sentiment_polarity(email_text)
    }

@app.route('/predict', methods=['POST'])
def predict_email():
    try:
        data = request.get_json()
        email_text = data.get('email', '')

        if not email_text.strip():
            return jsonify({'error': 'Email text is empty'}), 400

        # Clean and vectorize
        clean_text = re.sub(r'https?://\S+|www\.\S+', '', email_text)
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', clean_text.lower())

        vectorized = vectorizer.transform([clean_text]).toarray()
        feature_dict = extract_features(email_text)
        df_features = pd.DataFrame([feature_dict])
        scaled_features = scaler.transform(df_features)

        # Combine features for model input
        final_input = np.hstack((vectorized, scaled_features))
        prediction = model.predict(final_input)[0]
        probs = model.predict_proba(final_input)[0]
        label = 'Spam' if prediction == 1 else 'Not Spam'

        # Always generate LIME plot
        sample_texts = [
            "Hello, please review the attached document.",
            "Congratulations! You've won a free iPhone. Click here.",
            "Reminder: Update your account settings ASAP.",
            "Thank you for your message, we’ll get back to you soon.",
            "Dear user, urgent security update required."
        ]
        training_features = [extract_features(text) for text in sample_texts]
        training_scaled = scaler.transform(pd.DataFrame(training_features))

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_scaled,
            mode="classification",
            feature_names=list(feature_dict.keys()),
            class_names=['Not Spam', 'Spam'],
            discretize_continuous=True
        )

        exp = explainer.explain_instance(
            data_row=scaled_features[0],
            predict_fn=lambda x: model.predict_proba(np.hstack((vectorized.repeat(x.shape[0], axis=0), x))),
            num_features=8
        )

        fig = exp.as_pyplot_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # Only add reasons if it's Spam
        reasons = []
        if prediction == 1:
            if feature_dict['contains_spam_words']:
                reasons.append("Includes known spam keywords.")
            if feature_dict['urgency_score']:
                reasons.append("Urgent or fear-based language detected.")
            if feature_dict['num_links'] > 0:
                reasons.append(f"Contains {feature_dict['num_links']} link(s).")
            if feature_dict['num_uppercase_words'] > 5:
                reasons.append("Excessive capital letters.")
            if not feature_dict['starts_with_greeting']:
                reasons.append("Missing a greeting.")
            if not feature_dict['has_reply_keywords']:
                reasons.append("Missing polite closing.")
            if feature_dict['suspicious_word_ratio'] > 0.2:
                reasons.append("High ratio of suspicious words.")
            if feature_dict['phishing_pattern']:
                reasons.append("Phishing link pattern detected.")
            if feature_dict['domain_threat']:
                reasons.append("Links to known suspicious domains.")
            if feature_dict['sentiment_polarity'] < -0.2:
                reasons.append("Negative sentiment.")

        return jsonify({
            'prediction': label,
            'probabilities': {'Not Spam': float(probs[0]), 'Spam': float(probs[1])},
            'reasons': reasons,
            'lime_plot': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)
