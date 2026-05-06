#!/usr/bin/env python3

import os
import re
import pdfplumber
import spacy
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import numpy as np
import json

# -----------------------------
# BUREAUCRATIC STOPLIST
# -----------------------------
BUREAUCRATIC_KEYWORDS = [
    "vergaderjaar", "vergaderjaaren", "kamerstuk", "tweede kamer", "kst-", "decharge", "dechargeadvie",
    "begrotingsstaat", "begrotingsstate", "iv vaststelling", "iv jaar", "vasta", "opnemen tabel",
    "commissie sociaal", "auditdienst", "auditdienst rijk", "szabó", "kf", "karremans", "minister",
    "staatssecretaris", "ministerie", "departementaal", "kabinet", "kabinetsreactie", "kabinet structureel",
    "interim", "conform", "besluit", "besluiten", "regeling", "regelingsoverzicht", "wet", "wetgeving",
    "koninkrijksrelatie", "koninklijk besluit", "haag februari", "algemeen zaak", "afschrift",
    "administratief", "departement", "coördinatie", "protocol", "richtlijn", "overhevelen ministerie",
    "vergaderjaar figuur", "vergadering", "vergaderingen", "per maand", "per luchthaven", "uitvoering",
    "vaststelling", "stelling", "hiervan miljard", "stellen ministerie", "doorlooppen", "uitlegbaar",
    "dragen gezamenlijk", "volledig besteden", "hoog verwacht", "commissie", "fich mfk", "motie lid",
    "artikel lid", "kamer brief", "kamerbrief", "kamerstuk ii", "administratief last", "rekening houden",
    "periode", "periodiek rapportage", "staatcommissie", "groeifonds jaar", "begrotingspost", "begroting",
    "staatfonds", "auditrichtlijn", "opnemen", "indicatie", "uitkomst", "evaluatie", "rapport", "rapportage",
    "statistiek", "jaarverslag", "jaarverslagen", "kwartaalrapport", "kwartaal", "tabel", "tabel overzicht",
    "vergadernotulen", "notulen", "brief minister", "brief"
]

# -----------------------------
# CONFIG
# -----------------------------
N_TOPICS = 10
N_CLUSTERS = 10
N_TOP_WORDS = 40
MIN_DF = 3
MAX_DF = 0.85

# -----------------------------
# LOAD DUTCH NLP MODEL
# -----------------------------
try:
    nlp = spacy.load("nl_core_news_sm")
except OSError:
    print("⬇️ Downloading Dutch model...")
    from spacy.cli import download
    download("nl_core_news_sm")
    nlp = spacy.load("nl_core_news_sm")

# -----------------------------
# PDF EXTRACTION
# -----------------------------
def extract_text_from_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        return text
    except Exception as e:
        print(f"Warning: Could not read PDF '{path}': {e}")
        return ""

def load_pdfs(folder_path: str) -> List[str]:
    texts = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            print(f"Loading: {file}")
            text = extract_text_from_pdf(full_path)
            if text.strip():
                texts.append(text)
                filenames.append(file)
            else:
                print(f"Warning: PDF '{file}' is empty or unreadable.")
    return texts, filenames

# -----------------------------
# PREPROCESSING
# -----------------------------
def clean_kamerbrief(text: str) -> str:
    text = re.sub(r"Tweede Kamer der Staten-Generaal.*?\d{4}–\d{4}", "", text, flags=re.DOTALL)
    text = re.sub(r"Kamerstuk \d+ \d+, nr\. \d+", "", text)
    text = re.sub(r"ISSN \d{4} - \d{4}", "", text)
    text = re.sub(r"kst-\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def mask_entities(text: str) -> str:
    doc = nlp(text)
    tokens = [token.text if token.ent_type_ == "" else "" for token in doc]
    return " ".join(tokens)

def preprocess(text: str) -> str:
    text = clean_kamerbrief(text.lower())
    text = mask_entities(text)  # optional entity masking
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.lemma_ not in BUREAUCRATIC_KEYWORDS
        and token.pos_ in ["NOUN", "PROPN", "ADJ"]  # keep content words
    ]
    return " ".join(tokens)

def preprocess_corpus(texts: List[str]) -> List[str]:
    return [preprocess(t) for t in texts]

# -----------------------------
# TF-IDF
# -----------------------------
def compute_tfidf(texts: List[str]):
    vectorizer = TfidfVectorizer(max_df=MAX_DF, min_df=MIN_DF, ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# -----------------------------
# TOPIC MODELING
# -----------------------------
def extract_topics(X, vectorizer) -> List[List[str]]:
    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic in lda.components_:
        top_indices = topic.argsort()[::-1]
        top_words = [feature_names[i] for i in top_indices[:N_TOP_WORDS]
                     if not any(word in BUREAUCRATIC_KEYWORDS for word in feature_names[i].split())]
        topics.append(top_words)
    return topics

# -----------------------------
# CLUSTERING
# -----------------------------
def cluster_documents(X) -> List[int]:
    model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    model.fit(X)
    return model.labels_

# -----------------------------
# PHRASE EXTRACTION
# -----------------------------
def extract_phrases(texts: List[str]) -> List[str]:
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_df=MAX_DF, min_df=MIN_DF)
    X = vectorizer.fit_transform(texts)
    tfidf_scores = np.array(X.sum(axis=0)).flatten()
    top_indices = np.argsort(tfidf_scores)[::-1]
    phrases = [
        vectorizer.get_feature_names_out()[i]
        for i in top_indices
        if not any(word in BUREAUCRATIC_KEYWORDS for word in vectorizer.get_feature_names_out()[i].split())
    ]
    return phrases

# -----------------------------
# STYLE ANALYSIS
# -----------------------------
def compute_style_metrics(text: str) -> Dict:
    doc = nlp(text)
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    modal_verbs = ["moet", "zal", "mag", "kan", "kunnen"]
    modal_count = sum(1 for token in doc if token.lemma_ in modal_verbs)
    return {"avg_sentence_length": avg_sentence_length, "modal_verb_count": modal_count}

def corpus_style(texts: List[str]) -> Dict:
    metrics = [compute_style_metrics(t) for t in texts]
    avg_len = sum(m["avg_sentence_length"] for m in metrics) / len(metrics) if metrics else 0
    modal_total = sum(m["modal_verb_count"] for m in metrics)
    return {"avg_sentence_length": avg_len, "total_modal_verbs": modal_total}

# -----------------------------
# MAIN ANALYSIS
# -----------------------------
def analyze_corpus(folder_path: str) -> Dict:
    print("Loading PDFs")
    texts, filenames = load_pdfs(folder_path)
    if not texts:
        print(f"No readable PDFs found in '{folder_path}'.")
        return {}
    print(f"Loaded {len(texts)} documents")

    print("Preprocessing")
    processed = preprocess_corpus(texts)

    print("Computing TF-IDF")
    X, vectorizer = compute_tfidf(processed)

    print("Extracting topics")
    topics = extract_topics(X, vectorizer)

    print("Clustering documents")
    clusters = cluster_documents(X)

    print("Extracting phrases")
    phrases = extract_phrases(processed)

    print("Analyzing style")
    style = corpus_style(texts)

    return {
        "filenames": filenames,
        "topics": topics,
        "clusters": clusters.tolist(),
        "phrases": phrases[:50],
        "style": style
    }

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    folder_path = ""
    if not os.path.isdir(folder_path):
        print(f"Error: folder '{folder_path}' does not exist.")
        exit(1)

    results = analyze_corpus(folder_path)
    print("\n--- RESULTS ---\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))