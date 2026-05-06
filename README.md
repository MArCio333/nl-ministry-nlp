# nl-ministry-nlp

NLP pipeline for topic modeling and clustering of Dutch parliamentary letters (kamerstukken) using spaCy, TF-IDF, and LDA.

---

## What it does

This pipeline processes a corpus of Dutch-language PDF ministry letters and produces:

- **Topic modeling** — LDA-based extraction of latent topics across the corpus
- **Document clustering** — K-Means grouping of documents by content similarity
- **Key phrase extraction** — TF-IDF ranked bigrams and trigrams
- **Style metrics** — average sentence length and modal verb frequency

It was built to analyze parliamentary correspondence from the Dutch Ministry of Economic Affairs and Climate Policy, covering letters from 2020 onwards.

---

## Corpus

188 kamerstukken (parliamentary letters) from the Dutch Ministry of Economic Affairs and Climate Policy, 2020 to present. Topics identified in the output reflect the ministry's main policy areas over this period:

- Groningen gas storage operations and earthquake damage compensation
- Energy transition and security (post-Ukraine)
- National Growth Fund allocations
- EU policy and legislation
- Foreign investment and international trade
- Labour market reform (ZZP/self-employment)

---

## Why these tools

**spaCy `nl_core_news_sm`** — provides Dutch lemmatization, POS tagging, and named entity recognition. Used here for lemmatization and entity masking, with POS filtering retaining only nouns, proper nouns, and adjectives to reduce verb and function word noise.

**Bureaucratic stoplist** — Dutch parliamentary letters contain dense boilerplate formatting (`vergaderjaar`, `kamerstuk`, `begrotingsstaat`, `decharge`, etc.) that appears uniformly across documents and carries no discriminative analytical value. A domain-specific stoplist was built iteratively after inspecting initial topic model output, suppressing this noise before vectorization.

**Entity masking** — named entities are masked before preprocessing to prevent proper nouns from dominating topics, allowing thematic content terms to surface more clearly.

**LDA over BERTopic** — corpus size (188 documents) is sufficient for LDA but borderline for reliable dense embeddings. LDA produces interpretable, stable topics at this scale.

**MIN_DF=3** — set empirically. Lower values retained formatting artifacts; higher values suppressed domain-specific terms that appeared in meaningful subsets of letters.

**POS filtering** — retaining only nouns, proper nouns, and adjectives (`NOUN`, `PROPN`, `ADJ`) significantly improved topic coherence compared to unfiltered lemmatization, by removing verbal and procedural noise common in administrative text.

---

## Requirements

```
spacy==3.8.13
spacy-legacy==3.0.12
spacy-loggers==1.0.5
pdfplumber==0.11.9
scikit-learn==1.8.0
numpy==2.4.2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the Dutch spaCy model (downloaded automatically on first run):

```bash
python -m spacy download nl_core_news_sm
```

---

## Usage

```bash
python nl_ministry_nlp.py /path/to/your/letters
```

Output is printed as JSON to stdout. Redirect to save:

```bash
python nl_ministry_nlp.py /path/to/letters > results.json
```

### Parameters (edit at top of script)

| Parameter | Default | Description |
|---|---|---|
| `N_TOPICS` | 10 | Number of LDA topics |
| `N_CLUSTERS` | 10 | Number of K-Means clusters |
| `N_TOP_WORDS` | 40 | Words per topic |
| `MIN_DF` | 3 | Minimum document frequency for TF-IDF |
| `MAX_DF` | 0.85 | Maximum document frequency for TF-IDF |

---

## Stopwords

The pipeline applies three filtering layers:

- **spaCy Dutch stopwords** — built-in function word filtering
- **Bureaucratic stoplist** — 60+ terms specific to Dutch parliamentary document formatting, built iteratively after inspecting initial output (vergaderjaar, kamerstuk, begrotingsstaat, decharge, auditdienst, etc.)
- **POS filtering** — only nouns, proper nouns, and adjectives are retained after lemmatization

---

## Output structure

```json
{
  "filenames": ["doc1.pdf", "doc2.pdf"],
  "topics": [["term1", "term2", ...], ...],
  "clusters": [0, 1, 0, 2, ...],
  "phrases": ["key phrase one", "key phrase two", ...],
  "style": {
    "avg_sentence_length": 19.2,
    "total_modal_verbs": 3854
  }
}
```

- `topics` — list of N_TOPICS topic word lists, ordered by weight
- `clusters` — cluster assignment per document, aligned with `filenames`
- `phrases` — top 50 bigrams and trigrams by TF-IDF score across corpus
- `style` — corpus-level indicators; modal verbs tracked are moet, zal, mag, kan, kunnen

See `sample_output.json` for a real example with annotated topic labels.

---

## Notes

- Only `.pdf` files are processed; other formats are ignored
- Documents that fail to extract text are skipped with a warning
- The Dutch spaCy model is downloaded automatically on first run if not present
- Entity masking is applied before lemmatization to prevent named entities from dominating topics

---

## License

MIT
