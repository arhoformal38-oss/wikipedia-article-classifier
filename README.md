# NLP Mini Project 1 : Wikipedia Article-Type Classifier (DBpedia-14)

NLP Mini Project 1 (Phase 1) · Corpus Option 6: DBpedia-14 (Wikipedia-based articles)

**Dina Al-Deek · Rand Hussein** — Supervisor: Dr. Ahmad Fahel

Read the abstract of a Wikipedia article and predict its type — **Company, Artist, Athlete, Building, Film or
WrittenWork** — with a classical NLP pipeline: preprocessing, corpus statistics, N-gram analysis, a trigram
language model, Bag of Words, and Logistic Regression / Multinomial Naive Bayes.

## Results (official DBpedia-14 test set, 30,000 unseen articles)

| Model | Features | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | words + word pairs | **0.9673** | 0.9673 | 0.9673 | 0.9673 |
| Multinomial Naive Bayes | words + word pairs | 0.9555 | — | — | 0.9554 |

Trigram language model with interpolation: test perplexity **254.0** (bigram 287.7; with add-k: 3,934.5).

## Files

| File | Content |
|---|---|
| `notebooks/NLP_Project1_DBpedia.ipynb` | Full code with saved outputs (runs top to bottom) |
| `Report.pdf` | Project report |
| `presentation/index.html` | Presentation slides (← → keys, N for speaker notes) |
| `demo.py`, `notebooks/Demo_Colab.ipynb` | Live demo (command line / Google Colab) |
| `src/preprocessing.py`, `src/ngram_lm.py` | Reusable preprocessing function and N-gram language model |
| `data/train.csv`, `data/test.csv` | Frozen split (18,000 / 30,000); `orig_idx` = row in the official DBpedia-14 split |
| `models/` | Saved vectorizer, classifier, full pipeline and `config.json` |
| `results/` | Every table (CSV) and figure used in the report |

## How to run

1. Open `notebooks/NLP_Project1_DBpedia.ipynb` in Jupyter or Google Colab (`pip install -r requirements.txt`).
2. Run all cells. If `data/train.csv` and `data/test.csv` exist they are reused; otherwise the notebook downloads
   DBpedia-14 from Hugging Face (`fancyzhx/dbpedia_14`) and recreates the same split with `random_state=42`.
3. Live demo: [open Demo_Colab.ipynb in Colab](https://colab.research.google.com/github/arhoformal38-oss/wikipedia-article-classifier/blob/main/notebooks/Demo_Colab.ipynb)
   → Runtime → Run all, or run `python demo.py` locally.

**Presentation:** https://arhoformal38-oss.github.io/wikipedia-article-classifier/presentation/

## Reference

Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS.
