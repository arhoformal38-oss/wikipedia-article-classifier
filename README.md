# NLP Mini Project 1 — Wikipedia Article-Type Classifier (DBpedia-14)

Phase 1 of the semester project (Units 1–4). Corpus option 6: DBpedia-14, six classes
(Company, Artist, Athlete, Building, Film, WrittenWork).

## Results (official test split, 30,000 unseen articles)
| Model | Accuracy | Macro F1 |
|---|---|---|
| Bag of Words (CountVectorizer) + Logistic Regression | 0.9648 | 0.9648 |

Interpolated trigram language model: test perplexity 254.0 (bigram 287.7).

## Folder structure
```
NLP_Project1_DBpedia/
├── notebooks/NLP_Project1_DBpedia.ipynb   main notebook (runs top to bottom)
├── src/preprocessing.py                   reusable preprocessing function
├── src/ngram_lm.py                        bigram/trigram language model
├── data/train.csv, data/test.csv          frozen train/test split (orig_idx = index in official DBpedia-14 split)
├── models/bow_classifier.joblib           trained CountVectorizer + Logistic Regression pipeline
├── results/test_metrics.json              final test metrics
├── results/test_predictions.csv           prediction for every test article
├── results/figures/                       plots used in the report
└── report/Project1_Report.docx            short report
```

## How to run
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook NLP_Project1_DBpedia.ipynb
```
Run all cells. If `data/train.csv` and `data/test.csv` exist they are reused (frozen split);
otherwise the notebook downloads DBpedia-14 from Hugging Face (`fancyzhx/dbpedia_14`) and recreates them
with `random_state=42`.

**Google Colab:** upload the whole folder (or at least `src/` and `data/`), open the notebook from
`notebooks/`, and run `!pip install datasets nltk scikit-learn seaborn` first.

## Reproducibility
- Train: 3,000 random articles per class from the official training split (`random_state=42`).
- Test: all 5,000 official test articles per class; used only for the final evaluation.
- Classifier selection: 5-fold stratified CV on training data. LM interpolation weights: 10% validation slice of training data.

## Reference
Zhang, X., Zhao, J., LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS.
