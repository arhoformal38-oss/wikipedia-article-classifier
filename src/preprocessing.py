"""Reusable text preprocessing for the DBpedia article-type project.

One function, `preprocess`, is used everywhere (corpus statistics, N-grams,
classifier, demo) so that training, testing and new input are treated the same.
It will be reused in Project 2.
"""
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_nltk_resources():
    """Download the NLTK data this module needs (safe to call many times)."""
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)


ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalization: lowercase, strip accents (maría -> maria), remove URLs,
    digits, punctuation and non-Latin characters, collapse spaces."""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def preprocess(text: str, remove_stopwords: bool = True, normalize: str = "lemma",
               min_len: int = 2, as_string: bool = False):
    """Full pipeline: clean -> tokenize -> stop-word removal -> lemmatize/stem.

    Parameters
    ----------
    remove_stopwords : drop English stop words (good for classification,
                       bad for language modelling, where function words matter).
    normalize        : "lemma" (WordNet lemmatizer), "stem" (Porter) or None.
    min_len          : drop tokens shorter than this.
    as_string        : return a space-joined string (handy for CountVectorizer).
    """
    tokens = word_tokenize(clean_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if len(t) >= min_len]
    if normalize == "lemma":
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    elif normalize == "stem":
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens) if as_string else tokens


def sentences_for_lm(text: str):
    """Light preprocessing for the N-gram language model.

    Keeps stop words (they carry the word order the model must learn) and does
    not lemmatize; splits into sentences so <s> / </s> markers can be added.
    """
    out = []
    for sent in sent_tokenize(str(text)):
        tokens = word_tokenize(clean_text(sent))
        if len(tokens) >= 2:
            out.append(tokens)
    return out
