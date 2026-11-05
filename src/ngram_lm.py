"""A small N-gram language model (bigram or trigram) with two smoothing
options (add-k or linear interpolation), backoff for next-word prediction,
and perplexity."""
import math
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


class NGramLM:
    def __init__(self, n: int = 3, smoothing: str = "interp", k: float = 0.01, lambdas=None):
        """smoothing="addk":   P = (c(context, w) + k) / (c(context) + k*V)
        smoothing="interp": P = l_n*P_ML(w|context) + ... + l_1*P_add1(w)
        (Jelinek-Mercer linear interpolation; lambdas sum to 1, highest order first)."""
        assert n in (2, 3), "only bigram (n=2) or trigram (n=3) models"
        assert smoothing in ("addk", "interp")
        self.n, self.k, self.smoothing = n, k, smoothing
        self.lambdas = lambdas or ((0.6, 0.3, 0.1) if n == 3 else (0.7, 0.3))
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.next_after_1 = defaultdict(Counter)  # w1 -> Counter(w2)
        self.next_after_2 = defaultdict(Counter)  # (w1, w2) -> Counter(w3)

    def fit(self, sentences):
        """sentences: iterable of token lists."""
        for tokens in sentences:
            padded = [BOS] * (self.n - 1) + list(tokens) + [EOS]
            self.unigrams.update(padded)
            for a, b in zip(padded, padded[1:]):
                self.bigrams[(a, b)] += 1
                self.next_after_1[a][b] += 1
            for a, b, c in zip(padded, padded[1:], padded[2:]):
                self.trigrams[(a, b, c)] += 1
                self.next_after_2[(a, b)][c] += 1
        self.vocab = set(self.unigrams) - {BOS}
        self.V = len(self.vocab)
        self.N = sum(c for w, c in self.unigrams.items() if w != BOS)
        return self

    # ---------- probabilities ----------
    def _ml(self, num, den):
        return num / den if den else 0.0

    def prob(self, word, context):
        """Smoothed P(word | context); context is a tuple of 1 or 2 words."""
        if self.smoothing == "interp":
            p1 = (self.unigrams[word] + 1) / (self.N + self.V)
            b = context[-1]
            p2 = self._ml(self.bigrams[(b, word)], self.unigrams[b])
            if self.n == 2:
                l2, l1 = self.lambdas
                return l2 * p2 + l1 * p1
            a = context[-2]
            p3 = self._ml(self.trigrams[(a, b, word)], self.bigrams[(a, b)])
            l3, l2, l1 = self.lambdas
            return l3 * p3 + l2 * p2 + l1 * p1
        if self.n == 3:
            a, b = context[-2:]
            num = self.trigrams[(a, b, word)] + self.k
            den = self.bigrams[(a, b)] + self.k * self.V
        else:
            a = context[-1]
            num = self.bigrams[(a, word)] + self.k
            den = self.unigrams[a] + self.k * self.V
        return num / den

    def perplexity(self, sentences):
        """Perplexity on held-out sentences. Words never seen in training
        (out-of-vocabulary) are skipped, a simplification stated in the report."""
        log_sum, count = 0.0, 0
        for tokens in sentences:
            tokens = [t if t in self.vocab else None for t in tokens]
            padded = [BOS] * (self.n - 1) + [t for t in tokens if t] + [EOS]
            for i in range(self.n - 1, len(padded)):
                log_sum += math.log(self.prob(padded[i], tuple(padded[i - self.n + 1:i])))
                count += 1
        return math.exp(-log_sum / count)

    # ---------- prediction ----------
    def predict_next(self, prompt_tokens, top_k: int = 5):
        """Most likely next words. Uses the trigram context when it was seen,
        otherwise backs off to the bigram, then to the most frequent unigrams."""
        tokens = [BOS] * (self.n - 1) + list(prompt_tokens)
        if self.n == 3 and self.next_after_2.get(tuple(tokens[-2:])):
            dist, used = self.next_after_2[tuple(tokens[-2:])], "trigram"
        elif self.next_after_1.get(tokens[-1]):
            dist, used = self.next_after_1[tokens[-1]], "bigram"
        else:
            dist = Counter({w: c for w, c in self.unigrams.items() if w not in (BOS, EOS)})
            used = "unigram"
        total = sum(dist.values())
        return [(w, c / total) for w, c in dist.most_common(top_k)], used

    def generate(self, prompt_tokens, max_words: int = 12):
        """Greedy continuation of a prompt."""
        out = list(prompt_tokens)
        for _ in range(max_words):
            preds, _ = self.predict_next(out, top_k=1)
            if not preds or preds[0][0] == EOS:
                break
            out.append(preds[0][0])
        return " ".join(out)
