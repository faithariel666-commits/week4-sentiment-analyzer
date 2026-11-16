"""Train and evaluate a unigram movie-review sentiment classifier."""

from nltk import download
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABELS = ('negative', 'positive')
SEED = 42


def train_baseline():
    """Return the fitted pipeline and the two labeled data splits."""
    try:
        review_ids = sorted(movie_reviews.fileids())
    except LookupError:
        download('movie_reviews', quiet=True, raise_on_error=True)
        review_ids = sorted(movie_reviews.fileids())

    texts = [movie_reviews.raw(fileid) for fileid in review_ids]
    category_labels = {'neg': 'negative', 'pos': 'positive'}
    labels = [category_labels[movie_reviews.categories(fileid)[0]] for fileid in review_ids]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels
    )

    # Keep negation words and normalize case and curly apostrophes. The
    # tokenizer keeps contractions together and drops standalone punctuation.
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=normalize_review,
            token_pattern=r"(?u)\b\w+(?:'\w+)*\b",
            ngram_range=(1, 1),
        )),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    model.fit(train_texts, train_labels)
    return model, train_labels, test_texts, test_labels


def normalize_review(text):
    return text.lower().replace('\u2019', "'")


def main():
    model, train_labels, test_texts, test_labels = train_baseline()
    print('Split       Negative  Positive  Total')
    for name, labels in (
        ('All', train_labels + test_labels),
        ('Training', train_labels),
        ('Held out', test_labels),
    ):
        print(f'{name:<11} {labels.count("negative"):>8}  '
              f'{labels.count("positive"):>8}  {len(labels):>5}')
    predictions = model.predict(test_texts)
    print(f'\nHeld-out accuracy: {accuracy_score(test_labels, predictions):.2%}')
    print('Confusion matrix (rows actual, columns predicted; negative, positive):')
    print(confusion_matrix(test_labels, predictions, labels=LABELS))


if __name__ == '__main__':
    main()
