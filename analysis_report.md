# Week 4 Sentiment Analyzer for Reviews

## Setup

```sh
python -m pip install -r requirements.txt
python sentiment_analyzer.py
python challenge_set.py
```

Use Python 3.11 or 3.12. The baseline downloads the NLTK `movie_reviews` corpus on its first run if needed. The challenge script requires five labeled reviews per category.

## Dataset and held-out evaluation

The NLTK movie reviews corpus contains 2,000 labeled reviews. I used a stratified 80/20 split with random seed 42 and fit both the vectorizer and the logistic regression classifier on the training split.

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Full corpus | 1,000 | 1,000 | 2,000 |
| Training | 800 | 800 | 1,600 |
| Held-out test | 200 | 200 | 400 |

The classes are balanced. An always-positive prediction would get 50% accuracy on the held-out set. The unigram TF-IDF logistic regression model correctly classified 325 of 400 held-out reviews (**81.25%**).

| Actual label | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| Negative | 153 | 47 |
| Positive | 28 | 172 |

The model sees individual lowercased tokens weighted by their relative frequency in the review and their rarity in the training corpus. This representation discards word order. For example, `not good` and `good not` yield the same vector because the counts of `not` and `good` are identical.

## Challenge evaluation

The 25 challenge reviews and intended labels were drafted with AI for student review. The results below come from running those examples through the baseline without retraining on them.

| Category | Correct | Accuracy |
| --- | ---: | ---: |
| Negation | 3 / 5 | 60% |
| Sarcasm | 5 / 5 | 100% |
| Mixed sentiment | 4 / 5 | 80% |
| Intensity | 0 / 5 | 0% |
| Domain shift | 2 / 5 | 40% |
| Overall | 14 / 25 | 56% |

Challenge accuracy was 25.25 percentage points below the held-out score. Intensity was the hardest category. Each intensity example was labeled positive, but every one fell below the positive decision threshold. From mildest to strongest, the positive probabilities were 36.4%, 40.9%, 48.3%, 39.0%, and 35.3%. The score peaked in the middle, and “The movie was absolutely extraordinary” scored lower than “The movie was mildly enjoyable.” TF-IDF treats modifiers and opinion words as independent features, and logistic regression learns from binary labels with no strength ratings. Its positive probability reflects the learned class score, so a stronger expression need not get a higher score.

Both positive negation examples were predicted negative. For example, “I did not regret giving this movie a chance” received a 23.2% positive score. The vector retains `not` and `regret` as individual features and loses the information that `not` modifies `regret`. The five sarcasm examples all received the correct negative label, but the small sample gives little evidence that the classifier can recognize sarcasm in general.

## Improving negation

I would first add bigrams while keeping unigrams. The model could learn a separate coefficient for `not good` when the phrase appears in the training data. Bigrams increase the vocabulary, memory use, and training cost, and examples of rarer phrases will still be sparse. Negation that spans several words can still be difficult. I would compare the two models on the same held-out split and the negation challenge reviews.

## Deployment

I would use this classifier to assist human triage and keep a manual review path. In the held-out film reviews, it labeled 47 negative reviews as positive. It also missed three of the five non-movie challenge reviews, which shows how performance can change when the subject changes. With customer complaints, a sarcastic message could contain upbeat words that raise its positive score, leaving an angry customer without the human response they need. I would test on actual customer messages and measure missed complaints before relying on it to route them.
