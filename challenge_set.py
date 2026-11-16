"""Evaluate five labeled reviews in each challenge category."""

from sentiment_analyzer import LABELS, train_baseline


# AI-drafted examples and labels for student review.
REVIEWS = {
    "Negation": [
        ("The acting was not convincing, even in the quiet scenes.", 'negative'),
        ("I did not regret giving this movie a chance.", 'positive'),
        ("The ending never gave me a reason to care.", 'negative'),
        ("The jokes weren't funny enough to save the movie.", 'negative'),
        ("I wouldn't call it a masterpiece, but I had a good time.", 'positive'),
    ],
    "Sarcasm": [
        ("What a treat to hear the same joke six times.", 'negative'),
        ("Brilliant casting. Everyone sounded like they were reading the grocery list.", 'negative'),
        ("I loved the suspense of wondering whether the plot would ever start.", 'negative'),
        ("Fantastic editing. I felt like three different films had been shuffled together.", 'negative'),
        ("The best part was the credits. I finally got to stretch my legs.", 'negative'),
    ],
    "Mixed sentiment": [
        ("The soundtrack was beautiful, but the story was so thin I lost interest.", 'negative'),
        ("The effects looked cheap, but the characters made every scene worth watching.", 'positive'),
        ("I liked the lead actor, but the dull pacing made the movie a chore.", 'negative'),
        ("The first act dragged, yet the final hour was funny and genuinely moving.", 'positive'),
        ("Wonderful costumes and sets, but the script gave the actors nothing to work with.", 'negative'),
    ],
    "Intensity": [
        ("The movie was mildly enjoyable.", 'positive'),
        ("The movie was enjoyable.", 'positive'),
        ("The movie was quite enjoyable.", 'positive'),
        ("The movie was incredibly enjoyable.", 'positive'),
        ("The movie was absolutely extraordinary.", 'positive'),
    ],
    "Domain shift": [
        ("My coffee was fresh and the cashier thoughtfully made it again when I spilled it.", 'positive'),
        ("The zipper jammed on the first day and the stitching came undone by the weekend.", 'negative'),
        ("The charger fit my phone perfectly and brought it back to full power in an hour.", 'positive'),
        ("The soup arrived cold and the spoon still had food stuck to it.", 'negative'),
        ("The hiking shoes stayed dry on a muddy trail and were comfortable the entire morning.", 'positive'),
    ],
}


def main():
    missing = []
    for category, entries in REVIEWS.items():
        if len(entries) != 5:
            raise SystemExit(f'{category}: enter exactly five reviews.')
        for index, (review, label) in enumerate(entries, start=1):
            if not review.strip() or label not in LABELS:
                missing.append(f'{category} #{index}')
    if missing:
        print('Enter your review text and a positive or negative label for:')
        print('\n'.join(missing))
        return

    model, _, _, _ = train_baseline()
    total_correct = 0
    total_reviews = 0
    for category, entries in REVIEWS.items():
        texts, labels = zip(*entries)
        predictions = model.predict(texts)
        positive_column = list(model.classes_).index('positive')
        positive_probabilities = model.predict_proba(texts)[:, positive_column]
        correct = sum(actual == predicted for actual, predicted in zip(labels, predictions))
        total_correct += correct
        total_reviews += len(entries)
        print(f'\n{category}: {correct}/{len(entries)} ({correct / len(entries):.0%})')
        for index, (actual, predicted, probability) in enumerate(
            zip(labels, predictions, positive_probabilities), start=1
        ):
            print(f'  {index}. Label: {actual} | Prediction: {predicted} '
                  f'| P(positive): {probability:.1%}')
    print(f'\nOverall challenge accuracy: {total_correct}/{total_reviews} '
          f'({total_correct / total_reviews:.0%})')


if __name__ == '__main__':
    main()
