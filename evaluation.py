from nltk.translate.bleu_score import sentence_bleu

# Define your desired weights (example: higher weight for bi-grams)
weights = (0.25, 0.25, 0, 0)  # Weights for uni-gram, bi-gram, tri-gram, and 4-gram

# Reference and predicted texts (same as before)
reference = [["Much", "of", "personal", "computing", "is", "about", "\"", "can", "you", "top", "this", "?", "\""],]
predictions = ["A", "lot", "of", "personal", "computer", "use", "is", "," "\"", "Can", "you", "do", "better", "than", "this", "?"]

# Calculate BLEU score with weights
score = sentence_bleu(reference, predictions, weights=weights)
print(score)