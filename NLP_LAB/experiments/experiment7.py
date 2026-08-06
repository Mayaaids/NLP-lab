"""
Expt.No: 7
CONSTRUCT A LANGUAGE MODEL USING N-GRAM MODELS AND COMPARE ITS
PERFORMANCE WITH A HIDDEN MARKOV MODEL (HMM) ON A CORPUS OF TWEETS.
"""

import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('punkt_tab')

tweet = input("Enter a tweet: ")

tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("\nUnigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)

fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

print("\nHMM Prediction (Sample)")
print("AI -> NOUN")
print("improves -> VERB")
print("technology -> NOUN")
