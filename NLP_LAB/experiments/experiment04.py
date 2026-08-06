"""
Expt.No: 4
BUILD AN INFORMATION RETRIEVAL SYSTEM USING CLASSICAL AND
NONCLASSICAL MODELS AND COMPARE THEIR PERFORMANCE ON A DATASET
OF SCIENTIFIC PAPERS.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

docs = []

n = int(input("Enter number of documents: "))
for i in range(n):
    docs.append(input("Enter document: "))

query = input("\nEnter search query: ")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

query_vec = vectorizer.transform([query])

scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# n_components for SVD must be less than number of features/documents
n_components = min(2, X.shape[1] - 1, X.shape[0])
n_components = max(n_components, 1)

svd = TruncatedSVD(n_components=n_components)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")
for i, s in enumerate(lsa_scores[0]):
    print("Document", i + 1, ":", round(s, 3))

best = np.argmax(lsa_scores)
print("\nMost Relevant Document:")
print(docs[best])
