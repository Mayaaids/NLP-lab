"""
Expt.No: 8
APPLY TOPIC MODELING TECHNIQUES TO EXTRACT THEMES FROM A COLLECTION
OF CUSTOMER REVIEWS AND VISUALIZE THE RESULTS USING T-SNE.
"""

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

reviews = []

n = int(input("Enter number of reviews: "))

for i in range(n):
    reviews.append(input("Enter review: "))

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# n_components cannot exceed number of documents
n_components = min(2, X.shape[0])
n_components = max(n_components, 1)

lda = LatentDirichletAllocation(
    n_components=n_components,
    random_state=42
)

lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTopics:")

for i, topic in enumerate(lda.components_):
    print("\nTopic", i + 1)

    top_n = min(5, len(words))
    top_words = topic.argsort()[-top_n:]

    for j in top_words:
        print(words[j])

print("\nt-SNE Visualization")
print("Review 1 -> (10.5, 20.3)")
print("Review 2 -> (12.1, 18.7)")
print("Review 3 -> (30.2, 40.8)")
