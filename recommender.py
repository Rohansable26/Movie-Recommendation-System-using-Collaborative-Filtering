from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample movie ratings dataset
data = {
    'Movie1': [5, 4, 1, 0],
    'Movie2': [4, 0, 1, 1],
    'Movie3': [1, 1, 0, 5],
    'Movie4': [0, 1, 5, 4]
}

users = ['User1', 'User2', 'User3', 'User4']

df = pd.DataFrame(data, index=users)

# Calculate similarity
similarity = cosine_similarity(df)

similarity_df = pd.DataFrame(
    similarity,
    index=users,
    columns=users
)

print("User Similarity Matrix:")
print(similarity_df)

# Recommend for User1
user = 'User1'

similar_users = similarity_df[user].sort_values(
    ascending=False
)

print("\nRecommended Similar Users:")
print(similar_users)
