import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class HybridRecommenderSystem:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.user_movie_matrix = None
        self.user_similarity = None
        self.content_similarity = None

    # -----------------------------
    # Load Dataset
    # -----------------------------
    def load_data(self):
        # Movie dataset
        self.movies = pd.DataFrame({
            'movieId': [1, 2, 3, 4, 5, 6, 7, 8],
            'title': [
                'Inception',
                'Interstellar',
                'The Dark Knight',
                'Titanic',
                'Avengers Endgame',
                'The Matrix',
                'Joker',
                'Doctor Strange'
            ],
            'genre': [
                'Sci-Fi Action',
                'Sci-Fi Drama',
                'Action Crime',
                'Romance Drama',
                'Action Sci-Fi',
                'Sci-Fi Action',
                'Psychological Drama',
                'Fantasy Action'
            ]
        })

        # User ratings dataset
        self.ratings = pd.DataFrame({
            'userId': [1,1,1,2,2,2,3,3,3,4,4,4,5,5],
            'movieId': [1,2,3,2,3,4,1,5,6,3,7,8,2,5],
            'rating': [5,4,5,5,4,2,4,5,5,5,4,4,3,5]
        })

    # -----------------------------
    # Collaborative Filtering
    # -----------------------------
    def collaborative_filtering(self):
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        self.user_similarity = cosine_similarity(
            self.user_movie_matrix
        )

        print("\nUser Similarity Matrix:")
        similarity_df = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

        print(similarity_df)

    # -----------------------------
    # Content-Based Filtering
    # -----------------------------
    def content_based_filtering(self):
        tfidf = TfidfVectorizer(stop_words='english')

        tfidf_matrix = tfidf.fit_transform(
            self.movies['genre']
        )

        self.content_similarity = cosine_similarity(
            tfidf_matrix
        )

    # -----------------------------
    # Hybrid Recommendation
    # -----------------------------
    def recommend_movies(self, user_id, top_n=5):
        watched_movies = self.ratings[
            self.ratings['userId'] == user_id
        ]['movieId'].tolist()

        collaborative_scores = {}

        # Similar users
        sim_scores = list(
            enumerate(self.user_similarity[user_id - 1])
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        for sim_user, score in sim_scores[1:]:
            user_movies = self.ratings[
                self.ratings['userId'] == sim_user + 1
            ]

            for _, row in user_movies.iterrows():
                movie_id = row['movieId']

                if movie_id not in watched_movies:
                    collaborative_scores[movie_id] = (
                        collaborative_scores.get(movie_id, 0)
                        + score * row['rating']
                    )

        content_scores = {}

        for movie_id in watched_movies:
            idx = self.movies[
                self.movies['movieId'] == movie_id
            ].index[0]

            similar_movies = list(
                enumerate(self.content_similarity[idx])
            )

            for movie_idx, score in similar_movies:
                rec_movie_id = self.movies.iloc[
                    movie_idx
                ]['movieId']

                if rec_movie_id not in watched_movies:
                    content_scores[rec_movie_id] = (
                        content_scores.get(rec_movie_id, 0)
                        + score
                    )

        # Normalize Scores
        scaler = MinMaxScaler()

        collab_df = pd.DataFrame(
            collaborative_scores.items(),
            columns=['movieId', 'score']
        )

        content_df = pd.DataFrame(
            content_scores.items(),
            columns=['movieId', 'score']
        )

        if not collab_df.empty:
            collab_df['score'] = scaler.fit_transform(
                collab_df[['score']]
            )

        if not content_df.empty:
            content_df['score'] = scaler.fit_transform(
                content_df[['score']]
            )

        final_scores = {}

        for _, row in collab_df.iterrows():
            final_scores[row['movieId']] = (
                0.7 * row['score']
            )

        for _, row in content_df.iterrows():
            final_scores[row['movieId']] = (
                final_scores.get(row['movieId'], 0)
                + 0.3 * row['score']
            )

        recommendations = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nRecommended Movies:")
        for movie_id, score in recommendations[:top_n]:
            movie_name = self.movies[
                self.movies['movieId'] == movie_id
            ]['title'].values[0]

            print(
                f"{movie_name} "
                f"(Recommendation Score: "
                f"{score:.2f})"
            )


# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    recommender = HybridRecommenderSystem()

    recommender.load_data()

    recommender.collaborative_filtering()

    recommender.content_based_filtering()

    user_id = int(input(
        "\nEnter User ID (1-5): "
    ))

    recommender.recommend_movies(user_id)
