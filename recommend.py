
#  recommender system

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load data from local CSV files (assuming ratings.csv and movies.csv are in the project directory)
ratings = pd.read_csv('ratings.csv', sep='\t', header=None, names=['userId', 'movieId', 'rating', 'timestamp'])
movies = pd.read_csv('movies.csv', sep='|', encoding='latin-1', header=None)
# Genres are columns 5 to 23
genre_cols = movies.iloc[:, 5:24]
genres_df = pd.read_csv('ml-100k/u.genre', sep='|', header=None, names=['genre', 'id'])
genre_names = genres_df['genre'].tolist()
movies['genres'] = genre_cols.apply(lambda x: '|'.join([genre_names[i] for i, val in enumerate(x) if val == 1]), axis=1)
movies = movies.iloc[:, [0,1,24]]  # movieId, title, genres
movies.columns = ['movieId', 'title', 'genres']
ratings.head()

# %%
movies.head()

# %%
n_ratings = len(ratings)
n_movies = ratings['movieId'].nunique()
n_users = ratings['userId'].nunique()

print(f"Number of ratings: {n_ratings}")
print(f"Number of unique movieId's: {n_movies}")
print(f"Number of unique users: {n_users}")
print(f"Average number of ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average number of ratings per movie: {round(n_ratings/n_movies, 2)}")

# %%
sns.countplot(x="rating", data=ratings, palette="viridis")
plt.title("Distribution of movie ratings", fontsize=14)
plt.show()

# %%
print(f"Mean global rating: {round(ratings['rating'].mean(),2)}.")

mean_ratings = ratings.groupby('userId')['rating'].mean()
print(f"Mean rating per user: {round(mean_ratings.mean(),2)}.")

# %%
movie_ratings = ratings.merge(movies, on='movieId')
movie_ratings['title'].value_counts()[0:10]

# %%
mean_ratings = ratings.groupby('movieId')[['rating']].mean()
lowest_rated = mean_ratings['rating'].idxmin()
movies[movies['movieId']==lowest_rated]

# %%
highest_rated = mean_ratings['rating'].idxmax()
movies[movies['movieId'] == highest_rated]

# %%
ratings[ratings['movieId']==highest_rated]

# %%
movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
movie_stats.head()

# %%
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

print(f"Average number of ratings for a given movie: {C:.2f}")
print(f"Average rating for a given movie: {m:.2f}")

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return round(bayesian_avg, 3)

# %%
lamerica = pd.Series([5, 5])
bayesian_avg(lamerica)

# %%
bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')

# %%
movie_stats = movie_stats.merge(movies[['movieId', 'title']])
movie_stats.sort_values('bayesian_avg', ascending=False).head()

# %%
movie_stats.sort_values('bayesian_avg', ascending=True).head()

# %%
movies['genres'] = movies['genres'].apply(lambda x: x.split("|"))
movies.head()

# %%
from collections import Counter

genre_frequency = Counter(g for genres in movies['genres'] for g in genres)

print(f"There are {len(genre_frequency)} genres.")

genre_frequency

# %%
print("The 5 most common genres: \n", genre_frequency.most_common(5))

# %%
genre_frequency_df = pd.DataFrame([genre_frequency]).T.reset_index()
genre_frequency_df.columns = ['genre', 'count']

sns.barplot(x='genre', y='count', data=genre_frequency_df.sort_values(by='count', ascending=False))
plt.xticks(rotation=90)
plt.show()

# %%
from scipy.sparse import csr_matrix

def create_X(df):
    """
    Generates a sparse matrix from ratings dataframe.
    
    Args:
        df: pandas dataframe containing 3 columns (userId, movieId, rating)
    
    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))
    
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))
    
    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)

# %%
X.shape

# %%
n_total = X.shape[0]*X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings/n_total
print(f"Matrix sparsity: {round(sparsity*100,2)}%")

# %%
n_ratings_per_user = X.getnnz(axis=1)
len(n_ratings_per_user)

# %%
print(f"Most active user rated {n_ratings_per_user.max()} movies.")
print(f"Least active user rated {n_ratings_per_user.min()} movies.")

# %%
n_ratings_per_movie = X.getnnz(axis=0)
len(n_ratings_per_movie)

# %%
print(f"Most rated movie has {n_ratings_per_movie.max()} ratings.")
print(f"Least rated movie has {n_ratings_per_movie.min()} ratings.")


# %%
n_total = X.shape[0]*X.shape[1]
n_ratings = X.nnz
sparsity = n_ratings/n_total
print(f"Matrix sparsity: {round(sparsity*100,2)}%")

# %%
n_ratings_per_user = X.getnnz(axis=1)
len(n_ratings_per_user)

# %%
print(f"Most active user rated {n_ratings_per_user.max()} movies.")
print(f"Least active user rated {n_ratings_per_user.min()} movies.")

# %%
n_ratings_per_movie = X.getnnz(axis=0)
len(n_ratings_per_movie)

# %%
print(f"Most rated movie has {n_ratings_per_movie.max()} ratings.")
print(f"Least rated movie has {n_ratings_per_movie.min()} ratings.")

# %%
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
sns.kdeplot(n_ratings_per_user, shade=True)
plt.xlim(0)
plt.title("Number of Ratings Per User", fontsize=14)
plt.xlabel("number of ratings per user")
plt.ylabel("density")
plt.subplot(1,2,2)
sns.kdeplot(n_ratings_per_movie, shade=True)
plt.xlim(0)
plt.title("Number of Ratings Per Movie", fontsize=14)
plt.xlabel("number of ratings per movie")
plt.ylabel("density")
plt.show()

# %%
from sklearn.neighbors import NearestNeighbors


def find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, k, metric='cosine'):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve
        metric: distance metric for kNN calculations
    
    Output: returns list of k similar movie ID's
    """
    X = X.T
    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    # use k+1 since kNN output includes the movieId of interest
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids

# %%
similar_movies = find_similar_movies(1, X, movie_mapper, movie_inv_mapper, k=10)
similar_movies

# %%
movie_titles = dict(zip(movies['movieId'], movies['title']))

movie_id = 1

similar_movies = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}:")
for i in similar_movies:
    print(movie_titles[i])

# %%
movie_id = 1

similar_movies = find_similar_movies(movie_id, X, movie_mapper, movie_inv_mapper, metric='euclidean', k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}:")
for i in similar_movies:
    print(movie_titles[i])

# %%
n_movies = movies['movieId'].nunique()
print(f"There are {n_movies} unique movies in our movies dataset.")

# %%
genres = set(g for G in movies['genres'] for g in G)

for g in genres:
    movies[g] = movies.genres.transform(lambda x: int(g in x))
    
movie_genres = movies.drop(columns=['movieId', 'title','genres'])

# %%
movie_genres.head()

# %%
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(movie_genres, movie_genres)
print(f"Dimensions of our genres cosine similarity matrix: {cosine_sim.shape}")

# %%
from fuzzywuzzy import process

def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

# %%
title = movie_finder('juminji')
title

# %%
movie_idx = dict(zip(movies['title'], list(movies.index)))
idx = movie_idx[title]
print(f"Movie index for Jumanji: {idx}")

# %%
n_recommendations=10
sim_scores = list(enumerate(cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1:(n_recommendations+1)]
similar_movies = [i[0] for i in sim_scores]

# %%
print(f"Because you watched {title}:")
movies['title'].iloc[similar_movies]

# %%
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations+1)]
    similar_movies = [i[0] for i in sim_scores]
    print(f"Because you watched {title}:")
    print(movies['title'].iloc[similar_movies])

# %%
get_content_based_recommendations('toy story', 5)

# %%
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=20, n_iter=10)
Q = svd.fit_transform(X.T)
Q.shape

# %%
movie_id = 1
similar_movies = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title}:")
for i in similar_movies:
    print(movie_titles[i])

# %%
# Improvements for accuracy

# Add necessary imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# %%
# Train-test split for evaluation
ratings_train, ratings_test = train_test_split(ratings, test_size=0.2, random_state=42)
X_train, user_mapper_train, movie_mapper_train, user_inv_mapper_train, movie_inv_mapper_train = create_X(ratings_train)
movie_features = X_train.T.tocsr()

# %%
def compute_user_means(X, user_inv_mapper):
    """
    Compute average rating per user using training data only.
    """
    user_sums = np.array(X.sum(axis=1)).ravel()
    user_counts = np.array(X.getnnz(axis=1)).ravel()
    user_means = np.divide(user_sums, user_counts, out=np.zeros_like(user_sums, dtype=float), where=user_counts != 0)
    return {user_inv_mapper[i]: user_means[i] for i in range(len(user_inv_mapper))}

user_mean_map = compute_user_means(X_train, user_inv_mapper_train)
global_mean = ratings_train['rating'].mean()

# %%
def evaluate_baseline(ratings_split, user_mean_map, global_mean):
    """
    Compare predictions that rely solely on user averages (or global mean as fallback).
    """
    predictions = []
    actuals = []
    for _, row in ratings_split.iterrows():
        predictions.append(user_mean_map.get(row['userId'], global_mean))
        actuals.append(row['rating'])
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    return rmse, mae

# %%
def get_movie_neighbors(movie_id, movie_features, model, movie_mapper, movie_inv_mapper, k=10, metric='cosine'):
    """
    Retrieve k similar movies along with similarity scores from a pre-fit kNN model.
    """
    movie_ind = movie_mapper.get(movie_id)
    if movie_ind is None:
        return []
    n_neighbors = min(k + 1, movie_features.shape[0])
    distances, indices = model.kneighbors(movie_features[movie_ind], n_neighbors=n_neighbors, return_distance=True)
    neighbors = []
    for dist, idx in zip(distances[0], indices[0]):
        neighbour_id = movie_inv_mapper[idx]
        if neighbour_id == movie_id:
            continue
        if metric == 'cosine':
            similarity = 1 - dist
        else:
            similarity = 1 / (1 + dist)
        neighbors.append((neighbour_id, similarity))
        if len(neighbors) == k:
            break
    return neighbors

# %%
def predict_rating_knn(user_id, movie_id, X, movie_features, model, user_mapper, movie_mapper, movie_inv_mapper, user_mean_map, global_mean, k=10, metric='cosine'):
    """
    Predict a user's rating for a movie using similarity-weighted item-based kNN.
    """
    if user_id not in user_mapper or movie_id not in movie_mapper:
        return user_mean_map.get(user_id, global_mean)
    
    user_ind = user_mapper[user_id]
    user_vector = X.getrow(user_ind).toarray().ravel()
    neighbours = get_movie_neighbors(movie_id, movie_features, model, movie_mapper, movie_inv_mapper, k, metric)
    
    numerator = 0.0
    denominator = 0.0
    for neighbour_movie_id, similarity in neighbours:
        neighbour_index = movie_mapper.get(neighbour_movie_id)
        rating_value = user_vector[neighbour_index]
        if rating_value > 0:
            numerator += similarity * rating_value
            denominator += abs(similarity)
    
    if denominator > 0:
        return numerator / denominator
    return user_mean_map.get(user_id, global_mean)

# %%
def evaluate_knn(X_train, ratings_test, user_mapper, movie_mapper, movie_inv_mapper, user_mean_map, global_mean, movie_features, k=10, metric='cosine'):
    """
    Compute RMSE and MAE on the held-out test set for specific k and metric.
    """
    model = NearestNeighbors(metric=metric, algorithm='brute')
    model.fit(movie_features)
    
    predictions = []
    actuals = []
    for _, row in ratings_test.iterrows():
        pred = predict_rating_knn(
            row['userId'],
            row['movieId'],
            X_train,
            movie_features,
            model,
            user_mapper,
            movie_mapper,
            movie_inv_mapper,
            user_mean_map,
            global_mean,
            k,
            metric
        )
        predictions.append(pred)
        actuals.append(row['rating'])
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    return rmse, mae

# %%
def collect_predictions(X_train, ratings_subset, movie_features, model, user_mapper, movie_mapper, movie_inv_mapper, user_mean_map, global_mean, k, metric):
    """
    Return predictions and actual values for plotting/reporting.
    """
    predictions = []
    actuals = []
    for _, row in ratings_subset.iterrows():
        pred = predict_rating_knn(
            row['userId'],
            row['movieId'],
            X_train,
            movie_features,
            model,
            user_mapper,
            movie_mapper,
            movie_inv_mapper,
            user_mean_map,
            global_mean,
            k,
            metric
        )
        predictions.append(pred)
        actuals.append(row['rating'])
    return predictions, actuals

# %%
# Hyper-parameter tuning with cross-validation
k_values = [3, 5, 10, 20, 50, 100]
metrics = ['cosine', 'euclidean', 'manhattan']

def cross_validate_knn(ratings_subset, k_values, metrics, n_splits=5):
    """
    Perform K-fold cross-validation on the training split to choose k and metric.
    """
    cv_results = []
    ratings_subset = ratings_subset.reset_index(drop=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for metric in metrics:
        for k in k_values:
            fold_rmses = []
            fold_maes = []
            for train_idx, val_idx in kf.split(ratings_subset):
                train_fold = ratings_subset.iloc[train_idx]
                val_fold = ratings_subset.iloc[val_idx]
                X_fold, user_mapper_fold, movie_mapper_fold, user_inv_mapper_fold, movie_inv_mapper_fold = create_X(train_fold)
                movie_features_fold = X_fold.T.tocsr()
                user_mean_map_fold = compute_user_means(X_fold, user_inv_mapper_fold)
                global_mean_fold = train_fold['rating'].mean()
                rmse, mae = evaluate_knn(
                    X_fold,
                    val_fold,
                    user_mapper_fold,
                    movie_mapper_fold,
                    movie_inv_mapper_fold,
                    user_mean_map_fold,
                    global_mean_fold,
                    movie_features_fold,
                    k=k,
                    metric=metric
                )
                fold_rmses.append(rmse)
                fold_maes.append(mae)
            cv_results.append({
                'metric': metric,
                'k': k,
                'rmse_mean': np.mean(fold_rmses),
                'rmse_std': np.std(fold_rmses),
                'mae_mean': np.mean(fold_maes),
                'mae_std': np.std(fold_maes)
            })
            print(f"[CV] Metric={metric}, K={k} -> RMSE: {np.mean(fold_rmses):.4f}±{np.std(fold_rmses):.4f}, MAE: {np.mean(fold_maes):.4f}±{np.std(fold_maes):.4f}")
    return pd.DataFrame(cv_results)

# %%
def run_accuracy_pipeline():
    """
    Execute the full accuracy workflow (baseline, CV, final eval, plots) and return artifacts.
    """
    baseline_rmse, baseline_mae = evaluate_baseline(ratings_test, user_mean_map, global_mean)
    print(f"Baseline (user mean) -> RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")

    cv_results_df = cross_validate_knn(ratings_train, k_values, metrics)
    cv_results_df = cv_results_df.sort_values('rmse_mean').reset_index(drop=True)
    cv_results_df.to_csv("knn_cv_results.csv", index=False)
    print("\nTop CV configurations:")
    print(cv_results_df.head())
    print("Saved all CV results to knn_cv_results.csv for reporting.")

    best_params = cv_results_df.iloc[0][['k', 'metric']].to_dict()
    best_k = int(best_params['k'])
    best_metric = best_params['metric']
    best_model = NearestNeighbors(metric=best_metric, algorithm='brute')
    best_model.fit(movie_features)

    final_rmse, final_mae = evaluate_knn(
        X_train,
        ratings_test,
        user_mapper_train,
        movie_mapper_train,
        movie_inv_mapper_train,
        user_mean_map,
        global_mean,
        movie_features,
        k=best_k,
        metric=best_metric
    )
    print(f"\nFinal evaluation -> RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    print(f"Improvement vs baseline -> ΔRMSE: {baseline_rmse - final_rmse:.4f}, ΔMAE: {baseline_mae - final_mae:.4f}")

    summary_df = pd.DataFrame(
        [
            {"model": "Baseline (user mean)", "rmse": baseline_rmse, "mae": baseline_mae},
            {"model": f"KNN tuned (k={best_k}, metric={best_metric})", "rmse": final_rmse, "mae": final_mae},
            {"model": "Improvement (baseline - tuned)", "rmse": baseline_rmse - final_rmse, "mae": baseline_mae - final_mae},
        ]
    )
    summary_df.to_csv("benchmark_summary.csv", index=False)
    print("Saved benchmark metrics to benchmark_summary.csv.")

    preds, actuals = collect_predictions(
        X_train,
        ratings_test,
        movie_features,
        best_model,
        user_mapper_train,
        movie_mapper_train,
        movie_inv_mapper_train,
        user_mean_map,
        global_mean,
        best_k,
        best_metric
    )

    # Visual diagnostics: scatter highlights discrete ratings, histogram shows error spread.
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.scatter(actuals, preds, alpha=0.3)
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Predicted vs Actual Ratings")

    plt.subplot(1,2,2)
    errors = np.array(actuals) - np.array(preds)
    sns.histplot(errors, bins=30, kde=True)
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig("accuracy_diagnostics.png", dpi=300)
    plt.show()
    print("Saved accuracy plots to accuracy_diagnostics.png.")

    return {
        'best_model': best_model,
        'best_k': best_k,
        'best_metric': best_metric,
        'cv_results': cv_results_df,
        'baseline_metrics': {'rmse': baseline_rmse, 'mae': baseline_mae},
        'final_metrics': {'rmse': final_rmse, 'mae': final_mae}
    }

results = run_accuracy_pipeline()
best_model = results['best_model']
best_k = results['best_k']
best_metric = results['best_metric']

movie_id = 1
similar_movies = get_movie_neighbors(movie_id, movie_features, best_model, movie_mapper_train, movie_inv_mapper_train, k=best_k, metric=best_metric)
movie_title = movie_titles[movie_id]

print(f"Because you watched {movie_title} (with tuned params):")
for movie_idx, _ in similar_movies:
    print(movie_titles[movie_idx])
