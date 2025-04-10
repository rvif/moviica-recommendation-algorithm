import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from joblib import dump, load
import os
import time

# Control flag for matrix regeneration
REGENERATE_MATRICES = False

# Create a cache for recommendations
recommendation_cache = {}

# Track script execution time
start_time = time.time()

# Check if matrix files exist and load them
try:
    if os.path.exists('plot_matrix.joblib') and not REGENERATE_MATRICES:
        print("Loading existing matrices from disk...")
        plot_matrix = load('plot_matrix.joblib')
        cast_matrix = load('cast_matrix.joblib')
        studio_matrix = load('studio_matrix.joblib')
        title_matrix = load('title_matrix.joblib')
        
        # Load the associated movies dataframe with features
        movies = load('processed_movies.joblib')
        
        # Recreate lookup dictionaries
        title_to_idx = {title.lower(): i for i, title in enumerate(movies['title'])}
        id_to_idx = {id_val: i for i, id_val in enumerate(movies['id'])}
        
        print(f"Successfully loaded existing matrix files in {time.time() - start_time:.2f} seconds")
        matrices_loaded = True
    else:
        matrices_loaded = False
        if REGENERATE_MATRICES:
            print("Regenerating matrices as requested...")
        else:
            print("Matrix files don't exist yet - will generate them")
except Exception as e:
    print(f"Error loading matrix files: {e}")
    matrices_loaded = False

# Only run data processing if matrices weren't loaded
if not matrices_loaded:
    # Load datasets
    print("Loading and processing datasets...")
    movies_df = pd.read_csv('dataset/movies_metadata.csv', low_memory=False)
    credits_df = pd.read_csv('dataset/credits.csv')
    keywords_df = pd.read_csv('dataset/keywords.csv')

    # Fix bad ID values
    movies_df = movies_df[movies_df['id'].apply(lambda x: str(x).isdigit())]
    movies_df['id'] = movies_df['id'].astype(int)
    credits_df['id'] = credits_df['id'].astype(int)
    keywords_df['id'] = keywords_df['id'].astype(int)

    # Merge datasets
    movies = movies_df.merge(credits_df, on='id').merge(keywords_df, on='id')
    movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'production_companies', 'poster_path']].dropna(subset=['overview', 'poster_path'])

    # Remove duplicates by title
    movies = movies.drop_duplicates(subset=['title'])

    print(f"Merged dataset shape: {movies.shape}")

    # --- Feature Extraction Functions ---
    def extract_names(json_str):
        """Extract name values from JSON string."""
        try:
            data = ast.literal_eval(json_str)
            return ' '.join([obj['name'] for obj in data])
        except:
            return ''

    def extract_production_companies(json_str):
        """Extract production company names."""
        try:
            companies = ast.literal_eval(json_str)
            # Give higher weight to production company by repeating 3 times
            return ' '.join([company['name'] for company in companies] * 3)
        except:
            return ''

    def extract_cast_names(cast_str, top_n=5):
        """Extract cast names with preference for main characters."""
        try:
            cast_list = ast.literal_eval(cast_str)
            # Sort by order (lower order = more important role)
            sorted_cast = sorted(cast_list, key=lambda x: x.get('order', 999))
            # Extract names with weights (main characters get repeated for emphasis)
            names = []
            for i, member in enumerate(sorted_cast[:top_n]):
                weight = max(1, (top_n - i))  # More weight for earlier cast members
                names.extend([member['name']] * weight)
            return ' '.join(names)
        except:
            return ''

    def extract_franchise_keywords(overview, genres, keywords):
        """Extract genre and franchise-specific keywords to improve detection."""
        # Common franchise indicators and popular genres
        franchise_indicators = [
            'sequel', 'prequel', 'trilogy', 'series', 'universe', 
            'marvel', 'dc comics', 'disney', 'pixar', 'dreamworks',
            'star wars', 'harry potter', 'lord of the rings', 'hobbit',
            'batman', 'spider-man', 'avengers', 'fast furious',
            'james bond', 'mission impossible', 'transformers', 
            'jurassic', 'terminator', 'alien', 'predator',
            'superhero', 'animation', 'horror', 'thriller'
        ]
        
        combined_text = (overview + ' ' + genres + ' ' + keywords).lower()
        matched_keywords = []
        
        for keyword in franchise_indicators:
            if keyword in combined_text:
                # Add multiple times for emphasis
                matched_keywords.extend([keyword] * 5)
        
        return ' '.join(matched_keywords)

    def simplify_title(title):
        """Extract core words from title for better matching."""
        # Remove common articles, prefixes, etc.
        simplified = re.sub(r'^(the|a|an) ', '', title.lower())
        # Remove punctuation and numbers
        simplified = re.sub(r'[^\w\s]', ' ', simplified)
        simplified = re.sub(r'\d+', '', simplified)
        # Remove common franchise indicators
        simplified = re.sub(r'part [\w]+', '', simplified)
        simplified = re.sub(r'episode [\w]+', '', simplified)
        # Normalize whitespace
        simplified = ' '.join(simplified.split())
        return simplified

    # --- Apply Feature Extraction ---
    print("Extracting features...")
    # Extract key features
    movies['genres_text'] = movies['genres'].apply(extract_names)
    movies['keywords_text'] = movies['keywords'].apply(extract_names)
    movies['studio'] = movies['production_companies'].apply(extract_production_companies)
    movies['cast_features'] = movies['cast'].apply(lambda x: extract_cast_names(x))
    movies['simplified_title'] = movies['title'].apply(simplify_title)

    # Add franchise-specific feature enhancement for all movies
    movies['franchise_keywords'] = movies.apply(
        lambda x: extract_franchise_keywords(
            x['overview'], x['genres_text'], x['keywords_text']), 
        axis=1
    )

    # --- Create a title index for faster lookup ---
    title_to_idx = {title.lower(): i for i, title in enumerate(movies['title'])}
    id_to_idx = {id_val: i for i, id_val in enumerate(movies['id'])}

    # --- Memory-efficient TF-IDF vectorization ---
    print("Generating TF-IDF vectors...")

    # 1. Plot & Content-based features (added franchise keywords)
    content_features = (
        movies['overview'] + ' ' + 
        movies['genres_text'] + ' ' + 
        movies['keywords_text'] + ' ' + 
        movies['simplified_title'] * 2 + ' ' +
        movies['franchise_keywords']
    )
    plot_vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    plot_matrix = plot_vectorizer.fit_transform(content_features)
    print(f"Plot matrix shape: {plot_matrix.shape}")

    # 2. Cast features
    cast_vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
    cast_matrix = cast_vectorizer.fit_transform(movies['cast_features'])
    print(f"Cast matrix shape: {cast_matrix.shape}")

    # 3. Studio similarity
    studio_vectorizer = TfidfVectorizer(stop_words='english')
    studio_matrix = studio_vectorizer.fit_transform(movies['studio'])
    print(f"Studio matrix shape: {studio_matrix.shape}")

    # 4. Title similarity with n-gram tuning for franchise detection
    title_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    title_matrix = title_vectorizer.fit_transform(movies['simplified_title'])
    print(f"Title matrix shape: {title_matrix.shape}")

    # Save matrices and processed data
    print("Saving matrices and processed data to disk...")
    dump(plot_matrix, 'plot_matrix.joblib')
    dump(cast_matrix, 'cast_matrix.joblib')
    dump(studio_matrix, 'studio_matrix.joblib')
    dump(title_matrix, 'title_matrix.joblib')
    dump(movies, 'processed_movies.joblib')
    print("Matrices and data saved")

# Pre-compute similarity indices for faster lookups
similarity_indices = {}

# Find similar movies based on a specific feature matrix
def find_similar_movies(idx, matrix, matrix_name=None, top_n=50, exclude_ids=None):
    # Check if we already computed this result
    cache_key = f"{idx}_{matrix_name}_{top_n}"
    if cache_key in similarity_indices:
        return similarity_indices[cache_key]
    
    # Get the vector for the movie
    movie_vector = matrix[idx:idx+1]
    
    # Calculate similarity with all other movies - using batch processing for efficiency
    batch_size = 1000
    num_movies = matrix.shape[0]
    similarities = np.zeros(num_movies)
    
    for start in range(0, num_movies, batch_size):
        end = min(start + batch_size, num_movies)
        batch_similarities = cosine_similarity(movie_vector, matrix[start:end]).flatten()
        similarities[start:end] = batch_similarities
    
    # Create index-similarity pairs and sort
    pairs = [(i, sim) for i, sim in enumerate(similarities)]
    if exclude_ids:
        pairs = [(i, sim) for i, sim in pairs if i not in exclude_ids]
    
    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Get result and cache it
    result = [(i, sim) for i, sim in pairs if i != idx][:top_n]
    similarity_indices[cache_key] = result
    
    return result

# Main recommendation function using weighted combination of similarities
def get_recommendations(title, top_n=10, plot_weight=0.4, cast_weight=0.3, studio_weight=0.2, title_weight=0.1):
    # Check cache first
    cache_key = f"{title}_{top_n}_{plot_weight}_{cast_weight}_{studio_weight}_{title_weight}"
    if cache_key in recommendation_cache:
        print(f"Using cached recommendations for '{title}'")
        return recommendation_cache[cache_key]
    
    start = time.time()
    
    # Find movie index
    title_lower = title.lower()
    if title_lower in title_to_idx:
        idx = title_to_idx[title_lower]
    else:
        # Try partial matching
        partial_matches = [movie_title for movie_title in title_to_idx.keys() 
                          if title_lower in movie_title]
        if not partial_matches:
            return f"No movie found with title: {title}"
        
        matched_title = partial_matches[0]
        idx = title_to_idx[matched_title]
        print(f"Exact match not found. Using similar title: {movies.iloc[idx]['title']}")
    
    # Get movie info
    movie_info = movies.iloc[idx]
    movie_id = movie_info['id']
    
    # Identify movies to exclude (the movie itself)
    exclude_ids = {idx}
    
    # Get similar movies from each feature dimension
    print("Finding similar movies...")
    plot_similar = find_similar_movies(idx, plot_matrix, "plot", top_n=100, exclude_ids=exclude_ids)
    cast_similar = find_similar_movies(idx, cast_matrix, "cast", top_n=100, exclude_ids=exclude_ids)
    studio_similar = find_similar_movies(idx, studio_matrix, "studio", top_n=100, exclude_ids=exclude_ids)
    title_similar = find_similar_movies(idx, title_matrix, "title", top_n=100, exclude_ids=exclude_ids)
    
    # Create a scoring dictionary
    scores = {}
    
    # Add weighted scores from each similarity component
    for i, score in plot_similar:
        scores[i] = scores.get(i, 0) + plot_weight * score
    
    for i, score in cast_similar:
        scores[i] = scores.get(i, 0) + cast_weight * score
    
    for i, score in studio_similar:
        scores[i] = scores.get(i, 0) + studio_weight * score
    
    for i, score in title_similar:
        scores[i] = scores.get(i, 0) + title_weight * score
    
    # Recognize franchise patterns in all movies (not just Batman)
    # Extract franchise markers from the source movie
    source_title = movie_info['simplified_title'].lower()
    source_franchise = movies.iloc[idx]['franchise_keywords']
    
    # Give a bonus for movies in the same franchise
    if source_franchise:
        # Extract key franchise words
        franchise_terms = set(source_franchise.split())
        
        for i in scores:
            target_franchise = movies.iloc[i]['franchise_keywords']
            target_title = movies.iloc[i]['simplified_title'].lower()
            
            # Check for franchise matches
            if target_franchise:
                target_terms = set(target_franchise.split())
                # If there's significant overlap in franchise terms
                if len(franchise_terms & target_terms) >= 2:
                    scores[i] *= 1.3  # 30% bonus
                    
            # Check for title similarity indicating franchise
            # (like "Star Wars" and "Star Wars: The Force Awakens")
            if (len(source_title) > 4 and source_title in target_title) or \
               (len(target_title) > 4 and target_title in source_title):
                scores[i] *= 1.25  # 25% bonus
    
    # Sort by combined score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create lookup dictionaries for quick access to individual similarities
    plot_sim_dict = {i: score for i, score in plot_similar}
    cast_sim_dict = {i: score for i, score in cast_similar}
    studio_sim_dict = {i: score for i, score in studio_similar}
    title_sim_dict = {i: score for i, score in title_similar}
    
    # Return recommendation details
    recommendations = []
    for movie_idx, combined_score in sorted_scores:
        # Get individual similarity scores using dictionaries for O(1) lookup
        plot_sim = plot_sim_dict.get(movie_idx, 0)
        cast_sim = cast_sim_dict.get(movie_idx, 0)
        studio_sim = studio_sim_dict.get(movie_idx, 0)
        title_sim = title_sim_dict.get(movie_idx, 0)
        
        movie = movies.iloc[movie_idx]
        recommendations.append({
            'id': int(movie['id']),
            'title': movie['title'],
            'similarity_score': round(combined_score, 3),
            'plot_similarity': round(plot_sim, 3),
            'cast_similarity': round(cast_sim, 3),
            'studio_similarity': round(studio_sim, 3),
            'title_similarity': round(title_sim, 3)
        })
    
    # Cache the results
    recommendation_cache[cache_key] = recommendations
    
    print(f"Recommendation process completed in {time.time() - start:.2f} seconds")
    return recommendations

# Specialized recommendation functions with different weights
def get_plot_recommendations(title, top_n=10):
    """Recommendations based primarily on plot and content."""
    return get_recommendations(title, top_n, plot_weight=0.6, cast_weight=0.2, studio_weight=0.1, title_weight=0.1)

def get_cast_recommendations(title, top_n=10):
    """Recommendations based primarily on cast similarity."""
    return get_recommendations(title, top_n, plot_weight=0.15, cast_weight=0.7, studio_weight=0.1, title_weight=0.05)

def get_franchise_recommendations(title, top_n=10):
    """Recommendations prioritizing same franchise/studio."""
    return get_recommendations(title, top_n, plot_weight=0.2, cast_weight=0.1, studio_weight=0.4, title_weight=0.3)

# Example
def test_recommendations(movie_name):
    print(f"\n--- Plot-focused recommendations for '{movie_name}' ---")
    plot_recs = get_plot_recommendations(movie_name)
    for i, rec in enumerate(plot_recs):
        print(f"{i+1}. {rec['title']} (TMDB ID: {rec['id']}) - Score: {rec['similarity_score']}")

    print(f"\n--- Cast-focused recommendations for '{movie_name}' ---")
    cast_recs = get_cast_recommendations(movie_name)
    for i, rec in enumerate(cast_recs):
        print(f"{i+1}. {rec['title']} (TMDB ID: {rec['id']}) - Score: {rec['similarity_score']}")

    print(f"\n--- Franchise/Studio recommendations for '{movie_name}' ---")
    franchise_recs = get_franchise_recommendations(movie_name)
    for i, rec in enumerate(franchise_recs):
        print(f"{i+1}. {rec['title']} (TMDB ID: {rec['id']}) - Score: {rec['similarity_score']}")

# Test with different movies
test_movies = ["The Dark Knight", "Despicable Me", "The Godfather", "The Conjuring"]
for movie in test_movies:
    start_test = time.time()
    test_recommendations(movie)
    print(f"Test execution time for '{movie}': {time.time() - start_test:.2f} seconds")

print(f"Overall script execution time: {time.time() - start_time:.2f} seconds")