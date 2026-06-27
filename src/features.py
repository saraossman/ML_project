"""
Feature engineering for Airbnb listings and reviews.
Operates on the cleaned listings DataFrame from data/processed/listings_clean.parquet 
and reviews from data/raw/reviews.csv.gz.
"""

import ast
import numpy as np
import pandas as pd
from collections import Counter
from math import atan2, cos, radians, sin, sqrt
from sklearn.decomposition import PCA


SELECTED_AMENITIES = [
    'Dishwasher',
    'Self check-in',
    'TV',
    'Washer',
    'Private entrance',
    'Elevator',
    'Dedicated workspace',
    'Long term stays allowed',
    'Free street parking',
    'Shampoo',
    'Toaster',
    'Iron',
]

BERLIN_LANDMARKS = {
    'dist_city_center': (52.5200, 13.4050),  # Alexanderplatz
    'dist_kreuzberg':   (52.4988, 13.4028),  # Kreuzberg
}

COLS_TO_DROP_FINAL = [
    'amenities',
    'amenities_parsed',
    'name',
    'picture_url',
    'host_verifications',
]


# Categorical data encoding

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode room_type, neighbourhood_cleansed, and host_response_time."""
    df = pd.get_dummies(df, columns=['room_type'], drop_first=True)
    df = pd.get_dummies(df, columns=['neighbourhood_cleansed'], drop_first=True)
    df = pd.get_dummies(df, columns=['host_response_time'], drop_first=True)
    return df


# Amenities

def _parse_amenities(text) -> list:
    """Parse amenities from raw string representation of a list."""
    try:
        return ast.literal_eval(text)
    except Exception:
        return []


def build_amenity_features(
    df: pd.DataFrame,
    selected_amenities: list = SELECTED_AMENITIES,
) -> pd.DataFrame:
    """
    Parse the amenities column and create:
      - One binary value per selected amenity
      - amenities_count: total number of amenities listed
    """
    df['amenities_parsed'] = df['amenities'].apply(_parse_amenities)

    for amenity in selected_amenities:
        col = 'amenity_' + amenity.lower().replace(' ', '_').replace('-', '_')
        df[col] = df['amenities_parsed'].apply(lambda x: int(amenity in x))

    df['amenities_count'] = df['amenities_parsed'].apply(len)
    return df


# Host experience

def extract_host_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Convert host_since date to host_tenure_days. Drop raw date column."""
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['host_tenure_days'] = (pd.Timestamp.today() - df['host_since']).dt.days
    df.drop(columns=['host_since'], inplace=True)
    return df


# Text features

def add_description_length(df: pd.DataFrame) -> pd.DataFrame:
    """Add character count of the listing description."""
    df['description_length'] = df['description'].apply(len)
    return df


def add_description_sentiment(df: pd.DataFrame, analyzer) -> pd.DataFrame:
    """
    Add VADER compound sentiment score for the listing description.

    Args:
        df: listings DataFrame with a 'description' column.
        analyzer: instantiated SentimentIntensityAnalyzer from vaderSentiment.
    """
    df['description_sentiment'] = df['description'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if x else 0
    )
    return df


def build_description_embeddings(
    df: pd.DataFrame,
    model,
    n_components: int = 50,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Encode listing descriptions with a SentenceTransformer and reduce with PCA.

    Args:
        df: listings DataFrame with a 'description' column.
        model: SentenceTransformer model.
        n_components: number of PCA components to retain.
        batch_size: encoding batch size.

    Returns:
        df: listings DataFrame with description embeddings columns added.
        embeddings_raw: full-dimensional embeddings array (shape: n x D).
        embeddings_pca: PCA-reduced embeddings array (shape: n x n_components).
    """
    embeddings_raw = model.encode(
        df['description'].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )

    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_raw)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    emb_cols = [f'desc_emb_{i}' for i in range(n_components)]
    df_emb = pd.DataFrame(embeddings_pca, columns=emb_cols, index=df.index)
    df = pd.concat([df, df_emb], axis=1)

    return df, embeddings_raw, embeddings_pca


# Spatial features

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return shortest distance in km between two (lat, lon) points on a sphere"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def add_spatial_features(
    df: pd.DataFrame,
    landmarks: dict = BERLIN_LANDMARKS,
) -> pd.DataFrame:
    """
    Add km distance from each listing to each landmark.

    Args:
        df: listings DataFrame with 'latitude' and 'longitude' columns.
        landmarks: dict of {column_name: (lat, lon)}.
    """
    for col_name, (lat, lon) in landmarks.items():
        df[col_name] = df.apply(
            lambda row: _haversine(row['latitude'], row['longitude'], lat, lon),
            axis=1,
        )
    return df


# Review sentiment

def build_review_sentiment(
    reviews: pd.DataFrame,
    valid_ids: np.ndarray,
    analyzer,
) -> pd.DataFrame:
    """
    Compute per-listing VADER sentiment stats from review comments.

    Args:
        reviews: raw reviews DataFrame with 'listing_id' and 'comments' columns.
        valid_ids: array of listing IDs to keep (from cleaned listings).
        analyzer: SentimentIntensityAnalyzer.

    Returns:
        DataFrame with columns: listing_id, review_sentiment_mean, review_sentiment_std.
    """
    reviews_filtered = reviews[reviews['listing_id'].isin(valid_ids)].copy()
    reviews_filtered['sentiment'] = reviews_filtered['comments'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )
    review_features = reviews_filtered.groupby('listing_id').agg(
        review_sentiment_mean=('sentiment', 'mean'),
        review_sentiment_std=('sentiment', 'std'),
    ).reset_index()
    return review_features


def merge_review_features(
    listings: pd.DataFrame,
    review_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    add review sentiment features onto listings.
    Listings with no reviews get 0 for both sentiment columns.
    """
    listings = listings.merge(
        review_features, left_on='id', right_on='listing_id', how='left'
    )
    listings['review_sentiment_mean'] = listings['review_sentiment_mean'].fillna(0)
    listings['review_sentiment_std'] = listings['review_sentiment_std'].fillna(0)
    listings.drop(columns=['listing_id'], inplace=True)
    return listings


# Cleanup

def drop_unprocessed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that have been fully processed or are not useful for modelling."""
    return df.drop(columns=COLS_TO_DROP_FINAL)


# Master function 

def build_all_features(
    listings: pd.DataFrame,
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Full feature engineering pipeline.

    Args:
        listings: cleaned listings DataFrame (output of clean_listings).
        reviews: raw reviews DataFrame loaded from data/raw/reviews.csv.gz.

    Returns:
        listings: fully featured DataFrame.
        embeddings_raw: raw sentence embeddings (saved separately).
        embeddings_pca: PCA-reduced embeddings (saved separately).
    """
    from sentence_transformers import SentenceTransformer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    listings = encode_categoricals(listings)
    listings = build_amenity_features(listings)
    listings = extract_host_tenure(listings)
    listings = add_description_length(listings)
    listings = add_description_sentiment(listings, analyzer)
    listings, embeddings_raw, embeddings_pca = build_description_embeddings(listings, embed_model)
    listings = add_spatial_features(listings)

    review_features = build_review_sentiment(reviews, listings['id'].unique(), analyzer)
    listings = merge_review_features(listings, review_features)

    listings = drop_unprocessed_columns(listings)
    listings.drop(columns=['description'], inplace=True)

    return listings, embeddings_raw, embeddings_pca
