"""
Feature engineering for Airbnb listings and reviews.

Inputs:

  data/processed/listings_clean.parquet
  data/raw/reviews.csv.gz

Outputs:

* Two encoding variants are produced: one-hot (OHE) and target encoding (TE).
  data/processed/train_hotenc.parquet
  data/processed/test_hotenc.parquet
  data/processed/train_targetenc.parquet
  data/processed/test_targetenc.parquet
  outputs/models/pca_model.joblib
  outputs/models/train_embeddings_raw.npy
  outputs/models/test_embeddings_raw.npy
  outputs/models/selected_amenities.json
"""

import ast
import json
import re

import joblib
import numpy as np
import pandas as pd
from collections import Counter
from math import atan2, cos, radians, sin, sqrt
from scipy.stats import pointbiserialr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Fixed reference date = Berlin dataset scrape date.
SNAPSHOT_DATE = pd.Timestamp('2025-09-23')

CAP_PERCENTILE   = 0.99
RANDOM_STATE     = 42
TEST_SIZE        = 0.20

PCA_N_COMPONENTS         = 50
TOP_N_AMENITIES          = 20
MIN_AMENITY_FREQ         = 0.02   # amenity must appear in ≥2% of train listings
MAX_REVIEWS_PER_LISTING  = 10     # most-recent reviews to run sentiment on
REVIEW_HALF_LIFE_DAYS    = 365
TARGET_SMOOTHING         = 10

BERLIN_LANDMARKS = {
    'dist_city_center': (52.5200, 13.4050),   # Alexanderplatz
    'dist_kreuzberg':   (52.4988, 13.4028),   # Kreuzberg
}

# Categorical columns that need encoding
CAT_COLS = ['room_type', 'neighbourhood_cleansed', 'host_response_time', 'property_type']

# Columns to drop before text processing
_DROP_PRE_TEXT = ['amenities', 'amenities_parsed','name', 'picture_url', 'host_verifications', 'host_name']

# Columns to drop after text processing
_DROP_POST_TEXT = ['description']

# Any other leftover columns to clean up
_DROP_MISC = ['estimated_revenue_l365d']

# Cache path for review sentiment scores (avoids rerunning slow inference)
_REVIEW_CACHE_PATH = 'outputs/models/review_sentiments_cache.parquet'


# 1. Price cap + train/test split

def split_and_cap(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    cap_percentile: float = CAP_PERCENTILE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Split into train/test, then derive the price cap from the training set only.

    Capping from the full dataset would leak test-set into the
    threshold used to filter rows.

    Returns:
    train : pd.DataFrame
    test  : pd.DataFrame
    cap   : float  (the cap value)
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    cap = train['price'].quantile(cap_percentile)
    train = train[train['price'] <= cap].copy()
    test  = test[test['price']  <= cap].copy()

    train['log_price'] = np.log1p(train['price'])
    test['log_price']  = np.log1p(test['price'])

    print(f"  Price cap (train {cap_percentile:.0%}): €{cap:.0f}")
    print(f"  Train rows: {len(train):,}  |  Test rows: {len(test):,}")
    return train, test, cap


#2. Amenitiese

def _parse_amenities(text) -> list:
    try:
        return ast.literal_eval(text)
    except Exception:
        return []


def select_amenities_biserial(
    train: pd.DataFrame,
    top_n: int = TOP_N_AMENITIES,
    min_freq: float = MIN_AMENITY_FREQ,
) -> list:
    """
    Select amenities by point-biserial correlation with log_price.
    Only amenities present in ≥ min_freq of *training* listings are considered

    Parameters:
    train : training DataFrame
    top_n: how many amenities to keep
    min_freq: minimum fraction of listings that must have the amenity

    Returns
    list of amenity name strings
    """
    parsed = train['amenities'].apply(_parse_amenities)
    all_counts = Counter(a for sub in parsed for a in sub)
    min_count  = len(train) * min_freq

    correlations = {}
    for amenity, count in all_counts.items():
        if count < min_count:
            continue
        flag = parsed.apply(lambda x: int(amenity in x))
        n_pos = flag.sum()
        if n_pos == 0 or n_pos == len(train):
            continue   # constant features are dropped
        corr, _ = pointbiserialr(flag, train['log_price'])
        correlations[amenity] = abs(corr)

    top = sorted(correlations, key=correlations.get, reverse=True)[:top_n]
    print(f"  Selected {len(top)} amenities. Top 5: {top[:5]}")
    return top


def build_amenity_features(df: pd.DataFrame, selected_amenities: list) -> pd.DataFrame:
    """
    Create:
      - One binary flag per selected amenity  (amenity_<name>)
      - amenities_count: total number of amenities listed
    """
    df = df.copy()
    df['amenities_parsed'] = df['amenities'].apply(_parse_amenities)
    for amenity in selected_amenities:
        col = 'amenity_' + re.sub(r'[^a-z0-9]', '_', amenity.lower()).strip('_')
        df[col] = df['amenities_parsed'].apply(lambda x, a=amenity: int(a in x))
    df['amenities_count'] = df['amenities_parsed'].apply(len)
    return df


#3. Host tenure (fixed date)

def extract_host_tenure(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
) -> pd.DataFrame:
    """
    Compute host_tenure_days relative to the dataset snapshot date.
    """
    df = df.copy()
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['host_tenure_days'] = (snapshot_date - df['host_since']).dt.days
    df.drop(columns=['host_since'], inplace=True)
    return df


# 4. Text features 

def add_description_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add description_word_count
    """
    df = df.copy()
    df['description_word_count'] = df['description'].apply(
        lambda x: len(str(x).split()) if x else 0
    )
    return df


def load_sentiment_pipeline(
    model_name: str = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
):

    from transformers import pipeline as hf_pipeline

    print(f"  Loading sentiment model: {model_name}")
    pipe = hf_pipeline(
        'sentiment-analysis',
        model=model_name,
        device=-1,          # CPU set to 0 for a GPU
        batch_size=64,
        truncation=True,
        max_length=512,
    )
    return pipe


def _label_to_score(label: str) -> float:
    """Map positive to +1.0, neutral to 0.0, negative to -1.0."""
    return {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}.get(label.lower(), 0.0)


def compute_sentiment_scores(texts: list, pipe) -> np.ndarray:
    """
    Run the sentiment pipeline on a list of texts.
    Texts are truncated to 1000 characters before tokenisation.
    Returns an array of scores in [-1, 1]
    """
    clean = [str(t)[:1000] if t else "" for t in texts]
    results = pipe(clean)
    return np.array([_label_to_score(r['label']) for r in results])


def add_description_sentiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pipe,
) -> tuple:
    """
    Add multilingual description_sentiment to both splits
    """
    print("    Train descriptions...")
    train = train.copy()
    train['description_sentiment'] = compute_sentiment_scores(
        train['description'].tolist(), pipe
    )
    print("    Test descriptions...")
    test = test.copy()
    test['description_sentiment'] = compute_sentiment_scores(
        test['description'].tolist(), pipe
    )
    return train, test


#5. Embeddings + PCA (fit only on train)

def build_description_embeddings_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    embed_model,
    n_components: int = PCA_N_COMPONENTS,
    batch_size: int = 64,
) -> tuple:
    """
    Encode descriptions with a SentenceTransformer, then reduce with PCA.

    Returns
    train, test : DataFrames with desc_emb_n columns
    pca  : fitted PCA object
    train_raw, test_raw : full-dimensional embedding arrays
    """
    print("    Encoding train descriptions...")
    train_raw = embed_model.encode(
        train['description'].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )
    print("    Encoding test descriptions...")
    test_raw = embed_model.encode(
        test['description'].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )

    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(train_raw)   # fit on train
    test_pca  = pca.transform(test_raw)         # apply to test
    print(f"    PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    emb_cols = [f'desc_emb_{i}' for i in range(n_components)]
    train = pd.concat(
        [train, pd.DataFrame(train_pca, columns=emb_cols, index=train.index)], axis=1
    )
    test = pd.concat(
        [test,  pd.DataFrame(test_pca,  columns=emb_cols, index=test.index)],  axis=1
    )
    return train, test, pca, train_raw, test_raw


#6. Spatial features

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """sphere distance in km between two (lat, lon) points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def add_spatial_features(
    df: pd.DataFrame,
    landmarks: dict = BERLIN_LANDMARKS,
) -> pd.DataFrame:
    """Add km distance from each listing to each Berlin landmark."""
    df = df.copy()
    for col_name, (lat, lon) in landmarks.items():
        df[col_name] = df.apply(
            lambda row, la=lat, lo=lon: _haversine(
                row['latitude'], row['longitude'], la, lo
            ),
            axis=1,
        )
    return df


#7. Review sentiment (VADER + recency-weighted)

def build_review_sentiment(
    reviews: pd.DataFrame,
    valid_ids,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
    max_per_listing: int = MAX_REVIEWS_PER_LISTING,
    half_life_days: float = REVIEW_HALF_LIFE_DAYS,
    cache_path=None,
) -> pd.DataFrame:
    """
    Compute per-listing review sentiment features using VADER.

    Features
    review_sentiment_mean   : plain mean of VADER compound scores
    review_sentiment_min       : most negative review (problem signal)
    review_sentiment_std   : variability across reviews
    review_sentiment_recency_wt : exponentially weighted mean (recent = higher weight)
    """
    from pathlib import Path
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Cache
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"    Loading cached review sentiments from {cache_path}")
            return pd.read_parquet(cache_path)

    # Compute
    reviews = reviews[reviews['listing_id'].isin(valid_ids)].copy()
    reviews['date'] = pd.to_datetime(reviews['date'], errors='coerce')

    # keep only the most recent reviews per listing
    reviews = (
        reviews.sort_values('date', ascending=False)
               .groupby('listing_id', group_keys=False)
               .head(max_per_listing)
    )
    reviews = reviews.reset_index(drop=True)

    print(f"    Running VADER sentiment on {len(reviews):,} review texts...")
    analyzer = SentimentIntensityAnalyzer()
    reviews['sentiment'] = reviews['comments'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )

    # recency weight: exp(-ln2 * age / half_life) = 1.0 for today, 0.5 in half_life
    days_old = (snapshot_date - reviews['date']).dt.days.clip(lower=0)
    days_old = days_old.fillna(snapshot_date.year * 365)   # unknown date = very old
    reviews['recency_weight'] = np.exp(-np.log(2) * days_old / half_life_days)

    def _recency_mean(g):
        w = g['recency_weight'].values
        s = g['sentiment'].values
        return float(np.average(s, weights=w)) if w.sum() > 0 else float(s.mean())

    base_agg = reviews.groupby('listing_id').agg(
        review_sentiment_mean=('sentiment', 'mean'),
        review_sentiment_std=('sentiment', 'std'),
        review_sentiment_min=('sentiment', 'min'),
    )
    recency = reviews.groupby('listing_id').apply(_recency_mean).rename(
        'review_sentiment_recency_wt'
    )
    result = base_agg.join(recency).reset_index()
    result['review_sentiment_std'] = result['review_sentiment_std'].fillna(0)

    #Cache saved
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
        print(f"    Cached review sentiments to {cache_path}")

    return result


def merge_review_features(
    listings: pd.DataFrame,
    review_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join review sentiment onto listings.
    Listings with no reviews get 0 for all sentiment columns.
    """
    listings = listings.merge(
        review_features, left_on='id', right_on='listing_id', how='left'
    )
    for col in [
        'review_sentiment_mean', 'review_sentiment_std',
        'review_sentiment_min', 'review_sentiment_recency_wt',
    ]:
        listings[col] = listings[col].fillna(0)
    listings.drop(columns=['listing_id'], inplace=True)
    return listings


#8. Categorical encoding

def onehot_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list = CAT_COLS,
    top_property_types: int = 10,
) -> tuple:
    """
    One-hot encode categorical columns
    """
    train, test = train.copy(), test.copy()

    # cap property_type using training frequency
    top_types = train['property_type'].value_counts().nlargest(top_property_types).index
    train['property_type'] = train['property_type'].where(
        train['property_type'].isin(top_types), 'Other'
    )
    test['property_type'] = test['property_type'].where(
        test['property_type'].isin(top_types), 'Other'
    )

    train = pd.get_dummies(train, columns=cat_cols, drop_first=True)
    test  = pd.get_dummies(test,  columns=cat_cols, drop_first=True)

    # align suchthat splits have identical columns
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    return train, test


def target_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    cat_cols: list = CAT_COLS,
    smoothing: int = TARGET_SMOOTHING,
) -> tuple:
    """
    Smoothed (m-estimate) target encoding. Fitted on train.

    Each category is replaced by a weighted blend:
     = (count * category_mean + m * global_mean)/(count + m)

    where m = smoothing. High-frequency categories converge to their true mean;
    rare categories are pulled toward the global mean preventing the extreme
    values that simple target encoding assigns to rare categories.

    The original categorical columns are dropped from both splits.
    """
    train, test = train.copy(), test.copy()
    global_mean = y_train.mean()

    for col in cat_cols:
        stats = y_train.groupby(train[col]).agg(['mean', 'count'])
        smooth_enc = (
            (stats['count'] * stats['mean'] + smoothing * global_mean)
            / (stats['count'] + smoothing)
        )
        train[f'{col}_te'] = train[col].map(smooth_enc).fillna(global_mean)
        test[f'{col}_te']  = test[col].map(smooth_enc).fillna(global_mean)

    train.drop(columns=cat_cols, inplace=True)
    test.drop(columns=cat_cols, inplace=True)

    return train, test


#9. Cleanup

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any character that is not alphanumeric or underscore with '_'.
    Required for LightGBM
    """
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    return df


def _drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df.drop(columns=present)


#10. main pipeline

def build_features_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    reviews: pd.DataFrame,
) -> dict:
    """
    Full feature engineering pipeline.
      Returns a dict with:
        train_ohe, test_ohe  : one-hot encoded DataFrames
        train_te,  test_te  : target-encoded DataFrames
        pca           : fitted PCA object
        train_emb_raw    : raw train embeddings
        test_emb_raw    : raw test  embeddings
        selected_amenities : list of selected amenity name strings
    """
    from sentence_transformers import SentenceTransformer

    #  Step 1: Amenity selection (train)
    print("\n[1/8] Selecting amenities via point-biserial correlation (train only)...")
    selected_amenities = select_amenities_biserial(train)

    #  Step 2: Remaining features
    print("\n[2/8] Building amenity, host tenure, description, and spatial features...")
    for name, df in (('train', train), ('test', test)):
        df = build_amenity_features(df, selected_amenities)
        df = extract_host_tenure(df)
        df = add_description_features(df)
        df = add_spatial_features(df)
        df = _drop_columns(df, _DROP_PRE_TEXT)                                        
        df = _drop_columns(df, _DROP_MISC)
        if name == 'train':
            train = df
        else:
            test = df

    #Step 3: Load models
    print("\n[3/8] Loading multilingual sentiment pipeline and embedding model...")
    sent_pipe   = load_sentiment_pipeline()
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # Step 4: Description sentiment
    print("\n[4/8] Computing multilingual description sentiment...")
    train, test = add_description_sentiment(train, test, sent_pipe)

    # Step 5: Description embeddings + PCA
    print("\n[5/8] Computing description embeddings (PCA fitted on train only)...")
    train, test, pca, train_emb_raw, test_emb_raw = build_description_embeddings_split(
        train, test, embed_model
    )

    #  Step 6: Drop description column (processed) 
    train = _drop_columns(train, _DROP_POST_TEXT)
    test  = _drop_columns(test,  _DROP_POST_TEXT)

    #  Step 7: Review sentiment
    print("\n[6/8] Computing review sentiment (VADER, recency-weighted)...")
    all_valid_ids = set(train['id'].tolist()) | set(test['id'].tolist())
    review_feats  = build_review_sentiment(
        reviews, all_valid_ids,
        cache_path=_REVIEW_CACHE_PATH,
    )
    train = merge_review_features(train, review_feats)
    test  = merge_review_features(test,  review_feats)

    #  Step 8: Categorical encoding → two variants 
    print("\n[7/8] One-hot encoding categorical columns...")
    y_train = train['log_price']
    train_ohe, test_ohe = onehot_encode(train, test)
    train_ohe = clean_col_names(train_ohe)
    test_ohe  = clean_col_names(test_ohe)

    print("\n[8/8] Target encoding categorical columns...")
    train_te, test_te = target_encode(train, test, y_train)
    train_te = clean_col_names(train_te)
    test_te  = clean_col_names(test_te)

    print(f"\nOHE shapes  — train: {train_ohe.shape}, test: {test_ohe.shape}")
    print(f"TE  shapes  — train: {train_te.shape},  test: {test_te.shape}")

    return {
        'train_ohe':          train_ohe,
        'test_ohe':           test_ohe,
        'train_te':           train_te,
        'test_te':            test_te,
        'pca':                pca,
        'train_emb_raw':      train_emb_raw,
        'test_emb_raw':       test_emb_raw,
        'selected_amenities': selected_amenities,
    }
