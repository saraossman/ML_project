import numpy as np
import pandas as pd
import re

"""
Data cleaning functions for Airbnb listings
Operates on the raw listings data/raw/listings.csv.gz
"""

COLS_TO_DROP = [
    'listing_url', 'scrape_id', 'last_scraped', 'source',
    'host_url', 'host_thumbnail_url', 'host_picture_url',
    'neighbourhood',                  # 54.6% missing, use neighbourhood_cleansed
    'neighborhood_overview',          # 54.6% missing
    'host_neighbourhood',             # 59.3% missing
    'calendar_updated',               # 100% missing
    'license',                        # 34.9% missing, not useful for price
    'minimum_minimum_nights', 'maximum_minimum_nights',
    'minimum_maximum_nights', 'maximum_maximum_nights',
    'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',  # redundant
    'neighbourhood_group_cleansed',
    'has_availability',
    'calendar_last_scraped',
]

BOOL_COLS = [
    'host_is_superhost', 'host_has_profile_pic',
    'host_identity_verified', 'instant_bookable',
]

RATE_COLS = ['host_response_rate', 'host_acceptance_rate']

REVIEW_SCORE_COLS = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication',
    'review_scores_location', 'review_scores_value',
]

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are empty, redundant, or not useful for price prediction."""
    return df.drop(columns=COLS_TO_DROP)


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    removes $, convert to float, add log_price.
    Removes rows with missing or zero prices.
    """
    df = df[df['price'].notna()].copy()
    df['price'] = df['price'].str.replace('[$,]', '', regex=True).astype(float)
    df = df[df['price'] > 0]
    df['log_price'] = np.log1p(df['price'])
    return df


def filter_min_nights(df: pd.DataFrame, max_nights: int = 30) -> pd.DataFrame:
    """Remove long-term rental listings (min_nights > max_nights)."""
    return df[df['minimum_nights'] <= max_nights].copy()


def _parse_bathrooms(text) -> float:
    """Parse bathroom count from free-text field (e.g. '1.5 baths', 'Half-bath')."""
    if pd.isna(text):
        return np.nan
    text = text.lower()
    if 'half' in text or 'half-bath' in text:
        return 0.5
    nums = re.findall(r'\d+\.?\d*', text)
    return float(nums[0]) if nums else np.nan


def clean_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse bathrooms_text (0.1% missing) into a numeric column.
    Impute remaining nulls with median. Drop original bathroom columns.
    """
    df['bathrooms_parsed'] = df['bathrooms_text'].apply(_parse_bathrooms)
    df['bathrooms_parsed'] = df['bathrooms_parsed'].fillna(df['bathrooms_parsed'].median())
    df.drop(columns=['bathrooms_text', 'bathrooms'], inplace=True)
    return df


def impute_bedrooms_beds(df: pd.DataFrame) -> pd.DataFrame:
    """Impute bedrooms (14% missing) and beds (35% missing) using room_type group medians."""
    df['bedrooms'] = df.groupby('room_type')['bedrooms'].transform(
        lambda x: x.fillna(x.median())
    )
    df['beds'] = df.groupby('room_type')['beds'].transform(
        lambda x: x.fillna(x.median())
    )
    return df


def fix_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert t/f strings to 0/1 integers.
    removes % from host_response_rate and host_acceptance_rate.
    """
    for col in BOOL_COLS:
        df[col] = df[col].map({'t': 1, 'f': 0})
    df['host_is_superhost'] = df['host_is_superhost'].fillna(0)

    for col in RATE_COLS:
        df[col] = df[col].str.replace('%', '').astype(float)
    return df


def impute_review_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add has_reviews flag, then impute all review score columns with their median."""
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
    for col in REVIEW_SCORE_COLS:
        df[col] = df[col].fillna(df[col].median())
    return df


def engineer_presence_flags(df: pd.DataFrame, raw_listings: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary values for text fields.
    Drops the original high-cardinality text columns afterwards.
    raw_listings is the original df (before dropping neighborhood_overview).
    """
    df['has_neighborhood_overview'] = raw_listings['neighborhood_overview'].notna().astype(int)
    df['host_location_given'] = df['host_location'].notna().astype(int)
    df.drop(columns=['host_location'], inplace=True)

    df['has_host_about'] = df['host_about'].notna().astype(int)
    df.drop(columns=['host_about'], inplace=True)

    df['has_description'] = df['description'].notna().astype(int)
    df['description'] = df['description'].fillna('')
    return df


def fill_host_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing host response fields.
    Unknown response_time gets 'unknown' category.
    Unknown rates gets -1
    """
    df['host_response_time'] = df['host_response_time'].fillna('unknown')
    df['host_response_rate'] = df['host_response_rate'].fillna(-1)
    df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(-1)
    return df


def compute_host_effort_score(df: pd.DataFrame) -> pd.DataFrame:
    """score of how much effort a host put into their profile."""
    df['host_effort_score'] = (
        df['has_description'].astype(int) +
        df['has_neighborhood_overview'].astype(int) +
        df['has_host_about'].astype(int) +
        (df['host_has_profile_pic'] == 1).astype(int) +
        (df['host_identity_verified'] == 1).astype(int)
    )
    return df


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    calculate days_since_last_review from last_review.
    9999 means never reviewed. Drop raw date columns.
    """
    df['days_since_last_review'] = (
        pd.to_datetime('today') - pd.to_datetime(df['last_review'])
    ).dt.days
    df['days_since_last_review'] = df['days_since_last_review'].fillna(9999)
    df.drop(columns=['first_review', 'last_review'], inplace=True)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    return df


def drop_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 2 rows with nulls in host identity"""
    subset = [
        'host_name', 'host_since', 'host_listings_count',
        'host_total_listings_count', 'host_verifications',
        'host_has_profile_pic', 'host_identity_verified',
    ]
    return df.dropna(subset=subset)

def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for raw listings DataFrame.
    Returns a cleaned DataFrame ready for feature engineering.
    """
    raw = df.copy()

    df = drop_irrelevant_columns(df)
    df = clean_price(df)
    df = filter_min_nights(df)
    df = clean_bathrooms(df)
    df = impute_bedrooms_beds(df)
    df = fix_boolean_columns(df)
    df = impute_review_scores(df)
    df = engineer_presence_flags(df, raw)
    df = fill_host_response(df)
    df = compute_host_effort_score(df)
    df = extract_date_features(df)
    df = drop_remaining_nulls(df)

    return df
