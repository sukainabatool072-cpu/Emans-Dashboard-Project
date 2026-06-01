import pandas as pd
import numpy as np
import gzip
import os

DATA_PATH = os.path.join("data", "finefoods.txt.gz")

@st.cache_data
def load_data():
    """Load and clean the Fine Foods dataset"""
    records = []
    with gzip.open(DATA_PATH, 'rt', encoding='utf-8', errors='ignore') as f:
        current = {}
        for line in f:
            line = line.strip()
            if line == '':
                if current:
                    records.append(current)
                    current = {}
            elif ': ' in line:
                key, val = line.split(': ', 1)
                current[key] = val
        if current:
            records.append(current)

    df = pd.DataFrame(records)

    # Rename columns for cleaner access
    df.columns = df.columns.str.replace('product/', '').str.replace('review/', '')

    # Clean & convert types
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['date'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    # Helpfulness ratio
    def parse_helpfulness(h):
        try:
            num, den = h.split('/')
            num, den = int(num), int(den)
            return num / den if den > 0 else 0
        except:
            return 0

    df['helpfulness_ratio'] = df['helpfulness'].apply(parse_helpfulness)
    df['helpful_votes'] = df['helpfulness'].apply(lambda h: int(h.split('/')[0]) if '/' in str(h) else 0)
    df['total_votes'] = df['helpfulness'].apply(lambda h: int(h.split('/')[1]) if '/' in str(h) else 0)

    # Text features
    df['review_length'] = df['text'].fillna('').apply(len)
    df['word_count'] = df['text'].fillna('').apply(lambda x: len(x.split()))

    # Score label
    df['sentiment'] = df['score'].apply(lambda s: 'Positive' if s >= 4 else ('Neutral' if s == 3 else 'Negative'))

    # Drop rows with missing critical fields
    df = df.dropna(subset=['score', 'date'])

    return df


def apply_filters(df, score_range, year_range, sentiment_filter, search_text):
    """Apply all sidebar filters to the dataframe"""
    filtered = df.copy()

    # Score filter
    filtered = filtered[(filtered['score'] >= score_range[0]) & (filtered['score'] <= score_range[1])]

    # Year filter
    filtered = filtered[(filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])]

    # Sentiment filter
    if sentiment_filter and sentiment_filter != 'All':
        filtered = filtered[filtered['sentiment'] == sentiment_filter]

    # Search filter
    if search_text:
        mask = (
            filtered['summary'].fillna('').str.contains(search_text, case=False) |
            filtered['text'].fillna('').str.contains(search_text, case=False) |
            filtered['profileName'].fillna('').str.contains(search_text, case=False)
        )
        filtered = filtered[mask]

    return filtered
