"""
Decode Twitter Snowflake IDs to Timestamps
==========================================

Twitter Snowflake IDs encode millisecond-precision timestamps.
Formula: timestamp_ms = (tweet_id >> 22) + 1288834974657

This script decodes all tweet IDs in the dataset to recover real
timestamps — enabling temporal spreading dynamics analysis.

Expected: timestamps in ~2015-2020 range (FakeNewsNet collection period)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import json

# Twitter Snowflake epoch (Nov 4, 2010 01:42:54.657 UTC)
TWITTER_EPOCH_MS = 1288834974657

print("=" * 80)
print("DECODING TWITTER SNOWFLAKE TIMESTAMPS")
print("=" * 80)


def snowflake_to_timestamp_ms(tweet_id_str):
    """Convert a Twitter Snowflake ID string to Unix timestamp in milliseconds."""
    try:
        tweet_id = int(tweet_id_str.strip())
        if tweet_id <= 0:
            return None
        ts_ms = (tweet_id >> 22) + TWITTER_EPOCH_MS
        # Sanity: should be between 2010 and 2025
        if ts_ms < 1288834974657 or ts_ms > 1735689600000:  # 2010 to ~2025
            return None
        return ts_ms
    except (ValueError, TypeError, OverflowError):
        return None


# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================
print("\n1. Loading processed data...")
news_df = pd.read_parquet('data/processed/news_processed.parquet')
print(f"   Loaded {len(news_df)} articles")
print(f"   Columns: {list(news_df.columns)}")

# ============================================================================
# 2. DECODE TWEET TIMESTAMPS
# ============================================================================
print("\n2. Decoding Snowflake timestamps for all tweets...")

# tweet_ids may be stored as numpy arrays, lists, or strings after parquet round-trip
# Normalize to plain Python lists of strings
def normalize_tweet_ids(val):
    """Convert tweet_ids to a plain list of strings regardless of stored format."""
    import numpy as np
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        if val.startswith('['):
            import ast
            return ast.literal_eval(val)
        return val.split('\t')
    return []

news_df['tweet_ids'] = news_df['tweet_ids'].apply(normalize_tweet_ids)

total_tweets = 0
valid_tweets = 0
invalid_tweets = 0

all_timestamps = []  # list of lists — one per article

for idx, row in news_df.iterrows():
    tweet_ids = row['tweet_ids']
    if not isinstance(tweet_ids, list):
        tweet_ids = []

    timestamps = []
    for tid in tweet_ids:
        total_tweets += 1
        ts = snowflake_to_timestamp_ms(tid)
        if ts is not None:
            timestamps.append(ts)
            valid_tweets += 1
        else:
            invalid_tweets += 1

    # Sort timestamps chronologically
    timestamps.sort()
    all_timestamps.append(timestamps)

news_df['tweet_timestamps'] = all_timestamps
news_df['num_valid_tweets'] = news_df['tweet_timestamps'].apply(len)

print(f"   Total tweet IDs processed: {total_tweets:,}")
if total_tweets > 0:
    print(f"   Valid (decoded):           {valid_tweets:,} ({valid_tweets/total_tweets*100:.1f}%)")
    print(f"   Invalid (skipped):         {invalid_tweets:,} ({invalid_tweets/total_tweets*100:.1f}%)")
else:
    print("   WARNING: No tweet IDs found! Check tweet_ids column format.")

# ============================================================================
# 3. COMPUTE BASIC TEMPORAL STATS
# ============================================================================
print("\n3. Computing temporal statistics per article...")

first_tweet_ts = []
last_tweet_ts = []
spread_duration_hours = []

for timestamps in news_df['tweet_timestamps']:
    if len(timestamps) >= 1:
        first_tweet_ts.append(timestamps[0])
        last_tweet_ts.append(timestamps[-1])
        if len(timestamps) >= 2:
            duration_ms = timestamps[-1] - timestamps[0]
            spread_duration_hours.append(duration_ms / (1000 * 3600))
        else:
            spread_duration_hours.append(0.0)
    else:
        first_tweet_ts.append(None)
        last_tweet_ts.append(None)
        spread_duration_hours.append(None)

news_df['first_tweet_ts'] = first_tweet_ts
news_df['last_tweet_ts'] = last_tweet_ts
news_df['spread_duration_hours'] = spread_duration_hours

# Convert to human-readable dates for verification
news_df['first_tweet_date'] = news_df['first_tweet_ts'].apply(
    lambda x: datetime.fromtimestamp(x / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    if pd.notna(x) and x is not None else None
)

# ============================================================================
# 4. SANITY CHECKS
# ============================================================================
print("\n4. Sanity checks...")

valid_dates = news_df['first_tweet_date'].dropna()
articles_with_timestamps = news_df['num_valid_tweets'].gt(0).sum()
articles_without = len(news_df) - articles_with_timestamps

print(f"   Articles with timestamps: {articles_with_timestamps:,} / {len(news_df):,}")
print(f"   Articles without:         {articles_without:,}")

if len(valid_dates) > 0:
    print(f"\n   Date range of articles:")
    print(f"   Earliest: {valid_dates.min()}")
    print(f"   Latest:   {valid_dates.max()}")

    # Year distribution
    years = news_df['first_tweet_date'].dropna().apply(lambda x: x[:4])
    year_counts = years.value_counts().sort_index()
    print(f"\n   Articles by year:")
    for year, count in year_counts.items():
        print(f"     {year}: {count:,}")

# Spread duration stats
valid_spread = news_df['spread_duration_hours'].dropna()
if len(valid_spread) > 0:
    print(f"\n   Spread duration (hours):")
    print(f"     Mean:   {valid_spread.mean():.1f}")
    print(f"     Median: {valid_spread.median():.1f}")
    print(f"     Max:    {valid_spread.max():.1f}")
    print(f"     Min:    {valid_spread.min():.3f}")

# Fake vs Real comparison
for label in ['fake', 'real']:
    if 'label' in news_df.columns:
        subset = news_df[news_df['label'] == label]
        valid_sub = subset['spread_duration_hours'].dropna()
        valid_tweets_sub = subset['num_valid_tweets']
        if len(valid_sub) > 0:
            print(f"\n   {label.upper()} news:")
            print(f"     Avg tweets:          {valid_tweets_sub.mean():.1f}")
            print(f"     Avg spread (hours):  {valid_sub.mean():.1f}")
            print(f"     Median spread (hrs): {valid_sub.median():.1f}")

# ============================================================================
# 5. SAVE
# ============================================================================
print("\n5. Saving data with timestamps...")

output_path = Path('data/processed/news_with_tweet_timestamps.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the full dataframe
news_df.to_parquet(output_path, index=False)
print(f"   Saved to: {output_path}")

# Also save a summary JSON
summary = {
    'total_articles': len(news_df),
    'articles_with_timestamps': int(articles_with_timestamps),
    'total_tweets_processed': total_tweets,
    'valid_tweets': valid_tweets,
    'invalid_tweets': invalid_tweets,
    'date_range': {
        'earliest': str(valid_dates.min()) if len(valid_dates) > 0 else None,
        'latest': str(valid_dates.max()) if len(valid_dates) > 0 else None,
    },
    'spread_duration_hours': {
        'mean': float(valid_spread.mean()) if len(valid_spread) > 0 else None,
        'median': float(valid_spread.median()) if len(valid_spread) > 0 else None,
    }
}

summary_path = Path('data/processed/timestamp_decode_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   Summary saved to: {summary_path}")

print("\n" + "=" * 80)
print("DONE — Snowflake timestamps decoded successfully")
print("=" * 80)
