"""
Build Temporal Spreading Curves
================================

For each article, convert its tweet timestamps into a fixed-length
time series representing tweet arrival volume over time.

Each article gets a 48-bin normalized curve showing the SHAPE of how
it spread — bursty vs sustained, fast decay vs slow build, etc.

Also computes robust handcrafted temporal features as a fallback.

Input:  data/processed/news_with_tweet_timestamps.parquet
Output: data/processed/temporal_curves.npy      (N x 48 curves)
        data/processed/temporal_features.npy    (N x F handcrafted features)
        data/processed/temporal_article_ids.npy (N article IDs for alignment)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUILDING TEMPORAL SPREADING CURVES")
print("=" * 80)

# ============================================================================
# 1. LOAD TIMESTAMP DATA
# ============================================================================
print("\n1. Loading timestamp data...")
news_df = pd.read_parquet('data/processed/news_with_tweet_timestamps.parquet')
print(f"   Loaded {len(news_df)} articles")

# Normalize tweet_timestamps column
def to_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return []

news_df['tweet_timestamps'] = news_df['tweet_timestamps'].apply(to_list)

# Count articles with enough tweets for curve building
min_tweets_for_curve = 5
has_enough = news_df['tweet_timestamps'].apply(lambda x: len(x) >= min_tweets_for_curve)
print(f"   Articles with >= {min_tweets_for_curve} tweets: {has_enough.sum():,} / {len(news_df):,}")

# ============================================================================
# 2. BUILD TEMPORAL CURVES (48-bin time series per article)
# ============================================================================
print("\n2. Building 48-bin temporal curves...")

N_BINS = 48
curves = np.zeros((len(news_df), N_BINS), dtype=np.float32)

for i, row in news_df.iterrows():
    timestamps = row['tweet_timestamps']
    if len(timestamps) < 2:
        # Can't build a curve with < 2 points — leave as zeros
        continue

    timestamps = np.array(timestamps, dtype=np.float64)
    t_min = timestamps.min()
    t_max = timestamps.max()
    duration = t_max - t_min

    if duration == 0:
        # All tweets at same millisecond — single spike in first bin
        curves[i, 0] = 1.0
        continue

    # Normalize timestamps to [0, 1] relative to article's own timeline
    t_normalized = (timestamps - t_min) / duration

    # Bin into N_BINS equally-spaced time windows
    bin_indices = np.clip((t_normalized * N_BINS).astype(int), 0, N_BINS - 1)
    for b in bin_indices:
        curves[i, b] += 1.0

    # Normalize curve to represent SHAPE, not volume
    # (volume info is captured separately in handcrafted features)
    curve_max = curves[i].max()
    if curve_max > 0:
        curves[i] /= curve_max

print(f"   Curves shape: {curves.shape}")
print(f"   Non-zero curves: {(curves.sum(axis=1) > 0).sum():,}")
print(f"   Zero curves (< 2 tweets): {(curves.sum(axis=1) == 0).sum():,}")

# ============================================================================
# 3. COMPUTE HANDCRAFTED TEMPORAL FEATURES
# ============================================================================
print("\n3. Computing handcrafted temporal features...")

feature_names = [
    'spread_duration_hours',
    'propagation_speed',       # tweets per hour
    'burstiness',              # CV of inter-tweet intervals
    'early_ratio',             # fraction of tweets in first 25% of time window
    'late_ratio',              # fraction of tweets in last 25% of time window
    'early_late_ratio',        # early / late ratio
    'peak_bin_position',       # position of peak activity (0-1)
    'temporal_entropy',        # Shannon entropy of binned distribution
    'interval_mean_hours',     # mean inter-tweet interval
    'interval_std_hours',      # std of inter-tweet intervals
    'interval_median_hours',   # median inter-tweet interval
    'interval_skewness',       # skewness of intervals (positive = right-skewed)
    'num_tweets_log',          # log(num_tweets) — how much attention
    'acceleration_first_half', # tweet rate in first half vs second half
]

N_FEATURES = len(feature_names)
features = np.zeros((len(news_df), N_FEATURES), dtype=np.float32)

for i, row in news_df.iterrows():
    timestamps = row['tweet_timestamps']
    n_tweets = len(timestamps)

    if n_tweets < 2:
        # Only set num_tweets_log for articles with at least 1 tweet
        if n_tweets == 1:
            features[i, feature_names.index('num_tweets_log')] = np.log1p(1)
        continue

    timestamps = np.array(sorted(timestamps), dtype=np.float64)
    t_min = timestamps.min()
    t_max = timestamps.max()
    duration_ms = t_max - t_min
    duration_hours = duration_ms / (1000 * 3600)

    # Basic features
    features[i, 0] = duration_hours                                    # spread_duration_hours
    features[i, 1] = n_tweets / max(duration_hours, 0.001)            # propagation_speed
    features[i, 12] = np.log1p(n_tweets)                              # num_tweets_log

    # Inter-tweet intervals
    intervals_ms = np.diff(timestamps)
    intervals_hours = intervals_ms / (1000 * 3600)

    if len(intervals_hours) > 0 and intervals_hours.mean() > 0:
        features[i, 2] = intervals_hours.std() / intervals_hours.mean()  # burstiness (CV)
    features[i, 8] = intervals_hours.mean()                               # interval_mean
    features[i, 9] = intervals_hours.std() if len(intervals_hours) > 1 else 0  # interval_std
    features[i, 10] = np.median(intervals_hours)                          # interval_median

    if len(intervals_hours) >= 3:
        features[i, 11] = sp_stats.skew(intervals_hours)                 # interval_skewness

    # Temporal distribution features
    if duration_ms > 0:
        t_normalized = (timestamps - t_min) / duration_ms

        # Early/late ratios
        early_count = np.sum(t_normalized <= 0.25)
        late_count = np.sum(t_normalized >= 0.75)
        features[i, 3] = early_count / n_tweets                     # early_ratio
        features[i, 4] = late_count / n_tweets                      # late_ratio
        features[i, 5] = (early_count + 1) / (late_count + 1)       # early_late_ratio (+1 smooth)

        # Peak position
        curve = curves[i]
        if curve.sum() > 0:
            peak_bin = np.argmax(curve)
            features[i, 6] = peak_bin / N_BINS                       # peak_bin_position (0-1)

        # Temporal entropy
        bin_counts = np.histogram(t_normalized, bins=N_BINS, range=(0, 1))[0]
        bin_probs = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else bin_counts
        bin_probs = bin_probs[bin_probs > 0]
        if len(bin_probs) > 0:
            features[i, 7] = sp_stats.entropy(bin_probs)             # temporal entropy

        # Acceleration: first half rate vs second half rate
        first_half = np.sum(t_normalized <= 0.5)
        second_half = n_tweets - first_half
        features[i, 13] = (first_half + 1) / (second_half + 1)      # acceleration_first_half

print(f"   Features shape: {features.shape}")
print(f"   Feature names: {feature_names}")

# ============================================================================
# 4. HANDLE NaN/Inf
# ============================================================================
print("\n4. Cleaning features...")
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"   Cleaned NaN/Inf values")

# ============================================================================
# 5. FEATURE STATS (fake vs real)
# ============================================================================
print("\n5. Feature statistics (Fake vs Real)...")

label_encoded = news_df['label_encoded'].values if 'label_encoded' in news_df.columns else None
if label_encoded is None and 'label' in news_df.columns:
    label_encoded = (news_df['label'] == 'real').astype(int).values

if label_encoded is not None:
    fake_mask = label_encoded == 0
    real_mask = label_encoded == 1

    print(f"\n   {'Feature':<30} {'Fake Mean':>12} {'Real Mean':>12} {'Diff':>10} {'p-value':>10}")
    print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    discriminative_features = []
    for j, fname in enumerate(feature_names):
        fake_vals = features[fake_mask, j]
        real_vals = features[real_mask, j]

        fake_mean = fake_vals.mean()
        real_mean = real_vals.mean()
        diff = fake_mean - real_mean

        # Mann-Whitney U test (non-parametric, handles non-normal distributions)
        try:
            _, pval = sp_stats.mannwhitneyu(
                fake_vals[fake_vals != 0], real_vals[real_vals != 0],
                alternative='two-sided'
            )
        except ValueError:
            pval = 1.0

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"   {fname:<30} {fake_mean:>12.3f} {real_mean:>12.3f} {diff:>+10.3f} {pval:>10.2e} {sig}")

        if pval < 0.05:
            discriminative_features.append(fname)

    print(f"\n   Statistically significant features (p < 0.05): {len(discriminative_features)} / {N_FEATURES}")
    for fn in discriminative_features:
        print(f"     - {fn}")

# ============================================================================
# 6. NORMALIZE FEATURES (StandardScaler)
# ============================================================================
print("\n6. Normalizing features (zero mean, unit variance)...")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features).astype(np.float32)

# Clip extreme values to [-5, 5] to avoid outlier influence
features_normalized = np.clip(features_normalized, -5, 5)

print(f"   Normalized features: mean={features_normalized.mean():.4f}, std={features_normalized.std():.4f}")

# ============================================================================
# 7. SAVE OUTPUTS
# ============================================================================
print("\n7. Saving outputs...")

output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

# Save curves (N x 48)
np.save(output_dir / 'temporal_curves.npy', curves)
print(f"   Saved temporal curves: {curves.shape} -> {output_dir / 'temporal_curves.npy'}")

# Save handcrafted features (normalized) (N x 14)
np.save(output_dir / 'temporal_features.npy', features_normalized)
print(f"   Saved temporal features: {features_normalized.shape} -> {output_dir / 'temporal_features.npy'}")

# Save raw features (unnormalized) for analysis
np.save(output_dir / 'temporal_features_raw.npy', features)
print(f"   Saved raw features: {features.shape} -> {output_dir / 'temporal_features_raw.npy'}")

# Save article IDs for alignment
article_ids = news_df['id'].values
np.save(output_dir / 'temporal_article_ids.npy', article_ids)
print(f"   Saved article IDs: {article_ids.shape} -> {output_dir / 'temporal_article_ids.npy'}")

# Save feature names
with open(output_dir / 'temporal_feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"   Saved feature names -> {output_dir / 'temporal_feature_names.json'}")

# Summary
summary = {
    'n_articles': len(news_df),
    'n_bins': N_BINS,
    'n_handcrafted_features': N_FEATURES,
    'feature_names': feature_names,
    'articles_with_curves': int((curves.sum(axis=1) > 0).sum()),
    'articles_without_curves': int((curves.sum(axis=1) == 0).sum()),
    'min_tweets_for_curve': min_tweets_for_curve,
}
with open(output_dir / 'temporal_curves_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("DONE — Temporal curves and features built")
print("=" * 80)
print(f"\nOutputs:")
print(f"  temporal_curves.npy         ({curves.shape[0]} x {curves.shape[1]}): Normalized spreading curves")
print(f"  temporal_features.npy       ({features_normalized.shape[0]} x {features_normalized.shape[1]}): Handcrafted temporal features (normalized)")
print(f"  temporal_features_raw.npy   ({features.shape[0]} x {features.shape[1]}): Raw temporal features")
print(f"  temporal_article_ids.npy    ({article_ids.shape[0]}): Article ID alignment")
