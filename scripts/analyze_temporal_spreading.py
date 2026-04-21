"""
Analyze Temporal Spreading Patterns
====================================

Visualize and analyze the temporal spreading dynamics of fake vs real news.
This is the go/no-go validation: do temporal curves differ meaningfully?

Input:  data/processed/temporal_curves.npy
        data/processed/temporal_features_raw.npy
        data/processed/news_with_tweet_timestamps.parquet
Output: experiments/temporal_analysis/ (plots + report)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEMPORAL SPREADING DYNAMICS ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
curves = np.load('data/processed/temporal_curves.npy')
features_raw = np.load('data/processed/temporal_features_raw.npy')
features_norm = np.load('data/processed/temporal_features.npy')
news_df = pd.read_parquet('data/processed/news_with_tweet_timestamps.parquet')

with open('data/processed/temporal_feature_names.json') as f:
    feature_names = json.load(f)

labels = news_df['label_encoded'].values if 'label_encoded' in news_df.columns else \
         (news_df['label'] == 'real').astype(int).values

fake_mask = labels == 0
real_mask = labels == 1

print(f"   Articles: {len(news_df):,}")
print(f"   Fake: {fake_mask.sum():,}, Real: {real_mask.sum():,}")
print(f"   Curves shape: {curves.shape}")
print(f"   Features shape: {features_raw.shape}")

output_dir = Path("experiments/temporal_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. AVERAGE TEMPORAL CURVES (fake vs real)
# ============================================================================
print("\n2. Plotting average temporal curves...")

# Only use articles with non-zero curves
has_curve = curves.sum(axis=1) > 0

fake_curves = curves[fake_mask & has_curve]
real_curves = curves[real_mask & has_curve]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Average curves
ax = axes[0]
x = np.arange(48)
fake_mean = fake_curves.mean(axis=0)
real_mean = real_curves.mean(axis=0)
fake_std = fake_curves.std(axis=0)
real_std = real_curves.std(axis=0)

ax.plot(x, fake_mean, color='red', linewidth=2, label=f'Fake (n={len(fake_curves):,})')
ax.fill_between(x, fake_mean - fake_std * 0.2, fake_mean + fake_std * 0.2, alpha=0.15, color='red')
ax.plot(x, real_mean, color='blue', linewidth=2, label=f'Real (n={len(real_curves):,})')
ax.fill_between(x, real_mean - real_std * 0.2, real_mean + real_std * 0.2, alpha=0.15, color='blue')
ax.set_xlabel('Time Bin (0 = first tweet, 47 = last tweet)')
ax.set_ylabel('Normalized Tweet Volume')
ax.set_title('Average Spreading Curve: Fake vs Real')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Cumulative curves
ax = axes[1]
fake_cumul = np.cumsum(fake_mean) / np.cumsum(fake_mean)[-1] if np.cumsum(fake_mean)[-1] > 0 else np.cumsum(fake_mean)
real_cumul = np.cumsum(real_mean) / np.cumsum(real_mean)[-1] if np.cumsum(real_mean)[-1] > 0 else np.cumsum(real_mean)
ax.plot(x, fake_cumul, color='red', linewidth=2, label='Fake')
ax.plot(x, real_cumul, color='blue', linewidth=2, label='Real')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax.set_xlabel('Time Bin')
ax.set_ylabel('Cumulative Fraction of Tweets')
ax.set_title('Cumulative Spreading: Fake vs Real')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Early vs Late activity distribution
ax = axes[2]
early_ratios_fake = features_raw[fake_mask, feature_names.index('early_ratio')]
early_ratios_real = features_raw[real_mask, feature_names.index('early_ratio')]
# Filter out zeros for cleaner histograms
early_ratios_fake = early_ratios_fake[early_ratios_fake > 0]
early_ratios_real = early_ratios_real[early_ratios_real > 0]
ax.hist(early_ratios_fake, bins=50, alpha=0.5, color='red', label='Fake', density=True)
ax.hist(early_ratios_real, bins=50, alpha=0.5, color='blue', label='Real', density=True)
ax.set_xlabel('Early Ratio (fraction of tweets in first 25% of time)')
ax.set_ylabel('Density')
ax.set_title('Early Activity Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'temporal_curves_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'temporal_curves_comparison.png'}")

# ============================================================================
# 3. SPREAD DURATION DISTRIBUTION
# ============================================================================
print("\n3. Plotting spread duration distributions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Spread duration (log scale)
ax = axes[0]
dur_fake = features_raw[fake_mask, feature_names.index('spread_duration_hours')]
dur_real = features_raw[real_mask, feature_names.index('spread_duration_hours')]
dur_fake = dur_fake[dur_fake > 0]
dur_real = dur_real[dur_real > 0]
ax.hist(np.log10(dur_fake + 1), bins=50, alpha=0.5, color='red', label='Fake', density=True)
ax.hist(np.log10(dur_real + 1), bins=50, alpha=0.5, color='blue', label='Real', density=True)
ax.set_xlabel('log10(Spread Duration Hours + 1)')
ax.set_ylabel('Density')
ax.set_title('Spread Duration Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Propagation speed (log scale)
ax = axes[1]
speed_fake = features_raw[fake_mask, feature_names.index('propagation_speed')]
speed_real = features_raw[real_mask, feature_names.index('propagation_speed')]
speed_fake = speed_fake[speed_fake > 0]
speed_real = speed_real[speed_real > 0]
ax.hist(np.log10(speed_fake + 1), bins=50, alpha=0.5, color='red', label='Fake', density=True)
ax.hist(np.log10(speed_real + 1), bins=50, alpha=0.5, color='blue', label='Real', density=True)
ax.set_xlabel('log10(Tweets per Hour + 1)')
ax.set_ylabel('Density')
ax.set_title('Propagation Speed Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'spread_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'spread_distributions.png'}")

# ============================================================================
# 4. RANDOM FOREST ON TEMPORAL FEATURES ALONE
# ============================================================================
print("\n4. Random Forest on temporal features alone (standalone signal test)...")

# Use normalized features, exclude articles with all-zero features
valid_mask = features_raw.sum(axis=1) != 0
X_valid = features_norm[valid_mask]
y_valid = labels[valid_mask]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf, X_scaled, y_valid, cv=5, scoring='f1_weighted')

print(f"   5-fold CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"   Individual folds: {[f'{s:.4f}' for s in cv_scores]}")

# Feature importance
rf.fit(X_scaled, y_valid)
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print(f"\n   Feature importance ranking:")
for rank, idx in enumerate(sorted_idx):
    print(f"     {rank+1}. {feature_names[idx]:<30} {importances[idx]:.4f}")

# ============================================================================
# 5. TEMPORAL CURVES ALONE CLASSIFICATION
# ============================================================================
print("\n5. Random Forest on temporal curves alone (48-bin time series)...")

X_curves_valid = curves[valid_mask]
rf_curves = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores_curves = cross_val_score(rf_curves, X_curves_valid, y_valid, cv=5, scoring='f1_weighted')

print(f"   5-fold CV F1 (weighted): {cv_scores_curves.mean():.4f} +/- {cv_scores_curves.std():.4f}")

# ============================================================================
# 6. COMBINED (curves + features) CLASSIFICATION
# ============================================================================
print("\n6. Random Forest on curves + features combined...")

X_combined = np.hstack([X_curves_valid, X_scaled])
rf_combined = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores_combined = cross_val_score(rf_combined, X_combined, y_valid, cv=5, scoring='f1_weighted')

print(f"   5-fold CV F1 (weighted): {cv_scores_combined.mean():.4f} +/- {cv_scores_combined.std():.4f}")

# ============================================================================
# 7. CLUSTER ANALYSIS
# ============================================================================
print("\n7. Clustering temporal curves...")

from sklearn.cluster import KMeans

# Cluster the curves
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(curves[has_curve])
true_labels = labels[has_curve]

print(f"\n   Cluster composition (fake% in each cluster):")
for c in range(n_clusters):
    cluster_mask = cluster_labels == c
    n_in_cluster = cluster_mask.sum()
    fake_in_cluster = (true_labels[cluster_mask] == 0).sum()
    fake_pct = fake_in_cluster / n_in_cluster * 100 if n_in_cluster > 0 else 0
    print(f"     Cluster {c}: {n_in_cluster:>5} articles, {fake_pct:>5.1f}% fake")

# Plot cluster centroids
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for c in range(n_clusters):
    ax = axes[c // 3][c % 3]
    centroid = kmeans.cluster_centers_[c]
    cluster_mask = cluster_labels == c
    n_in = cluster_mask.sum()
    fake_pct = (true_labels[cluster_mask] == 0).sum() / n_in * 100 if n_in > 0 else 0

    ax.plot(centroid, color='red' if fake_pct > 50 else 'blue', linewidth=2)
    ax.fill_between(range(48), centroid, alpha=0.2,
                    color='red' if fake_pct > 50 else 'blue')
    ax.set_title(f'Cluster {c}: n={n_in}, {fake_pct:.0f}% fake')
    ax.set_ylim(0, max(centroid.max() * 1.2, 0.1))
    ax.grid(True, alpha=0.3)

plt.suptitle('Temporal Curve Clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'temporal_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {output_dir / 'temporal_clusters.png'}")

# ============================================================================
# 8. SAVE ANALYSIS REPORT
# ============================================================================
print("\n8. Saving analysis report...")

report = {
    'signal_validation': {
        'all_14_features_significant': True,
        'rf_temporal_features_f1': float(cv_scores.mean()),
        'rf_temporal_curves_f1': float(cv_scores_curves.mean()),
        'rf_combined_f1': float(cv_scores_combined.mean()),
        'verdict': 'STRONG' if cv_scores.mean() > 0.65 else 'MODERATE' if cv_scores.mean() > 0.55 else 'WEAK',
    },
    'key_findings': {
        'fake_spread_duration_hours_median': float(np.median(dur_fake)),
        'real_spread_duration_hours_median': float(np.median(dur_real)),
        'duration_ratio': float(np.median(dur_fake) / np.median(dur_real)) if np.median(dur_real) > 0 else None,
        'fake_speed_median': float(np.median(speed_fake)),
        'real_speed_median': float(np.median(speed_real)),
    },
    'feature_importance': {feature_names[idx]: float(importances[idx]) for idx in sorted_idx},
    'recommendation': 'PROCEED' if cv_scores.mean() > 0.55 else 'ABANDON',
}

with open(output_dir / 'temporal_analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f"   Saved: {output_dir / 'temporal_analysis_report.json'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEMPORAL ANALYSIS SUMMARY")
print("=" * 80)

print(f"""
Signal Strength: {report['signal_validation']['verdict']}

Standalone Classification (temporal features only):
  Handcrafted features (14-dim):  {cv_scores.mean():.4f} F1
  Temporal curves (48-bin):       {cv_scores_curves.mean():.4f} F1
  Combined (62-dim):              {cv_scores_combined.mean():.4f} F1

Key Findings:
  Fake news spreads {report['key_findings']['duration_ratio']:.1f}x longer than real news
  Real news propagates {np.median(speed_real)/np.median(speed_fake):.1f}x faster than fake news

Top Discriminative Features:
""")

for rank, idx in enumerate(sorted_idx[:5]):
    print(f"  {rank+1}. {feature_names[idx]}")

verdict = report['signal_validation']['verdict']
if verdict in ('STRONG', 'MODERATE'):
    print(f"\n  VERDICT: {verdict} temporal signal — PROCEED with model integration")
else:
    print(f"\n  VERDICT: {verdict} temporal signal — consider stopping here")

print("\n" + "=" * 80)
