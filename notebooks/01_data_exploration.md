# Fake News Detection with GAT - Data Exploration

This notebook explores the FakeNewsNet dataset and demonstrates the data preprocessing pipeline.

## 1. Setup

```python
import sys
sys.path.append('..')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import FakeNewsNetLoader, TextPreprocessor
from src.visualization import MetricsVisualizer

%matplotlib inline
sns.set_style('whitegrid')
```

## 2. Load Dataset

```python
# Initialize loader
loader = FakeNewsNetLoader('../data/raw/fakenewsnet')

# Get dataset statistics
stats = loader.get_statistics()
print("Dataset Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

## 3. Load and Explore News Data

```python
# Load news content
news_df, social_df = loader.load_all_data()

print(f"News articles: {len(news_df)}")
print(f"Social posts: {len(social_df)}")

# Display sample news
news_df.head()
```

## 4. Label Distribution

```python
# Plot label distribution
plt.figure(figsize=(8, 6))
news_df['label'].value_counts().plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
plt.title('Distribution of Fake vs Real News')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```

## 5. Text Preprocessing

```python
# Initialize preprocessor
preprocessor = TextPreprocessor(remove_stopwords=True, lowercase=True)

# Example preprocessing
sample_text = news_df.iloc[0]['title']
print(f"Original: {sample_text}")
print(f"Cleaned: {preprocessor.preprocess(sample_text)}")
```

## 6. Social Context Analysis

```python
# Analyze social engagement
if len(social_df) > 0:
    print("Social engagement statistics:")
    print(social_df[['retweet_count', 'favorite_count']].describe())
```

## 7. Next Steps

- Run graph construction: `src/features/build_graph.py`
- Generate embeddings: `src/features/embeddings.py`
- Train GAT model: `scripts/train_model.py`
