# Fake News Detection Using Graph Neural Networks

## Semester 7 — Project Progress Report

**Student:** Harsh Kanani
**Date:** February 2026

---

## 1. What is This Project About?

This project builds a system that can **automatically detect fake news articles** by understanding how they spread and connect with each other on the internet.

Instead of just reading the words in an article (like a human fact-checker would), this system also looks at **relationships between articles** — for example, which articles reference similar sources, which ones share the same publisher, and which ones are connected through social media sharing patterns.

Think of it like this:

> Imagine a map where every news article is a dot, and lines connect articles that are related. Fake news articles tend to cluster together and form their own "bubble." Our system learns to spot these bubbles automatically.

---

## 2. Why is This Important?

- Fake news spreads **faster** than real news on social media.
- Manual fact-checking is **too slow** — by the time an article is checked, millions have already seen it.
- Existing detection tools only look at the **text** of an article. Our approach also looks at **how it spreads**, which gives much better results.

---

## 3. What Has Been Completed So Far?

### 3.1 Data Collection & Preparation

| What was done | Details |
|---|---|
| Collected a large dataset | **23,196 news articles** from the FakeNewsNet research dataset |
| Cleaned and organized the data | Removed junk characters, fixed formatting, standardized text |
| Labeled correctly | About 40% fake articles, 60% real articles |

### 3.2 Building the Article Network (Graph)

We connected articles to each other based on meaningful relationships:

| Relationship Type | What It Means | Number of Connections |
|---|---|---|
| Content Similarity | Two articles talk about the same topic | 95,325 |
| Same Publisher | Two articles come from the same news source | 1,000 |
| Echo Chamber | Fake articles that reference other fake articles | 10,000 |
| High Activity | Articles that went viral or got high engagement | 594 |
| **Total connections** | | **106,919** |

### 3.3 Training the Detection Model

We built and trained multiple versions of our detection system:

| Model Version | Accuracy | Notes |
|---|---|---|
| First attempt (small test) | 78.95% | Tested on only 500 articles to verify the idea works |
| Improved version | 87.22% | Better settings, trained longer |
| Ensemble (multiple models combined) | 91.49% | Combined 3 models for better results |
| **Best model (Graph Transformer)** | **92.21%** | **Our final and best result** |

### 3.4 Key Result: 92% Accuracy

Our best model correctly identifies **92 out of every 100 articles** as either fake or real.

Breaking it down:
- **Fake news detection rate:** 97.8% — catches almost all fake articles
- **Real news detection rate:** 75.2% — correctly identifies most real articles
- **Overall accuracy:** 92.21%

### 3.5 Important Discovery: The Echo Chamber Effect

We discovered that **fake news articles are heavily connected to other fake news articles**. About 70.5% of the connections from a fake article lead to other fake articles.

This means fake news creates its own "echo chamber" — a self-reinforcing bubble where misinformation references and amplifies other misinformation. This pattern is a strong signal for detection.

### 3.6 Explainability

The system doesn't just say "this is fake" — it can also explain **why** it thinks so:
- Which related articles influenced the decision
- What features (sentiment, writing style, source credibility) were most important
- An interactive visual dashboard was built to explore results

---

## 4. Tools and Technologies Used

| Category | What We Used | Why |
|---|---|---|
| Programming Language | Python | Industry standard for AI/ML |
| AI Framework | PyTorch | For building and training the neural network |
| Graph Processing | PyTorch Geometric | Specialized library for graph-based AI |
| Text Understanding | BERT / Sentence Transformers | To convert article text into numerical features |
| Data Analysis | Pandas, NumPy | For data processing and statistics |
| Visualization | Plotly, Matplotlib | For charts and interactive dashboards |

---

## 5. Summary of Achievements

| Milestone | Status |
|---|---|
| Dataset collected and cleaned | ✅ Done |
| Article network (graph) constructed | ✅ Done |
| Baseline model trained | ✅ Done |
| Improved models trained | ✅ Done |
| Best model achieving 92% accuracy | ✅ Done |
| Echo chamber effect discovered and documented | ✅ Done |
| Interactive dashboard for result exploration | ✅ Done |
| Feature importance analysis | ✅ Done |
| Model explainability (why predictions are made) | ✅ Done |

---

## 6. Future Plans

### Short Term (Next 2–4 Weeks)

| Plan | Expected Benefit |
|---|---|
| Combine more models together (larger ensemble) | Push accuracy closer to 93–94% |
| Add more article features (readability scores, named entities) | Small but steady improvement |
| Prepare example case studies | 10 articles showing correct and incorrect predictions |

### Medium Term (1–3 Months)

| Plan | Expected Benefit |
|---|---|
| Add time-based analysis | Understand how fake news spreads over time |
| Include images/videos from articles | Multi-modal detection for higher accuracy |
| Build a simple web demo | Upload an article → get prediction with explanation |

### Long Term (Deployment)

| Plan | Expected Benefit |
|---|---|
| Create a web API for real-time detection | Anyone can check an article through a website |
| Store the article network in a graph database | Easier exploration and querying |
| Set up monitoring | Track model performance over time |

### Target: 95% Accuracy

Based on our analysis, reaching 95% accuracy is realistic with these improvements:

| Step | Projected Accuracy |
|---|---|
| Current best model | 92.2% |
| + Larger ensemble | ~93% |
| + Time-based features | ~94% |
| + Image/video features | ~95% |

---

## 7. Conclusion

This project successfully demonstrates that **looking at relationships between articles** (not just the text) significantly improves fake news detection. The system achieves **92% accuracy** and reveals important patterns like the echo chamber effect. With planned improvements, we are on track to reach 95% accuracy.

---

*Prepared by Harsh Kanani — Semester 7 Project Update*
