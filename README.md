# Can News Headlines Predict the Stock Market?  
_A Deep Learning Attempt at Understanding Financial Sentiment_

Financial markets move for many reasons. Some are obvious—like economic announcements or company earnings. But a lot of daily movement comes from something fuzzier: the overall mood or sentiment people have about the world.

News headlines are one of the fastest ways that sentiment spreads.

For this project, we asked a simple but ambitious question:

> **Can daily financial news headlines alone predict whether the stock market will go up or down the next day?**

We knew this would be difficult. Markets are noisy and react to countless factors. But we wanted to see whether machine learning models, reading only headlines, could pick up enough structure to beat random guessing.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Introduction & Problem Statement](#introduction--problem-statement)
- [Data](#data)
  - [News Dataset](#news-dataset)
  - [Index Dataset](#index-dataset)
  - [Label Construction](#label-construction)
- [Modeling Approaches](#modeling-approaches)
  - [Headline-Level Modeling](#headline-level-modeling)
  - [Daily-Level Modeling](#daily-level-modeling)
  - [Models Used](#models-used)
  - [Baselines](#baselines)
- [Results](#results)
  - [Baseline Performance](#baseline-performance)
  - [Headline-Level Results](#headline-level-results)
  - [Daily-Level Results](#daily-level-results)
  - [Hybrid BERT + FinBERT Model](#hybrid-bert--finbert-model)
- [Error Analysis](#error-analysis)
- [Discussion](#discussion)
  - [What the Model Learned](#what-the-model-learned)
  - [Limitations](#limitations)
  - [Relation to Industry Practices](#relation-to-industry-practices)
- [Conclusion](#conclusion)
- [Possible Extensions](#possible-extensions)
- [Pipeline Overview](#pipeline-overview)

---

## Project Overview

- **Goal:** Predict the **next-day direction** (up or down) of a market index using **only daily financial news headlines**.
- **Period:** Late 2017 to late 2019  
  - We **intentionally excluded the COVID period**, since those markets were dominated by an unusual global shock.
- **Target:** A broad market index (MSCI Emerging Markets Index).
- **Main Question:**

  > Can a model trained only on news headlines predict next-day stock market direction?

- **Key Methods:**
  - TF-IDF features
  - Logistic Regression
  - Feed-forward neural networks
  - BERT embeddings
  - FinBERT sentiment scores
  - A hybrid model combining BERT + FinBERT

- **Evaluation:**  
  Models were evaluated against simple baselines:
  - Always predict “up”
  - Random guessing

---

## Introduction & Problem Statement

Newsrooms publish headlines constantly. Investors react to these headlines almost instantly. We wanted to see if a machine learning model could do something similar:

> Read the headlines for a given day and predict whether a broad market index will rise or fall the **next day**.

Formally:

- **Input:** All financial news headlines for day *t*.
- **Output:** A binary label indicating whether the index went **up (1)** or **down / flat (0)** on day *t+1*.
- **Constraint:** The model only sees **textual data from headlines and descriptions**, not prices or other features at inference time.

We ensured realistic evaluation by using **strict date-based train/test splits** to avoid information leakage across days.

---

## Data

### News Dataset

- Source: A **Kaggle financial news dataset** with headlines from:
  - The Guardian
  - CNBC
  - Reuters
- Each row included:
  - `date`
  - `headline`
  - `short_description` (if available)
  - `news_source`
- We merged `headline` and `short_description` into a single text field:
  - `headline_description`

After cleaning and merging:

- Each row represented a headline (or headline + description).
- A typical day had **15–120+ headlines** in the 2017–2019 period.

> **Figure (placeholder):** Line plot of the market index over time.  
> `![Market index over time](path/to/index_plot.png)`

> **Figure (placeholder):** Plot of number of headlines per day.  
> `![Number of headlines per day](path/to/headline_count_plot.png)`

### Index Dataset

- Index: **MSCI Emerging Markets Index**
- Data fields:
  - `date`
  - `index_value` (daily close)

We merged the index data with the news dataset on `date`.

### Label Construction

We defined a **future value** and **percentage change**:

- `future_value` = index value on the **next trading day**
- `pct_change = (future_value – current_value) / current_value`

We converted this into a **binary classification** task:

- If `pct_change > 0` → `market_trend = 1` (up day)
- If `pct_change <= 0` → `market_trend = 0` (down or flat)

This produced **tens of thousands of labeled headline entries** between 2017–2019.

---

## Modeling Approaches

We used two main modeling frameworks.

### Headline-Level Modeling

- **Each headline = one training example.**
- All headlines from the same day share the **same label** (`market_trend` for the next day).
- Pros:
  - More data points.
- Cons:
  - Ignores the fact that markets move at the **daily** level, not per headline.

### Daily-Level Modeling

- **All headlines for a given day are aggregated** into a single daily representation.
- Approaches included:
  - Concatenating all text into a “daily document.”
  - Averaging embeddings (e.g., average BERT vector for all headlines in a day).
- Pros:
  - Better matches how the index actually moves (once per day).
  - More aligned with the true prediction target.

### Models Used

We tested a range of models and representations:

- **TF-IDF Features**
  - Vocabulary size: ~20,000 words
- **Logistic Regression**
  - Used as a strong linear baseline.
- **Feed-Forward Neural Networks (FNN)**
  - Dense layers
  - Dropout for regularization
- **BERT Embeddings**
  - 768-dimensional embeddings capturing semantic meaning
- **FinBERT Sentiment**
  - Finance-specific sentiment model
  - Produced probabilities for:
    - Positive
    - Negative
    - Neutral
- **Hybrid Daily Model**
  - Concatenated:
    - Daily BERT embedding (768 features)
    - Daily average FinBERT sentiment probabilities (3 features)
  - Final vector: **771 dimensions**
  - Classifier: Logistic regression with scaling and regularization.

### Baselines

We compared against:

- **Majority-class predictor**
  - Always predicts the more frequent class (typically “up”).
- **Random predictor**
  - Predicts up/down with equal probability.

Any model needs to significantly outperform **~50–55%** accuracy to be considered meaningful.

We also enforced **date-based train/test splits** to avoid leaking information between days (e.g., headlines from the same day appearing in both train and test).

---

## Results

### Baseline Performance

- **Majority baseline:** mid-50% accuracy
- **Random baseline:** ~50% accuracy

Conclusion: Any score **barely above 55%** is not especially meaningful.

---

### Headline-Level Results

- **TF-IDF + Logistic Regression:** mid-50% accuracy
- **TF-IDF + FNN:** slightly higher than logistic regression
- **BERT (per-headline) + Logistic Regression/FNN:** ~55–57% accuracy

When we enforced **strict day-based splits** (ensuring headlines from the same day never appear in both train and test):

- Accuracy dropped slightly but remained above baseline.
- This indicated the models were learning some signal, but headline-level modeling was not the best match to the problem.

---

### Daily-Level Results

This is where performance clearly improved.

- **TF-IDF daily document + FNN:** ~60% accuracy
- **Average BERT embedding per day + FNN:** ~61% accuracy

> **Figure (placeholder):** Comparison chart of baseline vs TF-IDF vs BERT.  
> `![Model performance comparison](path/to/comparison_chart.png)`

Daily aggregation made it easier for the models to capture the **overall sentiment and topic mix** for each day.

---

### Hybrid BERT + FinBERT Model

Our **strongest model** combined:

- Daily BERT embedding (768 features)
- Daily FinBERT sentiment averages (3 features)

This 771-dimensional vector, fed into **logistic regression with scaling and regularization**, achieved:

- **Test accuracy: ~62–63%**

> **Figure (placeholder):** Confusion matrix for the hybrid model.  
> `![Confusion matrix](path/to/confusion_matrix.png)`

Given the noise in daily markets, this performance is **non-trivial** and consistently beat both baselines.

---

## Error Analysis

We grouped predictions into:

- **True Positives (TP)** – predicted up, actual up  
- **True Negatives (TN)** – predicted down, actual down  
- **False Positives (FP)** – predicted up, actual down  
- **False Negatives (FN)** – predicted down, actual up  

For each group, we computed the **average `pct_change`**:

> **Figure (placeholder):** Bar chart of average `pct_change` for TP, TN, FP, FN.  
> `![Average pct_change by prediction group](path/to/pct_change_bar.png)`

Key insight:

- **Correct predictions (TP and TN)** tended to occur on days with **larger index moves**.
- **Incorrect predictions (FP and FN)** were concentrated on days when `pct_change` was close to **zero**.

In other words:

- The model is better at detecting direction when the market is **meaningfully moving**.
- It struggles when the index is essentially flat.

> **Figure (placeholder):** Boxplots of `pct_change` distributions for each group.  
> `![pct_change distributions](path/to/pct_change_boxplots.png)`

---

## Discussion

### What the Model Learned

The model does **not** “read the news” like a human. Instead, it captures **statistical patterns** in the text:

- Clusters of strongly **positive business headlines** tend to precede **up days**.
- Clusters of strongly **negative macro headlines** tend to precede **down days**.

However:

- Markets sometimes **anticipate** events before headlines appear.
- Sometimes the market reacts **opposite** to sentiment (“buy the rumor, sell the news”).
- Headlines represent only a **slice** of the information ecosystem driving prices.

Still, the fact that our **hybrid model** consistently beat baselines suggests:

> News headlines contain a **weak but real** predictive signal about next-day market direction.

---

### Limitations

Some key limitations of this project:

- **Headlines only**
  - Real traders also use price data, order flow, macro indicators, etc.
- **No intraday timing**
  - We treated all headlines in a day as if they arrived at the same time.
- **No additional financial features**
  - No volatility (e.g., VIX), volume, or technical indicators.
- **Limited news sources**
  - Only The Guardian, CNBC, and Reuters.
- **Data size**
  - Small compared to institutional datasets that may span decades and many asset classes.

---

### Relation to Industry Practices

Financial NLP is an active and growing field. In practice:

- Hedge funds and trading firms use:
  - Earnings call transcripts
  - Regulatory filings
  - Social media
  - News and blogs
- These text features usually provide **small predictive edges**, not huge advantages on their own.

Our findings fit this pattern:

- The models deliver **noisy but meaningful** improvements over baselines.
- This is closer to a **small alpha signal** than a stand-alone trading strategy.

---

## Conclusion

So, **can news headlines predict the stock market?**

- **Not perfectly**, and not well enough to build a serious trading strategy on headlines alone.
- But they **do** carry enough signal for a reasonably designed ML model to:

  - Beat **random guessing**
  - Beat a **naïve majority baseline**
  - Especially when:
    - Headlines are **aggregated at the daily level**
    - We use **modern embeddings** (BERT)
    - We incorporate **finance-specific sentiment** (FinBERT)
    - We enforce **time-based splits** to avoid leakage

Our best model:

- **Hybrid BERT + FinBERT daily representation**
- Achieved **~62–63% test accuracy**

Given how noisy daily market movements are, that’s a **notable result**.

More importantly, the project taught us how to:

- Clean and merge **large text datasets**
- Engineer **labels** from market index data
- Apply **TF-IDF, BERT, and FinBERT** in practice
- Understand how **neural networks** behave on sparse vs dense text features
- Evaluate models properly using **time-based validation**

---

## Possible Extensions

If we continued this project, we would explore:

- **Adding macroeconomic variables**
  - Interest rates, inflation data, economic surprise indices, etc.
- **Multi-day prediction horizons**
  - Predict returns over 3–5 days instead of just the next day.
- **More news sources**
  - Other outlets, social media, or alternative data.
- **Finetuning transformer models**
  - Finetune BERT/FinBERT directly on our financial prediction task.
- **Alternative targets**
  - Predicting volatility, large-move days, or sector-specific indices.

---

## Pipeline Overview

> **Figure (placeholder): Final pipeline diagram**  
> `![Final pipeline](path/to/pipeline_diagram.png)`

High-level pipeline:

1. **Raw Data**
   - Financial news headlines + descriptions
   - Market index prices

2. **Preprocessing**
   - Cleaning text
   - Merging headlines with index data by date
   - Computing `pct_change` and binary labels

3. **Feature Engineering**
   - TF-IDF features
   - BERT embeddings
   - FinBERT sentiment scores
   - Daily aggregation (concatenation or averaging)

4. **Modeling**
   - Logistic regression
   - Feed-forward neural networks
   - Hybrid BERT + FinBERT model

5. **Evaluation**
   - Time-based train/test split
   - Baseline comparison
   - Confusion matrices and error analysis
   - Analysis of performance vs. magnitude of market moves

---

*End of README*
