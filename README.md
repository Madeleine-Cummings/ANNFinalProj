Can News Headlines Predict the Stock Market?
A Deep Learning Attempt at Understanding Financial Sentiment
Financial markets move for many reasons. Some are obvious, like economic announcements or company earnings. But much of the daily movement comes from something harder to measure: the general mood or sentiment people have about the world. News headlines are one of the fastest ways that sentiment spreads.
For our final project, we wanted to explore a simple but ambitious question: Can daily financial news headlines alone predict whether the stock market will go up or down the next day?
We knew this would be difficult. Markets are famously noisy and reacting to countless factors. But we wanted to see whether machine learning models, reading only headlines, could pick up enough structure to outperform random guessing.

INTRODUCTION AND PROBLEM STATEMENT
Newsrooms publish headlines constantly. Investors react to these headlines almost instantly. The question is whether a machine learning model can do something similar: read the headlines, and guess whether a broad market index will rise or fall the next day.
More formally, our project explores:
Can a model trained only on news headlines predict next-day stock market direction?
We focused on late 2017 to late 2019, intentionally excluding the COVID period because those markets were dominated by a single global event. We tested a wide range of NLP and ML methods, including:
– TF-IDF features
– Logistic Regression
– Feed-forward neural networks
– BERT embeddings
– FinBERT sentiment scores
– A final hybrid model combining BERT + FinBERT
We evaluated all models against baselines like “always predict up” and random guessing.
METHODOLOGY AND DATA
Datasets
We used two primary data sources:
A Kaggle financial news dataset with headlines from The Guardian, CNBC, and Reuters. Each row includes a date, headline, short description (if available), and the news source. We merged headlines and descriptions into one text field.
An index dataset (MSCI Emerging Markets Index), giving daily prices and dates.
After cleaning and merging, each row had:
date
index value
headline_description
news source
future_value (the index value on a future date)
We computed pct_change as:
future_value – current_value
current_value
[FIGURE: Line plot of market index over time]
Defining Market Trend
To turn the problem into a binary classification task, we assigned labels:
If pct_change > 0 → market_trend = 1
If pct_change <= 0 → market_trend = 0
We ended up with roughly tens of thousands of headline entries between 2017–2019. A typical day had 15–120+ headlines.
[FIGURE: Plot of number of headlines per day]
Two Modeling Frameworks
We used two major strategies:
Headline-level modeling
Each headline becomes one training example. All headlines from the same day share the same label.
Daily-level modeling
All headlines from a given day are combined (either concatenated or averaged) to form a single “daily representation.”
Daily-level modeling more closely matches how markets behave (since the index moves once per day).
Models Used
We tested several model types:
TF-IDF features (20,000-word vocabulary)
Logistic Regression (as a strong baseline)
Feed-forward neural networks (dense layers with dropout)
BERT embeddings (768-dimensional vectors capturing semantic meaning)
FinBERT sentiment scores (positive, negative, neutral probabilities)
Hybrid daily model (BERT embedding + FinBERT sentiment → 771-dimensional vector)
We also used strict date-based train/test splits to prevent information leakage between days.
Baseline Models
We compared our results to:
– A majority-class predictor
– A random predictor
Any real model must beat both.
RESULTS
Baseline Performance
Majority baseline: mid-50%
Random baseline: ~50%
So anything barely above 55% is not meaningful.
Headline-Level Results
TF-IDF + Logistic Regression: mid-50%
TF-IDF + FNN: slightly higher
BERT per-headline + Logistic Regression/FNN: ~55–57%
When we enforced day-based train/test splits (no mixing of headlines from the same day), accuracy dropped slightly but stayed above baseline. This showed the models were learning some signal but the daily-level approach was more appropriate.
Daily-Level Results
This is where performance clearly improved.
TF-IDF daily document + FNN → ~60%
Average BERT embedding per day + FNN → ~61%
[FIGURE: Comparison chart of baseline vs TF-IDF vs BERT]
Hybrid BERT + FinBERT Model
Our strongest model combined:
– daily BERT embedding (768 features)
– daily FinBERT sentiment averages (3 features)
This 771-dimensional representation, fed into a logistic regression model with scaling and regularization, achieved 62–63% test accuracy.
[FIGURE: Confusion matrix for the hybrid model]
Error Analysis
We grouped predictions into:
True Positives (TP)
True Negatives (TN)
False Positives (FP)
False Negatives (FN)
Then we calculated the average pct_change in each group.
[FIGURE: Bar chart of average pct_change for TP, TN, FP, FN]
Key insight:
Correct predictions (TP and TN) tended to occur on days with larger market moves.
Incorrect predictions (FP and FN) occurred when pct_change was close to zero.
In other words, the model is better at detecting direction when the market is “meaningfully” moving, and struggles when the index barely moves.
[FIGURE: Boxplots of pct_change distributions]
DISCUSSION
What the Model Actually Learned
The model does not literally “read the news” like a human. Instead, it learns statistical relationships:
– clusters of highly positive business headlines tend to precede up days
– clusters of strongly negative macro headlines tend to precede down days
But this is noisy. Sometimes markets anticipate events before they appear in headlines. Sometimes the market reacts opposite to sentiment (buy the rumor, sell the news).
Still, the fact that our hybrid model consistently beats baselines suggests that headlines provide a weak but real predictive signal.
Limitations
We identified several limitations:
Headlines are only a fraction of the information traders use.
No intraday timing was captured.
No additional financial features (like VIX) were included.
Only a few news sources were used.
The dataset size is small compared to real-world institutional datasets.
Relation to Industry Practices
Financial NLP is a real and rapidly growing field. Hedge funds use transcripts, filings, and news to create text-based features. These provide small predictive edges, not dramatic ones. Our results fit this pattern: noisy but meaningful improvements over baseline.
CONCLUSION
So, can news headlines predict the stock market?
Not perfectly, and not well enough for serious trading.
But they do contain enough information for a machine learning model to outperform both random guessing and a naïve baseline, especially when:
– we aggregate headlines daily
– use modern embeddings like BERT
– incorporate finance-specific sentiment models like FinBERT
– and prevent train/test leakage across days
Our best model achieved around 62–63% accuracy, which is notable given how noisy daily market movements are.
More importantly, the project taught us:
– how to clean and merge large text datasets
– how to engineer labels from market index data
– how to apply TF-IDF, BERT, and FinBERT in practice
– how neural networks behave on sparse vs dense text features
– how to evaluate models properly using time-based splits
Possible extensions include:
– adding macroeconomic variables
– exploring multi-day prediction horizons
– using more news sources
– experimenting with finetuning transformer models

[FIGURE: Final pipeline diagram — “raw headlines + index data → preprocessing → embeddings + sentiment → model → evaluation”]

