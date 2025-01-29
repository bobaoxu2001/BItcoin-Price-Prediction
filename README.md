# Predicting the Unpredictable: Deep Learning for Bitcoin Price Dynamics

## Authors
- **Ao Xu** ([ax2183@nyu.edu](mailto:ax2183@nyu.edu))
- **Sam Lai** ([jl12560@nyu.edu](mailto:jl12560@nyu.edu))
- **Zexuan Yang** ([zy3035@nyu.edu](mailto:zy3035@nyu.edu))
- **Yichao Yang** ([yy5020@nyu.edu](mailto:yy5020@nyu.edu))

## Overview

This project explores the application of **deep learning models** for **Bitcoin price prediction**, integrating **traditional financial market data** with **social media sentiment analysis**. The study spans Bitcoin market trends from **2016 to July 2024**, utilizing state-of-the-art time series forecasting models to improve prediction accuracy.

## Why Bitcoin?

Bitcoin is a **highly volatile** and **decentralized** digital asset that has gained global recognition since its inception in 2009. Unlike traditional currencies, its value is influenced by factors such as:
- Market demand and supply dynamics
- External events and regulatory changes
- Sentiment trends on social media platforms like **Twitter (X), Reddit, and Bitcointalk**

By leveraging a combination of **market indicators** (e.g., Nasdaq, VIX, and Gold prices) and **social media sentiment metrics**, our project aims to enhance the accuracy of Bitcoin price predictions.

---

## Data Sources & Preparation

### **Dataset Overview**
Our dataset integrates multiple sources to support both **regression** and **classification** tasks. The time range covered is **2016-11-01 to 2024-08-22**.

- **Market Data**:
  - **Bitcoin Price**: Hourly closing prices
  - **Nasdaq OHLCV**: Daily stock market data
  - **Gold OHLCV**: Daily gold price data
  - **VIX Index**: Daily market volatility index (fear gauge)

- **Sentiment Data**:
  - Hourly sentiment scores from **Twitter, Reddit, and Bitcointalk**

### **Data Preprocessing**
- Daily data was aligned with **hourly granularity** using **forward filling**.
- Missing values (e.g., weekends) were handled using **backward filling**.
- Moving average gaps were addressed with **forward filling**.

---

## Exploratory Analysis

### **Key Insights**
1. **Correlation Analysis**
   - Bitcoin's closing price shows a **strong correlation** with Nasdaq trends.
   - Negative sentiment from Twitter and Bitcointalk correlates **negatively** with Bitcoin price.

2. **Autocorrelation (ACF & PACF)**
   - Helps identify **optimal lag features** for forecasting.
   - Suggests **ARIMA(1,1,1)** as a viable time-series model.

3. **Volatility Analysis**
   - Identifies **structural breaks** and **regime changes** in Bitcoin’s price behavior.

---

## Model Development

### **Baseline Models**
We implemented traditional models as a benchmark:
- **ARIMA**: Classical time-series forecasting
- **XGBoost**: Gradient-boosting framework for time-series
- **LightGBM**: Efficient tree-based model

### **Proposed Deep Learning Models**
We experimented with advanced architectures inspired by **NeurIPS and ICLR research**:
- **SE-GRN** (Squeeze-and-Excitation Gated Recurrent Network)
- **iTransformer** (Inverted Transformer for Time Series Forecasting)
- **Times-FM** (Google’s Time-Series Foundation Model)
- **SOFTS** (Efficient Multivariate Forecasting with Series-Core Fusion)
- **CNN-LSTM** (Combining CNNs for feature extraction with LSTM for temporal learning)

---

## Model Results & Performance

| **Model**       | **MAE (Mean Absolute Error)** | **MSE (Mean Squared Error)** |
|----------------|-----------------------------|-----------------------------|
| LightGBM      | 254.37                        | 258,918.80                  |
| ARIMA         | 1602.19                        | 4,847,913.23                |
| XGBoost       | 258.53                         | 271,929.61                  |
| **SE-GRN**    | 2377.06                        | 883,273                     |
| **iTransformer** | 1948.00                     | 8,123,500                    |
| **Times-FM**   | 2672.28                        | 3,600,000                    |
| **SOFTS**      | **304.23**                     | **183,679.47**               |
| **CNN-LSTM**   | 1941.32                        | 6,730,000                    |

### **Key Findings**
- **SOFTS outperformed** all other models, achieving the lowest **MAE** and **MSE**.
- **Times-FM and iTransformer performed poorly**, indicating that model complexity **does not always** improve forecasting.
- **Sentiment analysis enhanced model predictions**, especially when combined with traditional market data.

---

## Visualizations

We generated six types of visual analyses:
1. **Full timeline predictions** – capturing long-term trends.
2. **Single prediction window** – showing short-term accuracy.
3. **Time-series validation windows** – demonstrating model robustness.
4. **Returns comparison** – evaluating predicted vs actual daily returns.
5. **Error distribution plots** – highlighting prediction variance.
6. **Timeline of error percentages** – tracking fluctuations over time.

---

## Conclusion & Future Work

### **Key Takeaways**
1. **Feature fusion is crucial** – combining financial data with social media sentiment **improves accuracy**.
2. **Simpler models can outperform complex models** – SOFTS achieved the best results, while Times-FM struggled.
3. **Social media sentiment plays a role** – market sentiment contributes meaningfully to Bitcoin price dynamics.

### **Future Enhancements**
- **Real-time data integration**: Improve forecasting accuracy with live data streams.
- **External event impact**: Explore correlations between **global events (e.g., U.S. elections, policy changes)** and Bitcoin trends.
- **Alternative data sources**: Incorporate **news headlines, blockchain activity, and high-frequency trading data**.

---

## Getting Started

### **Installation & Dependencies**
To run the models, install the necessary libraries:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost tensorflow keras matplotlib
