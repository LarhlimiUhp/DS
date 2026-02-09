# Field Guide to Forecasting Multivariate Financial Time-Series

A compact yet comprehensive guide that mixes conceptual explanations, practical workflow advice, and pointers to concrete models and code so that you can move from raw data to a deployable forecaster (or trading signal).

---

## 1. Define the Problem Clearly

| Dimension | Examples |
|---|---|
| **Target(s)** | Prices, log-returns, volatility, drawdown risk, spreads, factors, PnL |
| **Forecast horizon** | Intraday minutes, EOD, weekly, t+20 |
| **Purpose** | Trading signal, VaR, asset-liability management, hedging, scenario generation |
| **Evaluation metric** | RMSE / MAE for accuracy; direction-of-change; Sharpe; hit-ratio; CVaR |
| **Exogenous inputs** | Macro releases, rates, sentiment, order-book features, ESG scores |

---

## 2. Data Pipeline

### 2.1 Collection

- Prices/quotes (Bloomberg / Refinitiv / crypto APIs)
- Fundamentals (Compustat)
- Macro (FRED)
- News feeds, alternative data

### 2.2 Cleaning

- De-duplication
- Corporate-action adjustments
- Timezone alignment
- Forward-fill or interpolation of small gaps
- Remove overnight or auction prints if intraday

### 2.3 Feature Engineering

- **Price transforms**: returns, log-returns, sqrt-vol, momentum, slopes
- **Technical indicators**: RSI, ATR, Bollinger width, OBV
- **Statistical**: rolling mean/sigma, z-scores, PCA factors, clustering labels
- **Cross-asset lags**: e.g. USDJPY returns lagged 1h -> Nikkei futures
- **Calendar**: day-of-week, quarter-end, FOMC days, option expiry
- **Text / sentiment**: news polarity scores, social-media buzz

### 2.4 Normalisation

- **Stationarity**: difference, percentage change, Box-Cox, log
- **Scaling**: z-score, min-max, robust scaler (fit only on train!)

### 2.5 Train / Validation / Test Split

- Non-shuffled, walk-forward or expanding-window CV
- Example: 80% train -> walk-forward 10% val -> final 10% out-of-sample

---

## 3. Model Families

### A. Classical Econometrics

1. **VAR / VARX / Bayesian VAR** - Good baseline, interpretable impulse responses
2. **VECM (Johansen)** - For cointegrated price series / spreads
3. **State-space & Kalman** - Time-varying loadings, factor models (e.g. Dynamic-Nelson-Siegel for term structure)
4. **(M)GARCH family (CCC-, DCC-, BEKK-, GO-)** - Volatility and correlation forecasting
5. **Markov-Switching / Regime VAR** - Captures bull/bear or low/high-vol regimes

### B. Machine Learning, Shallow

1. **Penalised regression** (Lasso-, Ridge-VAR) for many series
2. **Tree models** (Random Forest, XGBoost, LightGBM) with lagged features
3. **SVM** with RBF/poly kernels (works well for mid-frequency classification)
4. **k-NN / LOF** for anomaly detection, not forecasting

### C. Deep Learning

1. **RNN family** (LSTM, GRU, Bidirectional, Stacked)
2. **Sequence-to-sequence** with attention; Temporal Convolutional Networks (TCN)
3. **Transformers for time-series**: TFT, Informer, Autoformer, FEDformer
4. **Hybrid CNN-LSTM** (CNN extracts local patterns, LSTM captures long memory)
5. **Probabilistic DL**: DeepAR, DeepVAR (Amazon GluonTS), N-BEATS / N-HiTS

### D. Ensemble / Hybrid

- Combine statistical VAR with LSTM residuals
- Stack XGBoost on top of deep net features
- Average forecasts via simple mean, Bayesian model averaging, or meta-learner

---

## 4. How to Choose a Model

1. **Start simple** -> baseline: naive random-walk, last value, or VAR(1)
2. **Check data volume and dimensionality**:
   - Daily returns of 5 indices (~5x10^3 obs) -> VAR or Bayesian VAR
   - 1-second BTC/ETH prices for 2 years (~6x10^7 obs) -> scalable DL (TCN or Transformer)
3. **Do you need prediction intervals?** Choose probabilistic models (DeepAR, Bayesian VAR, GARCH)
4. **Latency constraints**: real-time may preclude heavy transformers
5. **Regulatory / interpretability**: use linear factor models or SHAP explainability on tree ensembles

---

## 5. Workflow / Example (Python)

### A. Data and Features

```python
import yfinance as yf, pandas as pd, numpy as np

tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
prices = yf.download(tickers, start='2015-01-01')['Adj Close']
rets   = np.log(prices).diff().dropna()

# lagged features up to 5 days
X = pd.concat([rets.shift(i).add_suffix(f'_lag{i}') for i in range(1, 6)], axis=1).dropna()
y = rets.loc[X.index]
```

### B. Split (expanding window)

```python
split  = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
```

### C. Fit a regularised VAR (statsmodels)

```python
from statsmodels.tsa.api import VAR

model = VAR(endog=y_train)
var_res = model.fit(maxlags=5, ic='aic')
forecast = var_res.forecast(y_train.values[-5:], steps=len(y_test))
pred_df  = pd.DataFrame(forecast, index=y_test.index, columns=y_test.columns)
```

### D. Evaluate

```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test.values, pred_df.values))
directional_acc = ((np.sign(pred_df) == np.sign(y_test)).mean()).mean()
print(f'RMSE={rmse:.5f}', f'DirAcc={directional_acc:.3%}')
```

### E. Swap in an LSTM (Keras example)

```python
import tensorflow as tf

timesteps, features = 20, len(tickers)

def create_xy(df, steps):
    X, y = [], []
    for i in range(len(df) - steps):
        X.append(df.iloc[i:i+steps].values)
        y.append(df.iloc[i+steps].values)
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_xy(rets, timesteps)
split = int(0.8 * len(X_lstm))

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(timesteps, features), return_sequences=False),
    tf.keras.layers.Dense(features)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm[:split], y_lstm[:split], epochs=20,
          validation_data=(X_lstm[split:], y_lstm[split:]))
```

### F. Probabilistic with GluonTS DeepVAR

```python
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer

train_ds = ListDataset(
    [{"target": rets[col].values, "start": str(rets.index[0])} for col in rets.columns],
    freq="1D"
)

est = DeepAREstimator(freq="1D", prediction_length=10,
                      trainer=Trainer(epochs=50))
predictor = est.train(train_ds)
```

---

## 6. Special Topics

- **Cointegration / Pairs**: Johansen test -> VECM -> hedge-ratio trading
- **Cross-section + time**: Use panel LASSO, Deep Cross-Sectional LSTM
- **Volatility focus**: Use multivariate DCC-GARCH or log-variance LSTM
- **Regime switching**: Hidden Markov (e.g. hmmlearn) or MS-VAR
- **Copulas**: Fit marginal GARCH, then model tail dependence for stress scenarios
- **Reinforcement learning**: Treat allocation as action, environment driven by forecast model
- **Transfer learning**: Pre-train on long history or similar asset class, fine-tune
- **Online learning**: Update weights incrementally (River, skmultiflow)

---

## 7. Backtesting and Risk

- Run signal generation fully out-of-sample; no peeking into future volumes/spreads
- Include realistic costs: slippage, commission, borrow fees, funding
- Measure turnover, drawdowns, max adverse excursion
- Apply White's reality check or Deflated Sharpe to avoid data-snooping bias

---

## 8. Where to Go Next

### Papers / Books

- *Forecasting Volatility and Correlation* (Engle) - multivariate GARCH
- *Time Series and Panel Data Econometrics* (Vogelsang, Lutkepohl)
- *Advances in Financial Machine Learning* (Marcos Lopez de Prado)
- *Temporal Fusion Transformers* (Lim et al., 2021)

### Libraries

`statsmodels`, `arch`, `sktime`, `darts`, `gluonts`, `prophet`, `tsfresh`, `kats`, `tensorflow-probability`, `pyflux`

### Competitions / Datasets

- Two Sigma Financial Modeling Challenge (Kaggle)
- FLA-FX, CryptoForecasting (ICML workshops)
- Numerai

---

## Key Takeaways

1. **Start with a crystal-clear objective and evaluation metric.**
2. **Build a disciplined pipeline**: data quality > model sophistication.
3. **Use simple statistical baselines first**; only escalate to deep nets if data size, horizon, or nonlinearities truly demand it.
4. **Backtest rigorously** with walk-forward CV and cost-aware metrics.
5. **Keep models updateable**; markets change faster than our code.

With this roadmap you can confidently experiment, compare models, and deploy a multivariate financial forecaster that is both statistically sound and operationally practical.
