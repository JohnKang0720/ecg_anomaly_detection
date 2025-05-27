import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def detect_anomalies(sample_n):
        
    ecg_train = pd.read_csv("./mitbih_train.csv", header=None)
    ecg_test = pd.read_csv("./mitbih_test.csv", header=None)

    test_signals, indices = synthetic_test_anomalies(ecg_test[ecg_test.iloc[:, -1] == 0].iloc[:10, :-1])

    channel_0 = ecg_train.iloc[sample_n, :-1].values
    normal_beats = ecg_train[ecg_train.iloc[:, -1] == 0].iloc[:, :-1]

    aggr_normal_beats = aggregate_beats(normal_beats)

    differenced = apply_differencing(aggr_normal_beats, check_stationary(aggr_normal_beats)[1]) #ARIMA performs pooly on non-stationary data, since it assumes past values are related to future values and that the past errors have patterns, but random variances would violate this.

    plot_acf_pacf(differenced)

    arima_model = ARIMA(aggr_normal_beats, order=(0,1,1))
    
    fitted = arima_model.fit()

    forecast_train = fitted.predict(start=0, end=len(channel_0)-1)
    anomalies_z = z_score(channel_0, forecast_train)
    anomalies_isolation = isolation_forest(channel_0, forecast_train)
    anomalies_rolling = rolling_stats(channel_0, forecast_train)

    forecast_test = fitted.predict(start = 0, end=len(test_signals.iloc[sample_n])-1)
    anomalies_test = z_score(test_signals.iloc[sample_n].values, forecast_test)
    
    fig, ax = plt.subplots(1,4, figsize=(12, 5))  # Optional: add figsize for better layout

    # Training data
    ax[0].plot(channel_0, label="Actual ECG Sample")
    ax[0].plot(forecast_train, label="ARIMA Predicted", alpha=0.7)
    ax[0].scatter(x=anomalies_z["idx"], y=anomalies_z["anomalies"], c="red", label="Anomalies")
    ax[0].set_title(f"ECG Sample {sample_n} and ARIMA Prediction - Z Score") 
    ax[0].legend()

    ax[1].plot(channel_0, label="Actual ECG Sample")
    ax[1].plot(forecast_train, label="ARIMA Predicted", alpha=0.7)
    ax[1].scatter(x=anomalies_isolation["idx"], y=anomalies_isolation["anomalies"], c="red", label="Anomalies")
    ax[1].set_title(f"ECG Sample {sample_n} and ARIMA Prediction - Isolation Forest") 
    ax[1].legend()

    ax[2].plot(channel_0, label="Actual ECG Sample")
    ax[2].plot(forecast_train, label="ARIMA Predicted", alpha=0.7)
    ax[2].scatter(x=anomalies_rolling["idx"], y=anomalies_rolling["anomalies"], c="red", label="Anomalies")
    ax[2].set_title(f"ECG Sample {sample_n} and ARIMA Prediction - Rolling Z") 
    ax[2].legend()

    # Test data
    ax[3].plot(test_signals.iloc[sample_n].values, label="Test ECG Sample")
    ax[3].plot(forecast_test, label="ARIMA Predicted", alpha=0.7)
    ax[3].scatter(x=anomalies_test["idx"], y=anomalies_test["anomalies"], c="red", label="Anomalies")
    ax[3].set_title("Anomaly Prediction on Test Data") 
    ax[3].axvline(x=indices[sample_n], color='black')
    ax[3].legend()

    plt.tight_layout()
    plt.show()

def plot_acf_pacf(data):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(data, lags=20, ax=ax[0])
    plot_pacf(data, lags=20, ax=ax[1])
    plt.show()

def apply_differencing(normal_beats_original, p_val):
    if p_val > 0.05:
        differenced = pd.DataFrame(normal_beats_original).diff()
        return differenced
    return normal_beats_original

def check_stationary(normal_beats):
    result_original = adfuller(normal_beats)

    print(f"ADF Statistic (Original): {result_original[0]:.4f}")
    print(f"p-value (Original): {result_original[1]:.4f}")

    return result_original

def synthetic_test_anomalies(signals):
    # signals - 10 clean ECG signals
    signals = signals.copy()
    indices = np.random.choice(signals.shape[1], size=signals.shape[0], replace=False)

    for k, i in enumerate(indices):
        signals.iloc[k, i] += np.random.rand() * 10

    return signals, indices


def aggregate_beats(beats):
    return np.sum(beats, axis=0) / beats.shape[0]

def z_score(actual, pred):
    resid = actual - pred
    z_score = (resid - np.mean(resid))/np.std(resid)
    idx = np.where(np.abs(z_score) > 0.95)

    return {"idx": idx, "anomalies": actual[idx]}

def rolling_stats(actual, pred):
    window_size = 10
    resid = pd.DataFrame({'residuals': actual - pred})
    resid['rolling_mean'] = resid['residuals'].rolling(window_size).mean()
    resid['rolling_std'] = resid['residuals'].rolling(window_size).std()
    resid = resid.dropna()
    resid['z'] = abs(resid['residuals'] - resid['rolling_mean'])/resid['rolling_std']

    anomalies = np.array(resid['z'].where(resid['z'] > 0.95).dropna().index)

    return {"idx": anomalies, "anomalies": actual[anomalies]}    

def isolation_forest(actual, pred):
    resid = np.array(actual - pred)

    resid = resid.reshape(-1, 1)
    iso = IsolationForest(contamination=0.05)
    iso.fit(resid)
    anomalies_idx = np.where(iso.predict(resid) == -1)
    return {"idx": anomalies_idx, "anomalies": actual[anomalies_idx]}

if __name__ == '__main__':
    sample_n = 1
    detect_anomalies(sample_n)
