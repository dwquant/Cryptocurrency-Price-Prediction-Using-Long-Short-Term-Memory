import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime as dt

# 상수 정의
CRYPTO_CURRENCIES = ['BTC']
AGAINST_CURRENCY = 'KRW'
LOOKBACK_DAYS = 60
START_DATE = dt.datetime(2018, 1, 1)
TEST_START_DATE = dt.datetime(2021, 1, 1)

# 전역 변수
accuracies = []
predictions_dict = {}

def build_lstm_model(input_shape):
    """LSTM 모델을 생성하고 컴파일합니다."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_training_data(df, scaler):
    """학습 데이터를 준비합니다. (Close만 사용)"""
    df_scaled = scaler.fit_transform(df[['Close']].values)
    x_train, y_train = [], []

    for i in range(LOOKBACK_DAYS, len(df_scaled)):
        x_train.append(df_scaled[i - LOOKBACK_DAYS:i, 0])
        y_train.append(df_scaled[i, 0])
    
    return np.array(x_train), np.array(y_train)

def prepare_testing_data(df_total, df_test, scaler):
    """테스트 데이터를 준비합니다. (Close만 사용)"""
    model_inputs = df_total[len(df_total) - len(df_test) - LOOKBACK_DAYS:].values.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    
    x_test, y_test = [], []
    for i in range(LOOKBACK_DAYS, len(model_inputs)):
        x_test.append(model_inputs[i - LOOKBACK_DAYS:i, 0])
        y_test.append(model_inputs[i, 0])
    
    return np.array(x_test), np.array(y_test)

def plot_predictions(actual, predicted, crypto, dates, title_suffix=""):
    """실제값과 예측값을 그래프로 시각화합니다."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, color='black', label='Actual Prices')
    plt.plot(dates, predicted, color='green', label='Predicted Prices')
    plt.title(f"{crypto} Price Prediction {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({AGAINST_CURRENCY})")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_crypto(crypto):
    """특정 암호화폐에 대해 예측을 수행합니다."""
    end_date = dt.datetime.now()
    df = yf.download(f'{crypto}-{AGAINST_CURRENCY}', start=START_DATE, end=end_date)
    df_test = yf.download(f'{crypto}-{AGAINST_CURRENCY}', start=TEST_START_DATE, end=end_date)
    df_total = df['Close']  # 중복 제거, 전체 데이터 사용

    print(f"\n=== {crypto} 데이터 살펴보기 ===")
    print("데이터 미리보기:\n", df.head())
    print("\n데이터 통계:\n", df.describe())

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, y_train = prepare_training_data(df, scaler)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(f"학습 샘플 수: {len(x_train)}")

    actual_prices = df_test['Close'].values
    x_test, y_test = prepare_testing_data(df_total, df_test, scaler)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print(f"테스트 샘플 수: {len(x_test)}")

    model = build_lstm_model((LOOKBACK_DAYS, 1))
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

    predicted_prices = model.predict(x_test, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_prices).flatten()
    accuracies.append(r2_score(actual_prices, predicted_prices))

    test_dates = df_test.index[LOOKBACK_DAYS:]  # 테스트 기간 날짜
    predictions_dict[crypto] = {
        'dates': test_dates,
        'actual': actual_prices,
        'predicted': predicted_prices
    }

    plot_predictions(actual_prices, predicted_prices, crypto, test_dates)
    return model, scaler, df_total

def get_price_on_date(crypto, date_str):
    """입력된 날짜의 실제 가격과 예측 가격을 반환합니다."""
    try:
        target_date = pd.to_datetime(date_str)
        if crypto not in predictions_dict:
            print(f"{crypto}에 대한 예측 데이터가 없습니다.")
            return
        
        data = predictions_dict[crypto]
        idx = np.where(data['dates'].date == target_date.date())[0]
        
        if len(idx) == 0:
            print(f"{date_str}에 대한 데이터가 테스트 범위({TEST_START_DATE.date()} ~ {dt.datetime.now().date()})에 없습니다.")
            return
        
        idx = idx[0]
        actual_price = float(data['actual'][idx])
        predicted_price = float(data['predicted'][idx])
        
        print(f"\n{crypto} - {date_str}:")
        print(f"실제 가격: {AGAINST_CURRENCY} {actual_price:,.2f}")
        print(f"예측 가격: {AGAINST_CURRENCY} {predicted_price:,.2f}")
        print(f"차이: {AGAINST_CURRENCY} {actual_price - predicted_price:,.2f} "
              f"({(actual_price - predicted_price) / actual_price * 100:.2f}%)")
    
    except ValueError:
        print("잘못된 날짜 형식입니다. 'YYYY-MM-DD' 형식을 사용하세요.")

if __name__ == "__main__":
    for crypto in CRYPTO_CURRENCIES:
        analyze_crypto(crypto)

    print("\n=== 모델 성능 ===")
    for i, crypto in enumerate(CRYPTO_CURRENCIES):
        print(f"{crypto}의 R² 정확도: {accuracies[i] * 100:.2f}%")
    print(f"평균 R² 정확도: {np.mean(accuracies) * 100:.2f}%")

    while True:
        date_input = input("\n확인하고 싶은 날짜를 입력하세요 (YYYY-MM-DD, 종료하려면 'exit'): ")
        if date_input.lower() == 'exit':
            break
        
        crypto_input = 'BTC'
        if crypto_input not in CRYPTO_CURRENCIES:
            print("잘못된 암호화폐입니다. BTC, ETH, LTC 중 하나를 선택하세요.")
            continue
        
        get_price_on_date(crypto_input, date_input)