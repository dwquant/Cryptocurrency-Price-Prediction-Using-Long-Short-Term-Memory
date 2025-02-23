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
CRYPTO_CURRENCIES = ['BTC', 'ETH', 'LTC']
AGAINST_CURRENCY = 'KRW'
LOOKBACK_DAYS = 60
START_DATE = dt.datetime(2018, 1, 1)
TEST_START_DATE = dt.datetime(2021, 1, 1)

# 전역 변수
accuracies = []

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
    df_scaled = scaler.fit_transform(df[['Close']].values)  # Close만 스케일링
    x_train, y_train = [], []

    for i in range(LOOKBACK_DAYS, len(df_scaled)):
        x_train.append(df_scaled[i - LOOKBACK_DAYS:i, 0])  # 과거 60일 종가
        y_train.append(df_scaled[i, 0])                    # 현재 종가
    
    return np.array(x_train), np.array(y_train)

def prepare_testing_data(df_total, df_test, scaler):
    """테스트 데이터를 준비합니다. (Close만 사용)"""
    model_inputs = df_total[len(df_total) - len(df_test) - LOOKBACK_DAYS:].values.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)  # 학습 데이터의 scaler 사용
    
    x_test, y_test = [], []
    for i in range(LOOKBACK_DAYS, len(model_inputs)):
        x_test.append(model_inputs[i - LOOKBACK_DAYS:i, 0])
        y_test.append(model_inputs[i, 0])
    
    return np.array(x_test), np.array(y_test)

def plot_predictions(actual, predicted, crypto, forecast=None, title_suffix=""):
    """실제값과 예측값을 그래프로 시각화합니다."""
    plt.figure(figsize=(10, 6))
    plt.plot(actual, color='black', label='Actual Prices')
    plt.plot(predicted, color='green', label='Predicted Prices')
    plt.title(f"{crypto} Price Prediction {title_suffix}")
    plt.xlabel("Time")
    plt.ylabel(f"Price ({AGAINST_CURRENCY})")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def predict_next_week(model, last_data, scaler, days=7):
    """다음 7일 가격을 예측합니다."""
    predictions = []
    real_data = last_data.copy()

    for _ in range(days):
        input_data = np.reshape(real_data, (1, LOOKBACK_DAYS, 1))
        pred = model.predict(input_data, verbose=0)
        price = scaler.inverse_transform(pred)[0][0]
        predictions.append(price)
        
        # 다음 입력을 준비
        real_data = np.roll(real_data, -1)
        real_data[-1] = pred[0]
    
    return np.array(predictions)

def analyze_crypto(crypto):
    """특정 암호화폐에 대해 예측을 수행합니다."""
    # 데이터 다운로드
    end_date = dt.datetime.now()
    df = yf.download(f'{crypto}-{AGAINST_CURRENCY}', start=START_DATE, end=end_date)
    df_test = yf.download(f'{crypto}-{AGAINST_CURRENCY}', start=TEST_START_DATE, end=end_date)
    df_total = pd.concat((df['Close'], df_test['Close']))

    # 데이터 정보 출력
    print(f"\n=== {crypto} 데이터 살펴보기 ===")
    print("데이터 미리보기:\n", df.head())
    print("\n데이터 통계:\n", df.describe())

    # 스케일러 초기화 (종가만 사용)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 학습 데이터 준비
    x_train, y_train = prepare_training_data(df, scaler)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(f"학습 샘플 수: {len(x_train)}")

    # 테스트 데이터 준비
    actual_prices = df_test['Close'].values
    x_test, y_test = prepare_testing_data(df_total, df_test, scaler)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print(f"테스트 샘플 수: {len(x_test)}")

    # 모델 학습
    model = build_lstm_model((LOOKBACK_DAYS, 1))
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

    # 예측 수행
    predicted_prices = model.predict(x_test, verbose=0)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    accuracies.append(r2_score(actual_prices, predicted_prices))

    # 결과 시각화
    plot_predictions(actual_prices, predicted_prices, crypto)

    # 다음 7일 예측
    last_data = scaler.transform(df_total[-LOOKBACK_DAYS:].values.reshape(-1, 1))
    week_forecast = predict_next_week(model, last_data, scaler)
    final_predictions = np.vstack((predicted_prices, week_forecast.reshape(-1, 1)))
    plot_predictions(actual_prices, final_predictions, crypto, title_suffix="with 7-Day Forecast")

    # 상승/하락률 계산
    latest_actual = actual_prices[-1]
    latest_predicted = predicted_prices[-1][0]
    max_forecast = np.max(week_forecast)
    min_forecast = np.min(week_forecast)
    
    upside = max(0, ((max_forecast - latest_predicted) * 100) / latest_predicted)
    downside = ((min_forecast - latest_predicted) * 100) / latest_predicted
    
    return [upside, downside]

def recommend_investment(amount, sides):
    """투자 금액 분배를 추천합니다."""
    total_upside = sum(side[0] for side in sides)
    
    if total_upside == 0:
        total_downside = sum(side[1] for side in sides)
        print(f"투자하기 좋은 시기가 아닙니다! 최대 손실률: {round(total_downside, 2)}%")
        return
    
    shares = [amount * (side[0] / total_upside) for side in sides]
    upsides = [shares[i] * sides[i][0] / 100 for i in range(3)]
    downsides = [shares[i] * sides[i][1] / 100 for i in range(3)]

    total_up = sum(upsides)
    total_down = sum(downsides)

    print("\n=== 투자 추천 ===")
    for i, crypto in enumerate(CRYPTO_CURRENCIES):
        print(f"{AGAINST_CURRENCY} {round(shares[i], 2)}를 {crypto}에 투자")
    print(f"예상 수익률: +{round((total_up / amount) * 100, 2)}% / "
          f"-{round((total_down / amount) * 100, 2)}%")

# 메인 실행
if __name__ == "__main__":
    sides = [analyze_crypto(crypto) for crypto in CRYPTO_CURRENCIES]

    # 정확도 출력
    print("\n=== 모델 성능 ===")
    for i, crypto in enumerate(CRYPTO_CURRENCIES):
        print(f"{crypto}의 R² 정확도: {accuracies[i] * 100:.2f}%")
    print(f"평균 R² 정확도: {np.mean(accuracies) * 100:.2f}%")

    # 투자 금액 입력 및 추천
    amount = float(input("\n투자 금액을 입력하세요 (KRW): "))
    recommend_investment(amount, sides)