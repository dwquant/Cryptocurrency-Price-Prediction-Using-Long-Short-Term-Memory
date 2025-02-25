import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
import requests

# 바이낸스 API에서 BTCUSDT 데이터를 가져오는 함수
def get_binance_data(symbol, interval, start_date, end_date=None):
    if end_date is None:
        end_date = dt.datetime.now()
    
    # Unix timestamp로 변환
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    # 바이낸스 API URL
    url = f"https://api.binance.com/api/v3/klines"
    
    # 파라미터 설정
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_timestamp,
        'endTime': end_timestamp,
        'limit': 1000
    }
    
    all_data = []
    
    # 1000개 이상의 데이터를 가져오기 위한 반복
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # 마지막 데이터의 시간을 다음 요청의 시작 시간으로 설정
        params['startTime'] = data[-1][0] + 1
        
        # 종료 시간에 도달하면 반복 종료
        if params['startTime'] >= end_timestamp:
            break
    
    # 데이터프레임으로 변환
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                        'close_time', 'quote_asset_volume', 'number_of_trades',
                                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # 데이터 타입 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # 필요한 열만 선택
    df = df[['timestamp', 'close', 'volume']]
    df.rename(columns={'timestamp': 'Date', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.set_index('Date', inplace=True)
    
    return df

def create_and_fit_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)

    return model

def prediction_function(df, train_end_date=None):
    if train_end_date is None:
        # 기본값으로 데이터의 80%를 학습 데이터로 사용
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
    else:
        # 지정된 날짜를 기준으로 학습/테스트 데이터 분리
        df_train = df.loc[:train_end_date]
        df_test = df.loc[train_end_date:]

    # 스케일링을 위한 MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 종가 데이터만 스케일링
    scaled_train_data = scaler.fit_transform(df_train[['Close']].values.reshape(-1, 1))
    
    # 모델 학습을 위한 데이터 준비
    lookback = 60  # 60일 데이터로 예측
    
    x_train, y_train = [], []
    
    # x_train과 y_train 데이터 생성
    for i in range(lookback, len(scaled_train_data)):
        x_train.append(scaled_train_data[i - lookback:i, 0])
        y_train.append(scaled_train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    # 모델 생성 및 학습
    model = create_and_fit_model(x_train, y_train)
    
    # 테스트 데이터 준비
    actual_prices = df_test['Close'].values
    actual_dates = df_test.index
    
    # 테스트 데이터 스케일링
    # 입력 데이터에는 학습 데이터의 마지막 lookback일과 테스트 데이터가 포함되어야 함
    model_inputs = np.concatenate((
        scaled_train_data[-lookback:], 
        scaler.transform(df_test[['Close']].values.reshape(-1, 1))
    ), axis=0)
    
    x_test = []
    for i in range(lookback, len(model_inputs)):
        x_test.append(model_inputs[i - lookback:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # 예측
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)
    
    # 향후 7일 예측
    last_data = model_inputs[-lookback:]
    last_data = np.reshape(last_data, (1, lookback, 1))
    
    future_predictions = []
    future_dates = []
    
    last_date = df_test.index[-1] if len(df_test) > 0 else df_train.index[-1]
    
    for i in range(7):
        future_date = last_date + dt.timedelta(days=i+1)
        future_dates.append(future_date)
        
        prediction = model.predict(last_data)
        future_predictions.append(prediction[0][0])
        
        # 다음 예측을 위해 입력 데이터 업데이트
        last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)
    
    # 미래 예측값 역변환
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # R2 점수 계산
    r2 = r2_score(actual_prices, prediction_prices)
    
    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, actual_prices, color='black', label='실제 가격')
    plt.plot(actual_dates, prediction_prices, color='green', label='예측 가격')
    plt.plot(future_dates, future_predictions, color='red', label='향후 7일 예측')
    
    plt.title("BTCUSDT 가격 예측")
    plt.xlabel("날짜")
    plt.ylabel("가격 (USDT)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 예측 가격과 실제 가격을 포함한 데이터프레임 생성
    results_df = pd.DataFrame({
        '날짜': actual_dates,
        '실제가격': actual_prices,
        '예측가격': prediction_prices.flatten()
    })
    
    # 향후 예측 데이터프레임
    future_df = pd.DataFrame({
        '날짜': future_dates,
        '예측가격': future_predictions.flatten()
    })
    
    return model, scaler, results_df, future_df, r2

def main():
    # 데이터 가져오기
    start_date = dt.datetime(2017, 1, 1)
    end_date = dt.datetime.now()
    
    print("바이낸스에서 BTCUSDT 데이터를 가져오는 중...")
    df = get_binance_data('BTCUSDT', '1d', start_date, end_date)
    print(f"데이터 불러오기 완료: {len(df)} 개의 데이터")
    
    # 학습/테스트 데이터 분할 날짜
    train_end_date = dt.datetime(2023, 1, 1)
    
    print("모델 학습 및 예측 중...")
    model, scaler, results_df, future_df, r2 = prediction_function(df, train_end_date)
    
    print(f"모델 R2 점수: {r2*100:.2f}%")
    
    # 특정 날짜의 가격 확인 기능
    while True:
        try:
            date_input = input("\n날짜를 입력하세요 (YYYY-MM-DD 형식, 종료하려면 'q' 입력): ")
            
            if date_input.lower() == 'q':
                break
                
            check_date = dt.datetime.strptime(date_input, '%Y-%m-%d')
            
            # 과거 데이터에서 날짜 확인
            if check_date in results_df['날짜'].values:
                row = results_df[results_df['날짜'] == check_date].iloc[0]
                print(f"\n날짜: {row['날짜'].strftime('%Y-%m-%d')}")
                print(f"실제 가격: ${row['실제가격']:.2f}")
                print(f"예측 가격: ${row['예측가격']:.2f}")
                print(f"오차율: {abs(row['실제가격'] - row['예측가격']) / row['실제가격'] * 100:.2f}%")
            
            # 미래 예측 데이터에서 확인
            elif check_date in future_df['날짜'].values:
                row = future_df[future_df['날짜'] == check_date].iloc[0]
                print(f"\n날짜: {row['날짜'].strftime('%Y-%m-%d')}")
                print(f"예측 가격: ${row['예측가격']:.2f}")
                print("(미래 날짜의 예측 데이터입니다)")
            
            # 데이터가 없는 날짜는 가장 가까운 과거 날짜 데이터 사용
            else:
                # 입력된 날짜가 과거인지 미래인지 확인
                all_dates = pd.concat([results_df['날짜'], future_df['날짜']])
                
                if check_date < min(all_dates):
                    print(f"\n{check_date.strftime('%Y-%m-%d')}는 데이터 범위 이전 날짜입니다.")
                elif check_date > max(all_dates):
                    print(f"\n{check_date.strftime('%Y-%m-%d')}는 예측 범위를 벗어난 미래 날짜입니다.")
                else:
                    # 가장 가까운 날짜 찾기
                    all_data = pd.concat([results_df, future_df], ignore_index=True)
                    all_data['날짜차이'] = abs(all_data['날짜'] - check_date)
                    closest_row = all_data.loc[all_data['날짜차이'].idxmin()]
                    
                    print(f"\n정확한 날짜 데이터가 없습니다. 가장 가까운 날짜의 데이터를 표시합니다:")
                    print(f"날짜: {closest_row['날짜'].strftime('%Y-%m-%d')}")
                    
                    if '실제가격' in closest_row:
                        print(f"실제 가격: ${closest_row['실제가격']:.2f}")
                    
                    print(f"예측 가격: ${closest_row['예측가격']:.2f}")
        
        except ValueError:
            print("올바른 날짜 형식이 아닙니다. YYYY-MM-DD 형식으로 입력해주세요.")
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()