# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import requests
import copy
import plotly.graph_objects as go


# +
def db_connect():
    USER = 'infomax'
    PASSWORD = "dlsvhaortm12!@"
    HOST =  "miraebond2.kro.kr"
    DATABASE = 'infomax'
    PORT = 4004
    conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, db=DATABASE, charset='utf8mb4')
    engine = create_engine(f"mysql+pymysql://{USER}:{quote_plus(PASSWORD)}@{HOST}:{PORT}/{DATABASE}")
    connection = engine.connect()
    return engine

engine = db_connect()
sheet_dict = {'국고' : '시가평가 4사평균 국고채권', 
              '통안' : '시가평가 4사평균 통안증권', 
              '특은' : '시가평가 4사평균 금융채 산금채 AAA',
              '공사' : '시가평가 4사평균 특수채 공사채 AAA', 
              '시은' : '시가평가 4사평균 금융채 은행채 AAA', 
              '카드' : '시가평가 4사평균 금융채 카드채 AA+',
              '캐피탈' : '시가평가 4사평균 기타금융채AA-'}
date_dict = {'3m' : 60,
        '6m' : 120,
        '12m' : 252}

def read_from_sql(name: str, ascending: bool = True):
    # 기본 SELECT 쿼리 생성
    query = f'SELECT * FROM `{name}`'
    order_by = None

    # order_by가 지정되면 ORDER BY 구문 추가
    if order_by:
        order_direction = 'ASC' if ascending else 'DESC'
        query += f' ORDER BY `{order_by}` {order_direction}'
    
    # SQL 쿼리 실행 및 결과 반환
    read_data = pd.read_sql(query, con=engine)
    read_data = read_data.set_index("index")
    return read_data

def load_data_from_db():
    #DB에서 데이터 불러오기
    spread_dict = {}
    roll_dict = {}
    
    for sheet in sheet_dict.keys():
        spread_dict[sheet] = read_from_sql(f"{sheet} 스왑 스프레드")
        roll_dict[sheet] = read_from_sql(f"{sheet} 스왑 롤")
    return spread_dict, roll_dict

raw_spread, raw_roll = load_data_from_db()
# -

tot_data = pd.DataFrame()
for key, val in raw_spread.items():
    for col in val.columns:
        tot_data[f'{key}{col}'] = val[col]

data = tot_data.sort_index(ascending=True)
data


# 1. MACD 신호 계산 함수
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    tot_macd = pd.DataFrame()
    tot_signal = pd.DataFrame()
    for col in data.columns:
        df = pd.DataFrame()
        df['Value'] = data[col]
        short_ema = df['Value'].ewm(span=short_period, min_periods=1, adjust=False).mean()
        long_ema = df['Value'].ewm(span=long_period, min_periods=1, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, min_periods=1, adjust=False).mean()
        tot_macd[col] = macd
        tot_signal[col] = signal_line
    return tot_macd, tot_signal


# 백분위상 계산 함수 (특정 기간 내에서 계산)
def calculate_percentile_rank(series, current_value, lookback_period):
    # 지정된 기간만큼 시계열 데이터 슬라이싱
    lookback_series = series[-lookback_period:]
    
    # 분모가 0이 될 경우 예외 처리
    if lookback_series.max() == lookback_series.min():
        return 50  # 모든 값이 동일하면 50번째 백분위수로 처리
    
    # 백분위수 계산
    percentile = 100 * (current_value - lookback_series.min()) / (lookback_series.max() - lookback_series.min())
    
    # 백분위수가 0~100 범위를 벗어날 경우 처리
    if percentile < 0:
        return 0
    elif percentile > 100:
        return 100
    else:
        return percentile


# sigma 계산 함수
def calculate_sigma(df, lookback_period):
    sigma_df = pd.DataFrame(index=df.index)
    for stock in df.columns:
        sigma_series = []
        for i in range(1,len(df)+1):
            if i < lookback_period:
                sigma_series.append(np.nan)
                continue
            series = df[stock][:i]
            lookback_series = series[-lookback_period:]
            current_sigma = lookback_series.std()
            sigma_series.append(current_sigma)
            
        sigma_df[stock] = sigma_series
        
    return sigma_df


def calculate_percentile_ranks(df, lookback_period=60):
    """
    각 시점에 대해 각 종목의 백분위상 위치를 계산하여 누적 시계열 데이터프레임으로 반환
    df: 각 종목의 시계열 데이터 (인덱스는 날짜, 컬럼은 종목명)
    lookback_period: 누적 백분위수를 계산할 기간 (예: 60일)
    """
    percentile_df = pd.DataFrame(index=df.index)  # 결과를 저장할 빈 데이터프레임 생성
    
    # 각 종목별로 백분위상 계산
    for stock in df.columns:
        percentile_series = []
        for i in range(1, len(df) + 1):  # 첫 번째부터 마지막까지
            if i < lookback_period:  # 3개월(60일) 미만일 경우 백분위수를 계산하지 않음
                percentile_series.append(np.nan)
                continue
            current_value = df[stock].iloc[i-1]
            
            # 백분위수 계산: 특정 기간 (lookback_period)까지의 데이터를 기반으로
            percentile_rank = calculate_percentile_rank(df[stock][:i], current_value, lookback_period)
            percentile_series.append(percentile_rank)
            
        percentile_df[stock] = percentile_series  # 백분위상 데이터를 컬럼으로 추가
    return percentile_df.dropna()


# +
# data
# -

# 1. 상위/하위 5개 종목 선정
def get_top_bottom_percentile_stocks(percentile_df, top_n=5, bottom_n=5):
    """
    Percentile 데이터를 기준으로 상위/하위 n개의 종목을 선택.
    percentile_df: 백분위수 데이터프레임
    top_n: 상위 n개 종목
    bottom_n: 하위 n개 종목
    """
    top_stocks = percentile_df.apply(lambda x: x.nlargest(top_n).index, axis=1)  # 각 시점별 상위 n개 종목
    bottom_stocks = percentile_df.apply(lambda x: x.nsmallest(bottom_n).index, axis=1)  # 각 시점별 하위 n개 종목
    return top_stocks, bottom_stocks


# 2. 매수/매도 신호 생성 (MACD와 Signal의 교차점 기반)
def generate_macd_signals(macd_df, signal_df):
    """
    MACD와 Signal 데이터의 교차점을 기반으로 매수/매도 신호 생성.
    macd_df: MACD 값
    signal_df: Signal 값
    """
    buy_signals = (macd_df > signal_df) & (macd_df.shift(1) <= signal_df.shift(1))  # MACD가 Signal을 상향 교차
    sell_signals = (macd_df < signal_df) & (macd_df.shift(1) >= signal_df.shift(1))  # MACD가 Signal을 하향 교차
    return buy_signals, sell_signals


# 3. 포지션 구축 및 청산
def build_and_manage_positions(top_stocks, bottom_stocks, price_df, macd_df, signal_df, percentile_df, sigma_df, n):
    positions = pd.DataFrame(index=macd_df.index, columns=percentile_df.columns)  # 포지션 데이터프레임
    positions[:] = np.nan  # 초기에는 모든 포지션을 NaN으로 설정
    current_position = {key: None for key in list(percentile_df.columns)}  # 현재 포지션
    entry_prices = {key: None for key in list(percentile_df.columns)}  # 진입 가격
    entry_sigmas = {key: None for key in list(percentile_df.columns)}  # 진입 시그마
    
    # 매수/매도 신호
    buy_signals, sell_signals = generate_macd_signals(macd_df, signal_df)
    current_position = {key: None for key in list(percentile_df.columns)}
    
    # 포지션 구축
    for i in range(1, len(percentile_df)):
        current_top_stocks = top_stocks.iloc[i]  # 현재 시점의 상위 종목들
        current_bottom_stocks = bottom_stocks.iloc[i]  # 현재 시점의 하위 종목들
        
        for stock in percentile_df.columns:
            current_price = price_df.iloc[i][stock]
            current_sigma = sigma_df.iloc[i][stock]
            
            # 현재 포지션 파악 후 청산 시그널이면  ->  이건 상/하위 종목 고려 X
            if current_position[stock] == 'long':
                if sell_signals.iloc[i][stock] or (current_price < entry_prices[stock] - n*entry_sigmas[stock]):
                    positions.loc[percentile_df.index[i], stock] = 'close'
                    current_position[stock] = None
                    entry_prices[stock] = None  # 진입 가격 초기화
                    entry_sigmas[stock] = None  # 진입 시그마 초기화
                    
            if current_position[stock] == 'short':
                if buy_signals.iloc[i][stock] or (current_price > entry_prices[stock] + n*entry_sigmas[stock]):
                    positions.loc[percentile_df.index[i], stock] = 'close'
                    current_position[stock] = None
                    entry_prices[stock] = None  # 진입 가격 초기화
                    entry_sigmas[stock] = None  # 진입 시그마 초기화
            
            if stock in current_top_stocks:  # 상위 종목이면 매수 신호로 포지션 구축
                if buy_signals.iloc[i][stock] and current_position[stock] == None:
                    positions.loc[percentile_df.index[i], stock] = 'long'  # 매수 포지션 구축
                    current_position[stock] = 'long'
                    entry_prices[stock] = current_price  # 진입 가격 저장
                    entry_sigmas[stock] = current_sigma  # 진입 시그마 저장
                    
            elif stock in current_bottom_stocks:  # 하위 종목이면 매도 신호로 포지션 구축
                if sell_signals.iloc[i][stock] and current_position[stock] == None:
                    positions.loc[percentile_df.index[i], stock] = 'short'  # 매도 포지션 구축
                    current_position[stock] = 'short'
                    entry_prices[stock] = current_price  # 진입 가격 저장
                    entry_sigmas[stock] = current_sigma  # 진입 시그마 저장
    
    return positions


def calculate_cumulative_pnl(positions, prices):
    """
    포지션 데이터와 가격 데이터를 입력받아 누적 손익을 계산하는 함수.

    Parameters:
        positions (pd.DataFrame): 포지션 정보 데이터프레임 ('long', 'short', 'close').
        prices (pd.DataFrame): 가격 정보 데이터프레임 (positions와 동일한 index와 columns).
    
    Returns:
        pd.Series: 각 시점의 누적 손익.
    """
    # 초기값 설정
    trade_details = []  # 거래 정보
    current_positions = {}  # 각 종목별 보유 포지션 상태 (매수가/매도가 기록)
    cum_profit = 0 # 누적 손익
    
    # 포지션 순회
    for date in positions.index:
        for stock in positions.columns:
            action = positions.loc[date, stock]
            price = prices.loc[date, stock] if stock in prices.columns else None
            
            if action == 'long' and price is not None:
                current_positions[stock] = ('long', price, date)  # 매수 기록
            elif action == 'short' and price is not None:
                current_positions[stock] = ('short', price, date)  # 매도 기록
            elif action == 'close' and stock in current_positions and price is not None:
                # 포지션 청산
                position_type, entry_price, entry_date = current_positions.pop(stock)
                if position_type == 'long':
                    profit = price - entry_price  # 매수 -> 매도 손익
                elif position_type == 'short':
                    profit = entry_price - price  # 매도 -> 매수 손익
                else:
                    profit = 0
                cum_profit += profit
                    
                # 거래 세부정보 저장
                trade_details.append({
                    "Stock": stock,
                    "Position" : position_type,
                    "Entry Date": entry_date,
                    "Entry Price": entry_price,
                    "Exit Date": date,
                    "Exit Price": price,
                    "Position PnL": profit,
                    "Cumulative PnL":cum_profit
                })
    return pd.DataFrame(trade_details)


def plot_pnl(trade_data):
    """
    Exit Date를 기준으로 겹치는 날짜 중 마지막 Cumulative PnL 값을 사용하여 누적 손익 그래프를 그리는 함수.

    Parameters:
        trade_data (pd.DataFrame): 거래 내역 데이터프레임. 
            Columns: ['Stock', 'Position', 'Entry Date', 'Entry Price', 
                      'Exit Date', 'Exit Price', 'Position PnL', 'Cumulative PnL'].
    
    Returns:
        None: 누적 손익 그래프를 플롯.
    """
    # Exit Date별로 마지막 Cumulative PnL 값 선택
    last_pnl_by_date = (
        trade_data.groupby('Exit Date')['Cumulative PnL']
        .last()  # 같은 Exit Date 중 마지막 값 선택
    )
    
    # Plotly 그래프 생성
    fig = go.Figure()

    # Line chart 추가
    fig.add_trace(go.Scatter(
        x=last_pnl_by_date.index,  # Exit Date
        y=last_pnl_by_date.values,  # 마지막 Cumulative PnL 값
        mode='lines+markers',
        name='Cumulative PnL',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))

    # 레이아웃 설정
    fig.update_layout(
        title="Cumulative PnL Over Time",
        xaxis_title="Exit Date",
        yaxis_title="Cumulative PnL",
        xaxis=dict(showgrid=True, tickangle=45),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        hovermode="x unified",
        width=900,
        height=500
    )
    return fig


def macd_plot(data, short, long, signal, lookback_period):
    macd, signal = calculate_macd(data, short, long, signal)
    percentile = calculate_percentile_ranks(data, lookback_period)
    sigma = calculate_sigma(data, lookback_period)
    top_5, bottom_5 = get_top_bottom_percentile_stocks(percentile, top_n=5, bottom_n=5)
    positions = build_and_manage_positions(top_5, bottom_5, data, macd, signal, percentile, sigma, 1)
    positions = positions.dropna(how='all')
    trade_hist_data = calculate_cumulative_pnl(positions, data)    
    last_pnl_by_date = (trade_hist_data.groupby('Exit Date')['Cumulative PnL'].last())

    fig = go.Figure()

    # Line chart 추가
    fig.add_trace(go.Scatter(
        x=last_pnl_by_date.index,  # Exit Date
        y=last_pnl_by_date.values,  # 마지막 Cumulative PnL 값
        mode='lines+markers',
        name='Cumulative PnL',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))

    # 레이아웃 설정
    fig.update_layout(
        title="Cumulative PnL Over Time",
        xaxis_title="Exit Date",
        yaxis_title="Cumulative PnL",
        xaxis=dict(showgrid=True, tickangle=45),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        hovermode="x unified",
        width=900,
        height=500
    )
    return fig, trade_hist_data


fig, trade_data = macd_plot(data, 3, 8, 4, 20)

trade_data


