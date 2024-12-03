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

# +
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import subprocess

import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler
st.set_page_config(layout="wide")

import pymysql
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import requests
import copy


# -

# ## Data

@st.cache_resource
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


# +
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

@st.cache_data
def find_last_val(data_dict):
    df = pd.DataFrame()
    for sheet in sheet_dict.keys():
        df[sheet] = data_dict[sheet].iloc[0,:]
    return df.transpose()

@st.cache_data
def min_max_scaling(data_dict): # 백분위상 치치
    tot_dict = {}
    scaler = MinMaxScaler()
    for key, n in date_dict.items():
        df = pd.DataFrame()
        for sheet in sheet_dict.keys():
            data = data_dict[sheet]
            sliced_data = data.iloc[:n, :]
            scaled_data = scaler.fit_transform(sliced_data)
            scaled_data_df = pd.DataFrame(scaled_data, columns = sliced_data.columns)
            df[sheet] = scaled_data_df.iloc[0,:]
        tot_dict[key] = df.round(2).transpose()
    return tot_dict

def find_high_low(df:pd.DataFrame):
    df_unstacked = df.unstack()
    top_5 = df_unstacked.nlargest(5)
    bottom_5 = df_unstacked.nsmallest(5)
    
    # (index, column) 형태로 반환
    top_5_results = [idx for idx, val in top_5.items()]
    bottom_5_results = [idx for idx, val in bottom_5.items()]
    
    return top_5_results, bottom_5_results

def high_low_data(df, dt, spread_dict, roll_dict):
    top_dict = {}
    bottom_dict = {}
    top_5, bottom_5 = find_high_low(df)
    if dt == "spread":
        data = spread_dict
    else:
        data = roll_dict
        
    for top in top_5:
        top_dict[f"{top[1]}{top[0]}"] = data[top[1]][top[0]]

    for bot in bottom_5:
        bottom_dict[f"{bot[1]}{bot[0]}"] = data[bot[1]][bot[0]]

    return pd.DataFrame(top_dict), pd.DataFrame(bottom_dict)

@st.cache_data
def load_data_from_db():
    #DB에서 데이터 불러오기
    spread_dict = {}
    roll_dict = {}
    
    for sheet in sheet_dict.keys():
        spread_dict[sheet] = read_from_sql(f"{sheet} 스왑 스프레드")
        roll_dict[sheet] = read_from_sql(f"{sheet} 스왑 롤")
    return spread_dict, roll_dict

raw_spread, raw_roll = load_data_from_db()

last_val_spread_df = find_last_val(raw_spread)
last_val_roll_df = find_last_val(raw_roll)
spread_min_max_dict = min_max_scaling(raw_spread)
roll_min_max_dict = min_max_scaling(raw_roll)

#last_val -> 실질적인 Cover 전체 데이터! [range][실제/누적백분위][spread/roll] 형태
last_val = {}
for n_range in date_dict.keys():
    spread_data = [last_val_spread_df, spread_min_max_dict[n_range]]
    roll_data = [last_val_roll_df, roll_min_max_dict[n_range]]
    last_val[n_range] = (spread_data, roll_data)

tot_data = {}
table_dict = {'spread':spread_min_max_dict, 'roll':roll_min_max_dict}

for dt in ["spread", "roll"]:
    dt_data = {}
    for n_range in ["3m", "6m", "12m"]:
        df = table_dict[dt][n_range]
        top_5, bottom_5 = high_low_data(df, dt, raw_spread, raw_roll)
        dt_data[(n_range, "Top")] = top_5
        dt_data[(n_range, "Bottom")] = bottom_5
    tot_data[dt] = dt_data

spread_cum_data = tot_data['spread']
roll_cum_data = tot_data['roll']


# +
# Cover table
# -

def highlight_top_bottom(df): # % dataframe 값 상위/하위 5개 종목씩 color mapping
    # NaN 제외한 값으로만 처리
    flat_values = df.values.flatten()
    valid_values = flat_values[~np.isnan(flat_values)]  # NaN 제거

    # 상위 및 하위 5개 값 계산
    top_5_values = valid_values[np.argsort(valid_values)[-5:]]
    bottom_5_values = valid_values[np.argsort(valid_values)[:5]]

    def apply_style(val):
        if pd.isna(val):  # NaN 값 확인
            return 'background-color: #EAEAEA'  # NaN은 회색
        elif val in top_5_values:
            return 'background-color: #FF9436'  # 상위 5개 값: 노란색
        elif val in bottom_5_values:
            return 'background-color: #6798FD'  # 하위 5개 값: 파란색
        return ''  # 나머지 값은 스타일 없음
        
    return df.style.applymap(apply_style).format(precision=2)


def style_dataframe(value_df, percent_df):
    styled = value_df.style.format(precision=2)  # Styler 객체 생성
    
    def categorize_dataframe(df):
        return df.applymap(lambda x: (int(255), int(255 - 95 * (x - 0.5) * 2), int(255 - (255 - 122) * (x - 0.5) * 2)) if x > 0.5
                           else ((int(255 + (255 - 122) * (x - 0.5) * 2),int(255 + 95 * (x - 0.5) * 2), int(255)) if pd.notna(x)
                                 else (234, 234, 234))) # %val>0.5->(255,140,120) / %val<0.5->(120,140,255) 로 그라데이션 Color Mapping (NaN은 회색)
    rgb_df = categorize_dataframe(percent_df)
    
    for row in value_df.index:
        for col in value_df.columns:
            r, g, b = rgb_df.loc[row, col]  # RGB 값 직접 읽기
            styled = styled.set_properties(
                subset=pd.IndexSlice[row, col],
                **{'background-color': f'rgb({r},{g},{b})'}
            )
    return styled


# ## Cover

st.title('Title')
main_tab, tab_1, tab_2 = st.tabs(['Cover','Chart','Analysis'])

@st.cache_resource
def macd(data:pd.DataFrame, short_window, long_window, signal_window, tot_window):
    value = data.columns[0]
    data=data.sort_index(ascending=True)
    data['Date'] = data.index
    
    data['short'] = data[value].ewm(span=short_window, adjust=False).mean()
    data['long'] = data[value].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['short'] - data['long']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    data = data.sort_index(ascending=False)
    data['Crossover'] = data['MACD'] - data['Signal']
    data['Cross_Up'] = ((data['Crossover'] < 0) & (data['Crossover'].shift(1) >= 0))
    data['Cross_Down'] = ((data['Crossover'] > 0) & (data['Crossover'].shift(1) <= 0))

    # 시계열 데이터 플롯
    fig1 = go.Figure()
    
    # Value 라인
    fig1.add_trace(go.Scatter(
        x=data["Date"], y=data[value],
        mode="lines", name=value,
        line=dict(color="grey")
    ))
    
    # Cross_Up 빨간 화살표
    for idx, row in data.loc[data["Cross_Up"]].iterrows():
        fig1.add_annotation(
            x=row["Date"],
            y=row[value],  # y좌표에 value 컬럼 사용
            ax=0,
            ay=5,  # 위쪽으로 화살표
            xanchor="center",
            yanchor="bottom",
            text="",  # 텍스트 제거
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            arrowcolor="red"
        )
    
    # Cross_Down 파란 화살표
    for idx, row in data.loc[data["Cross_Down"]].iterrows():
        fig1.add_annotation(
            x=row["Date"],
            y=row[value],  # y좌표에 value 컬럼 사용
            ax=0,
            ay=-5,  # 아래쪽으로 화살표
            xanchor="center",
            yanchor="top",
            text="",  # 텍스트 제거
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            arrowcolor="blue"
        )
    
    fig1.update_layout(
        title="Value & Cross Points",
        xaxis_title="Date",
        yaxis_title=value,
        template="plotly_white"
    )
    
    # MACD & Signal 플롯
    fig2 = go.Figure()
    
    # MACD 라인
    fig2.add_trace(go.Scatter(
        x=data["Date"], y=data["MACD"],
        mode="lines", name="MACD",
        line=dict(color="green")
    ))
    
    # Signal 라인
    fig2.add_trace(go.Scatter(
        x=data["Date"], y=data["Signal"],
        mode="lines", name="Signal",
        line=dict(color="orange")
    ))

    fig2.update_layout(
        title="MACD and Signal",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    
    # 플롯 출력
    return fig1, fig2


with main_tab: #Cover table page
    st.markdown("<h1 style='font-size: 20px;'>본드스왑 스프레드</h1>", unsafe_allow_html=True)
    for i, tenor in enumerate(['3m', '6m', '12m']):
        col1, col2 = st.columns(2)
        with col1:
            st.write(style_dataframe(last_val[tenor][0][0], last_val[tenor][0][1]))
        with col2:
            st.write(highlight_top_bottom(last_val[tenor][0][1]))
            
    st.markdown("<h3 style='font-size: 20px;'>본드스왑 롤</h3>", unsafe_allow_html=True)       
    for i, tenor in enumerate(['3m', '6m', '12m']):
        col3, col4 = st.columns(2)
        with col3:
            st.write(style_dataframe(last_val[tenor][1][0], last_val[tenor][1][1]))
        with col4:
            st.write(highlight_top_bottom(last_val[tenor][1][1]))


# ## Chart

def fig_update(fig, data, n_days):
    fig.update_layout(
        xaxis=dict(
            range=[data.index[-2007+n_days], data.index[0]]  # x축 범위를 최신 n일로 설정
        ), showlegend=True
    )


def macd_st_plot(num):
    col_list=[]
    selected_type = st.selectbox('Spread / Roll:', options=['Spread','Roll'], key=f"type_{num}_macd")
    selected_direction = st.selectbox('Top5 / Bottom5:', options=['Top','Bottom'], key=f"dir_{num}_macd")
    selected_tenor = st.multiselect("Select Range:", options=['3m','6m','12m'], key=f"tenor_{num}_macd")
    if selected_type == 'Spread':
        data = spread_cum_data
    else:
        data = roll_cum_data
    for tenor in selected_tenor:
        col_list = col_list + list(data[(tenor,selected_direction)].columns)
    unique_col_list = list(set(col_list))
    column = st.selectbox("Bond Issuer & Tenor:", options=unique_col_list, key=f"col_{num}_macd")
    # 사용자가 원하는 최근 n일 선택 (슬라이더)
    n_days = st.slider('Select the number of recent days to display:', min_value=1, max_value=800, value=60, key=f"slider_{num}_macd")
    
    for tenor in selected_tenor:
        if column in data[(tenor,selected_direction)].columns:
            data = pd.DataFrame(data[(tenor,selected_direction)][column])
            fig1, fig2 = macd(data, 3, 10, 3, tenor_dict[tenor]+1)
            fig_update(fig1, data, n_days)
            fig_update(fig2, data, n_days)     
            
            st.plotly_chart(fig1, key=f"fig1_{tenor}_{selected_direction}_{i}_{num}")
            st.plotly_chart(fig2, key=f"fig2_{tenor}_{selected_direction}_{i}_{num}")
            break
        else:
            pass


# +
tenor_dict = {'3m':60, '6m':120, '12m':252}

with tab_1: #MACD plot page
    chart_count = st.number_input('Enter the number of charts to display:', min_value=1, max_value=6, value=1)
    table_columns = st.columns(chart_count)
    for i in range(chart_count):
        with table_columns[i]:
            macd_st_plot(i)
# -

# ## Curve

# raw_spread, raw_col 이용
raw_data_dict = {'국고':'시가평가 4사평균 국고채권','통안':'시가평가 4사평균 통안증권','공사':'시가평가 4사평균 특수채 공사채 AAA','특은':'시가평가 4사평균 금융채 산금채 AAA','시은':'시가평가 4사평균 금융채 은행채 AAA','카드':'시가평가 4사평균 금융채 카드채 AA+','캐피탈':'시가평가 4사평균 기타금융채AA-'}
range_dict = {'3m':60,'6m':120,'12m':252}


# Box Plot
def box_plot_update(df):
    fig = go.Figure()
    tenors = df.columns
    for tenor in tenors:
        fig.add_trace(go.Box(
            y=df[tenor],
            name=tenor,
            marker_color='yellow',
            boxmean='sd',
            fillcolor='lightgrey',
            line=dict(color='black'),
            boxpoints=False
        ))
    fig.add_trace(go.Scatter(
        y=df.iloc[5,:],
        x=df.columns,
        mode='markers',
        marker=dict(color='mediumpurple', size=12, line_width=2),      
        name=f'1W before'
    ))
    fig.add_trace(go.Scatter(
        y=df.iloc[0,:],
        x=df.columns,
        mode='markers',
        marker=dict(color='skyblue', size=12, line_width=2),
        name=f'Current'
    ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title='Cumulative Box Plots',
        xaxis_title='Tenor',
        yaxis_title='Spread (bps)',
        template='plotly_dark',
        showlegend=True
    )
    
    return fig


def box_plot_st(num):
    col_list=[]
    selected_tenor = st.selectbox('Select Range:', options=['3m', '6m','12m'], key=f"tenor_{num}_box")
    selected_type = st.selectbox('Spread / Roll:', options=['Spread','Roll'], key=f"type_{num}_box")
    selected_column = st.selectbox('Bond Issuer:', options=['국고','통안','공사','특은','시은','카드','캐피탈'], key=f"col_{num}_box")
    if selected_type == 'Spread':
        data = raw_spread
    else:
        data = raw_roll
    range = range_dict[selected_tenor]
    raw_df = data[raw_data_dict[selected_column]].iloc[0:range, :]
    fig = box_plot_update(raw_df)
    st.plotly_chart(fig, key=f"plot_{num}_box")


def curve_update(tenors, spreads, fine_tenors, fine_spreads, fig, date): #원본 및 보간 모두 입력
    # 보간된 스프레드 곡선
    fig.add_trace(go.Scatter(
        x=fine_tenors, y=fine_spreads, mode='lines', name=f'{date} Curve',
        line=dict(width=4)
    ))
    
    # 원본 데이터 포인트
    fig.add_trace(go.Scatter(
        x=tenors, y=spreads, mode='markers', name=f'{date} Data',
        marker=dict(size=10)
    ))
    
    return fig


@st.cache_data
def interploate_curve(tenor_list, data_row, fine_tenors):
    spline = CubicSpline(tenor_list, data_row)
    fine_spreads = spline(fine_tenors)
    return fine_spreads


def curve_st(num): #current data, 
    col_list=[]
    selected_comp = st.multiselect('Select Compare Date:', options=['1w','1m','3m','6m','12m'], key=f"comp_{num}_curve")
    selected_type = st.selectbox('Spread / Roll:', options=['Spread','Roll'], key=f"type_{num}_curve")
    selected_column = st.selectbox('Bond Issuer:', options=['국고','통안','공사','특은','시은','카드','캐피탈'], key=f"col_{num}_curve")

    # 국고채만 10y, 나머지는 3y 까지를 최대 tenor로 보간
    if selected_column == '국고':
        tenor_list = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10]
    else:
        tenor_list = [0.5, 0.75, 1, 1.5, 2, 3]
    
    if selected_type == 'Spread':
        data = raw_spread
    else:
        data = raw_roll
        tenor_list = tenor_list[1:]

    fine_tenors = np.linspace(min(tenor_list), max(tenor_list), 200)
    fine_spreads_current = interploate_curve(tenor_list, data[selected_column].iloc[0,:], fine_tenors)
    
    fig = go.Figure()
    fig = curve_update(tenor_list, data[selected_column].iloc[0,:], fine_tenors, fine_spreads_current, fig, "last")
    
    comp_date_dict = {'1w':5, '1m':20, '3m': 60, '6m':120, '12m':252}
    
    for comp_date in selected_comp:
        comp_data = data[selected_column].iloc[comp_date_dict[comp_date],:]
        spline_comp = CubicSpline(tenor_list, comp_data)
        fine_spreads_comp = spline_comp(fine_tenors)
        
        fig = curve_update(tenor_list, comp_data, fine_tenors, fine_spreads_comp, fig, comp_date)
        
    # 그래프 레이아웃 설정
    fig.update_layout(
        title='Interpolated Spread Curve',
        xaxis_title='Tenor (Years)',
        yaxis_title='Spread (bps)',
        template='plotly_dark',
        showlegend=True
    )
    
    # Streamlit에 Plotly 그래프 표시
    st.plotly_chart(fig, key=f"plt_curve_{num}")


with tab_2: # chart type 구분문 넣기
    chart_count = st.number_input('Enter the number of charts to display:', min_value=1, max_value=6, value=1, key = 'Analysis')
    table_columns = st.columns(chart_count)
    for i in range(chart_count):
        with table_columns[i]:
            selected_type = st.selectbox('Curve / Box Plot:', options=['Curve','Box Plot'], key=f"type_tab2_{i}")
            if selected_type == 'Box Plot':
                box_plot_st(i)
            if selected_type == 'Curve':
                curve_st(i)
