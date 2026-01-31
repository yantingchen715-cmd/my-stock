import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 設定網頁標題
st.set_page_config(page_title="2026 股市戰情室", page_icon="📈", layout="wide")

# 讓字體變大，適合閱讀
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    .stButton>button {width: 100%; height: 60px; font-size: 24px !important; background-color: #ff4b4b; color: white;}
    p, div, label {font-size: 20px !important;} 
    </style>
    """, unsafe_allow_html=True)

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
    
    def fetch_data(self):
        try:
            df = yf.download(self.ticker, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 100: return None
            return df
        except: return None

    def get_health_report(self, df):
        close = df['Close']
        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        rs = close.pct_change(20).iloc[-1] * 100
        bias = ((price - ma20) / ma20) * 100
        
        trend = "趨勢向上 (多頭) 🔥" if price > ma20 else "趨勢向下 (空頭) ❄️"
        suggestion, color = "觀望", "off"
        
        if price > ma20:
            if rs > 0:
                if bias < 5: suggestion, color = "✅ 很安全，可以買一點", "green"
                elif bias > 15: suggestion, color = "⚠️ 漲太多了，先不要追", "orange"
                else: suggestion, color = "🔵 趨勢向上，繼續抱著", "blue"
            else: suggestion, color = "🟡 雖然在漲，但比較沒力氣", "yellow"
        else: suggestion, color = "🔴 跌破月線，趕快賣掉", "red"
            
        return {"現價": price, "趨勢": trend, "動能": rs, "乖離": bias, "建議": suggestion, "顏色": color}

    def run_monte_carlo(self, df, simulations=1000, days=20):
        returns = df['Close'].pct_change().dropna().values
        last_price = df['Close'].iloc[-1]
        simulation_df = pd.DataFrame()
        end_prices = []
        
        for i in range(simulations):
            random_returns = np.random.choice(returns, days, replace=True)
            price_path = last_price * (1 + random_returns).cumprod()
            end_prices.append(price_path[-1])
            if i < 30: simulation_df[f'模擬_{i}'] = price_path
                
        return simulation_df, np.percentile(end_prices, 5), np.percentile(end_prices, 50), np.percentile(end_prices, 95)

st.title("📈 2026 股市戰情室 (永久版)")

with st.sidebar:
    st.header("👇 1. 在這裡輸入股票代碼")
    user_input = st.text_input("代碼 (例如 2330, 2317)", value="2330, 2317")
    st.write("---")
    st.header("👇 2. 按下按鈕")
    run_btn = st.button("🚀 開始診斷")

if run_btn:
    tickers = [x.strip() for x in user_input.split(',')]
    tab1, tab2 = st.tabs(["🏥 健康檢查 (能不能買?)", "🔮 未來預測 (會漲到哪?)"])
    
    with tab1:
        for ticker in tickers:
            analyzer = StockAnalyzer(ticker)
            df = analyzer.fetch_data()
            if df is not None:
                report = analyzer.get_health_report(df)
                with st.expander(f"【{ticker}】 現價: {report['現價']:.1f} 元", expanded=True):
                    if report['顏色'] == 'green': st.success(f"建議：{report['建議']}")
                    elif report['顏色'] == 'red': st.error(f"建議：{report['建議']}")
                    elif report['顏色'] == 'orange': st.warning(f"建議：{report['建議']}")
                    else: st.info(f"建議：{report['建議']}")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("趨勢狀態", report['趨勢'])
                    c2.metric("資金力道", f"{report['動能']:.1f}%")

    with tab2:
        for ticker in tickers:
            analyzer = StockAnalyzer(ticker)
            df = analyzer.fetch_data()
            if df is not None:
                sim_df, p5, p50, p95 = analyzer.run_monte_carlo(df)
                st.markdown(f"### 📍 {ticker} 未來一個月走勢圖")
                st.line_chart(sim_df, height=300)
                m1, m2, m3 = st.columns(3)
                m1.metric("運氣最差 (P5)", f"{p5:.1f}")
                m2.metric("正常情況 (P50)", f"{p50:.1f}")
                m3.metric("運氣最好 (P95)", f"{p95:.1f}")
                st.divider()
