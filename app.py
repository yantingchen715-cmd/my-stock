import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import skew

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="2026 專業操盤室 (Pro)", page_icon="📊", layout="wide")

# CSS 優化：加大字體，適合長輩閱讀
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px; border: 1px solid #d6d6d6;}
    .stButton>button {width: 100%; height: 60px; font-size: 24px !important; background-color: #d32f2f; color: white;}
    h1, h2, h3 {color: #333;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 核心邏輯：工程師的運算大腦
# ==========================================
class ProAnalyzer:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
        self.code = ticker.replace('.TW', '')

    def fetch_data(self):
        try:
            # 抓取 1 年份資料，計算指標才準
            df = yf.download(self.ticker, period="1y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 120: return None
            return df
        except: return None

    def calculate_indicators(self, df):
        # 1. 移動平均線 (MA)
        df['MA20'] = df['Close'].rolling(window=20).mean() # 月線 (生命線)
        df['MA60'] = df['Close'].rolling(window=60).mean() # 季線 (趨勢線)

        # 2. ATR 吊燈停損 (Chandelier Exit)
        # 這是最關鍵的「賣出」邏輯
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # 設定停損點：最高價 - 3倍 ATR
        df['Highest_High'] = df['High'].rolling(window=22).max()
        df['Stop_Loss'] = df['Highest_High'] - (df['ATR'] * 3.0)

        # 3. 買賣訊號標記 (Signal)
        # 買進訊號：股價 > 月線 且 乖離率 < 5% (回測支撐)
        df['Buy_Signal'] = np.where(
            (df['Close'] > df['MA20']) & 
            (df['MA20'] > df['MA60']) & 
            (((df['Close'] - df['MA20']) / df['MA20']) < 0.05) &
            (((df['Close'] - df['MA20']) / df['MA20']) > 0), 
            df['Low'] * 0.98, np.nan
        )

        # 賣出訊號：股價跌破「吊燈停損點」
        df['Sell_Signal'] = np.where(df['Close'] < df['Stop_Loss'], df['High'] * 1.02, np.nan)
        
        return df

    def get_advanced_stats(self, df):
        # 計算 Hurst 指數 (簡易版)
        try:
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(df['Close'].values[lag:], df['Close'].values[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
        except: hurst = 0.5
        
        # 計算偏態 (Skew)
        returns = df['Close'].pct_change().dropna()
        skew_val = skew(returns)
        
        # 目前狀態
        price = df['Close'].iloc[-1]
        stop_price = df['Stop_Loss'].iloc[-1]
        dist_to_stop = ((price - stop_price) / price) * 100
        
        return {
            "Hurst": hurst,
            "Skew": skew_val,
            "停損距離": dist_to_stop,
            "停損價": stop_price
        }

# ==========================================
# 前端介面：互動式 K 線圖
# ==========================================
st.title("📊 2026 專業操盤 K 線室")
st.caption("紅色箭頭賣，綠色箭頭買，紫色線是保命符")

with st.sidebar:
    st.header("👇 輸入代碼")
    user_input = st.text_input("股票代碼", value="2330, 2317, 3231")
    run_btn = st.button("🚀 啟動專業分析")

if run_btn:
    tickers = [x.strip() for x in user_input.split(',')]
    
    for ticker in tickers:
        analyzer = ProAnalyzer(ticker)
        df = analyzer.fetch_data()
        
        if df is not None:
            df = analyzer.calculate_indicators(df)
            stats = analyzer.get_advanced_stats(df)
            
            # --- 建立互動式 K 線圖 (Plotly) ---
            fig = go.Figure()

            # 1. 畫 K 線
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='K線'
            ))

            # 2. 畫月線 (黃色)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MA20'],
                line=dict(color='orange', width=1.5),
                name='月線 (20MA)'
            ))

            # 3. 畫保命停損線 (紫色)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Stop_Loss'],
                line=dict(color='purple', width=2, dash='dash'),
                name='吊燈停損線 (ATR)'
            ))

            # 4. 標記買點 (綠色三角)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Buy_Signal'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='買進訊號'
            ))

            # 5. 標記賣點 (紅色倒三角)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Sell_Signal'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='賣出訊號 (破線)'
            ))

            # 設定圖表版面
            fig.update_layout(
                title=f"<b>{ticker} 互動 K 線分析</b>",
                yaxis_title="股價",
                xaxis_rangeslider_visible=False, # 隱藏下方滑桿
                height=500,
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20)
            )

            # --- 顯示區塊 ---
            st.markdown(f"### 📍 {analyzer.code} 深度分析")
            
            # 顯示 K 線圖
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示進階數據儀表板
            c1, c2, c3, c4 = st.columns(4)
            
            # 狀態判讀
            price = df['Close'].iloc[-1]
            status = "安全持股 🟢"
            if price < stats['停損價']: status = "危險！快逃 🔴"
            
            c1.metric("目前狀態", status, f"現價 {price:.1f}")
            c2.metric("ATR 停損價", f"{stats['停損價']:.1f}", delta=f"距離 {stats['停損距離']:.1f}%")
            
            hurst_msg = "強趨勢 🔥" if stats['Hurst'] > 0.55 else "無方向 ☁️"
            c3.metric("Hurst 趨勢力", f"{stats['Hurst']:.2f}", hurst_msg)
            
            skew_msg = "有爆發力 🚀" if stats['Skew'] > 0 else "小心崩盤 ⚠️"
            c4.metric("Skew 風險偏態", f"{stats['Skew']:.2f}", skew_msg)
            
            st.divider()

        else:
            st.error(f"找不到 {ticker} 的資料。")
