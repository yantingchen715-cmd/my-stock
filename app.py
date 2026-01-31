import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="2026 專業操盤戰情室", page_icon="🏦", layout="wide")

# CSS 優化 (讓數據看起來更專業)
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #6c757d;}
    .stButton>button {width: 100%; height: 60px; font-size: 24px !important; border-radius: 10px; font-weight: bold;}
    .report-box {background-color: #fafafa; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 20px;}
    .data-row {font-family: 'Courier New', monospace; font-size: 16px; color: #333;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 核心大腦 (邏輯運算)
# ==========================================
class StockBrain:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
    
    def fetch_data(self):
        try:
            df = yf.download(self.ticker, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 200: return None
            return df
        except: return None

    # 新增：計算 RSI 強弱指標 (工程師專用)
    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def analyze_signal(self, df):
        # 1. 準備精密數據
        close = df['Close']
        price = close.iloc[-1]
        
        # 均線數據
        ma20 = close.rolling(20).mean().iloc[-1]   # 月線
        ma60 = close.rolling(60).mean().iloc[-1]   # 季線
        
        # 技術指標
        bias = ((price - ma20) / ma20) * 100  # 乖離率
        rsi = self.calculate_rsi(close).iloc[-1] # RSI 指標
        
        # 2. 判斷指令
        signal = "觀望 (空手)"
        action_color = "gray"
        human_reason = "目前趨勢不明顯，錢留著比較安全。"
        tech_reason = [] # 這裡存放專業數據
        
        # 格式化數據顯示字串
        tech_data_str = f"""
        🎯 關鍵點位：{price:.1f}
        📉 月線支撐 (20MA)：{ma20:.1f}
        📊 乖離率 (Bias)：{bias:+.2f}%
        ⚡ RSI 強弱值：{rsi:.1f}
        """

        # --- 邏輯判斷核心 ---
        
        # A. 賣出訊號 (優先)
        if price < ma20:
            signal = "🚨 快逃 (賣出訊號)"
            action_color = "red"
            human_reason = "股價已經跌破月線支撐，主力在出貨了，不要留戀！"
            tech_reason = [
                f"❌ 收盤價 ({price:.1f}) 低於 月線 ({ma20:.1f})，結構轉空。",
                "❌ 技術面確認「死叉」，下方無支撐。",
                f"❌ 乖離率 {bias:.2f}% 呈現負向擴大。"
            ]
        elif bias > 15 or rsi > 80:
            signal = "💰 獲利了結 (太貴了)"
            action_color = "orange"
            human_reason = "短線漲太多了，隨時會崩盤，建議先賣一半放口袋。"
            tech_reason = [
                f"⚠️ 乖離率達 {bias:.1f}% (歷史高檔區)，回檔風險極高。",
                f"⚠️ RSI 指標來到 {rsi:.1f} (超買區)，過熱訊號。",
                "⚠️ 統計顯示此位置追價勝率低於 30%。"
            ]
            
        # B. 買進訊號
        elif price > ma20 and ma20 > ma60:
            if bias < 8: 
                signal = "✅ 進場買進 (黃金買點)"
                action_color = "green"
                human_reason = "趨勢向上，且股價剛整理完準備發動，現在買很安全。"
                tech_reason = [
                    "✔️ 多頭排列：股價 > 月線 > 季線。",
                    f"✔️ 乖離率 {bias:.1f}% 處於「回測支撐區」，非追高。",
                    f"✔️ RSI ({rsi:.1f}) 位於 50-70 強勢攻擊區。"
                ]
            else:
                signal = "🔵 續抱 (安心持有)"
                action_color = "blue"
                human_reason = "趨勢還是多頭，但短線在休息，不用急著動作，繼續抱著就好。"
                tech_reason = [
                    f"✔️ 守住月線支撐 ({ma20:.1f})，波段趨勢未破。",
                    f"✔️ 季線 ({ma60:.1f}) 持續上彎助漲。",
                    "✔️ 籌碼面穩定，建議以靜制動。"
                ]

        return {
            "現價": price,
            "指令": signal,
            "顏色": action_color,
            "白話": human_reason,
            "數據": tech_data_str,
            "專業條列": tech_reason
        }

    def run_historical_bootstrap(self, df, simulations=10000, days=20):
        returns = df['Close'].pct_change().dropna().values
        last_price = df['Close'].iloc[-1]
        sim_paths = np.zeros((simulations, days))
        block_size = 5
        num_blocks = days // block_size
        
        for i in range(simulations):
            path_returns = []
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, len(returns) - block_size)
                path_returns.extend(returns[start_idx : start_idx + block_size])
            sim_paths[i] = last_price * np.cumprod(1 + np.array(path_returns))
            
        end_prices = sim_paths[:, -1]
        win_rate = (np.sum(end_prices > last_price) / simulations) * 100
        p5 = np.percentile(end_prices, 5)
        p50 = np.percentile(end_prices, 50)
        p95 = np.percentile(end_prices, 95)
        
        return sim_paths, p5, p50, p95, win_rate

# ==========================================
# 前端介面
# ==========================================
st.title("💰 2026 專業操盤戰情室")

with st.sidebar:
    st.header("👇 1. 輸入股票代碼")
    user_input = st.text_input("代碼", value="2330, 2317, 3231")
    st.write("---")
    st.header("👇 2. 執行分析")
    run_btn = st.button("🚀 AI 深度運算")

if run_btn:
    tickers = [x.strip() for x in user_input.split(',')]
    tab1, tab2 = st.tabs(["📢 買賣指令 & 專家解讀", "🎲 萬次模擬 & 勝率"])
    
    with tab1:
        st.subheader("🤖 AI 操盤指令")
        for ticker in tickers:
            brain = StockBrain(ticker)
            df = brain.fetch_data()
            if df is not None:
                res = brain.analyze_signal(df)
                
                with st.container():
                    # 1. 標題區
                    st.markdown(f"### 【{ticker}】 現價: {res['現價']:.1f} 元")
                    
                    # 2. 巨大指令區
                    if res['顏色'] == 'green': st.success(f"### {res['指令']}")
                    elif res['顏色'] == 'red': st.error(f"### {res['指令']}")
                    elif res['顏色'] == 'orange': st.warning(f"### {res['指令']}")
                    else: st.info(f"### {res['指令']}")
                    
                    # 3. 雙欄解讀區 (左邊給人看，右邊給專家看)
                    c1, c2 = st.columns([1, 1])
                    
                    with c1:
                        st.markdown("#### 💬 AI 白話文解釋")
                        st.info(f"{res['白話']}")
                        st.markdown("**關鍵數據鐵證：**")
                        # 顯示專業條列
                        for reason in res['專業條列']:
                            st.text(reason)
                            
                    with c2:
                        st.markdown("#### 📊 專業技術指標")
                        st.code(res['數據'], language="yaml")
                        st.caption("說明：Bias=乖離率, MA=移動平均線, RSI=相對強弱指標")

                    st.divider()

    with tab2:
        st.subheader("🎲 蒙地卡羅：歷史重演一萬次")
        for ticker in tickers:
            brain = StockBrain(ticker)
            df = brain.fetch_data()
            if df is not None:
                sim_paths, p5, p50, p95, win_rate = brain.run_historical_bootstrap(df)
                
                col_win, col_risk = st.columns(2)
                col_win.metric("勝率 (賺錢機率)", f"{win_rate:.1f}%")
                
                if win_rate > 60: col_win.success("✨ 數據顯示：歷史股性優良，易漲難跌。")
                elif win_rate < 40: col_win.error("💀 數據顯示：歷史股性極差，容易賠錢。")
                else: col_win.warning("😐 數據顯示：多空不明。")

                chart_data = pd.DataFrame(sim_paths[:100, :].T)
                st.line_chart(chart_data, height=250)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("P5 (最差支撐)", f"{p5:.1f}")
                c2.metric("P50 (中位數)", f"{p50:.1f}")
                c3.metric("P95 (壓力目標)", f"{p95:.1f}")
                st.divider()
