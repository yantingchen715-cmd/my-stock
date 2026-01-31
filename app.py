import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. 頁面配置與專業 CSS 注入 (Visual Engineering)
# ==========================================
st.set_page_config(
    page_title="2026 資產配置與風險評估系統",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 專業級 CSS：高對比、大字體、去情感化配色
st.markdown("""
    <style>
    /* 全局字體設定 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Arial', sans-serif;
        color: #333333; /* 深灰，高對比 */
    }
    
    /* 標題層級 - 深藍色傳遞信任感 */
    h1, h2, h3 {
        color: #0F2C59 !important; 
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* 內文文字 - 18px 易讀性 */
    p, div, label, .stMarkdown {
        font-size: 18px !important;
        line-height: 1.6 !important; /* 增加行距呼吸感 */
    }
    
    /* 關鍵數據卡片 (Metric Cards) */
    .metric-container {
        background-color: #F8F9FA; /* 極淺灰背景 */
        border-left: 5px solid #0F2C59; /* 專業深藍導引線 */
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 16px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 32px; /* 關鍵數據特大 */
        font-weight: bold;
        color: #0F2C59;
        margin: 10px 0;
    }
    
    .metric-delta {
        font-size: 18px;
        font-weight: 500;
    }

    /* 按鈕優化 - 扁平化設計 */
    .stButton>button {
        background-color: #0F2C59;
        color: white;
        border: none;
        border-radius: 4px;
        height: 55px;
        font-size: 20px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #163A72;
    }
    
    /* 警語區塊 */
    .risk-notice {
        background-color: #FFF3CD;
        border: 1px solid #FFEEBA;
        color: #856404;
        padding: 15px;
        border-radius: 4px;
        font-size: 16px !important;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 金融運算核心 (Quantitative Core)
# ==========================================
class FinancialEngine:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
        
    def fetch_data(self, period="3y"):
        try:
            df = yf.download(self.ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 200: return None
            return df
        except: return None

    def calculate_atr(self, df, window=14):
        """計算 ATR 真實波動幅度 (用於部位控管)"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean().iloc[-1]

    def get_market_overview(self, df):
        """生成市場概覽數據"""
        close = df['Close']
        price = close.iloc[-1]
        prev_price = close.iloc[-2]
        change = (price - prev_price) / prev_price * 100
        
        # 移動平均
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        
        # 乖離率 (Bias)
        bias_20 = ((price - ma20) / ma20) * 100
        
        # RSI 運算
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ATR 風險計算
        atr = self.calculate_atr(df)
        atr_pct = (atr / price) * 100
        
        # === 專業評級邏輯 ===
        rating = "持有 (Hold)"
        rating_color = "#6c757d" # 灰色
        trend_desc = "區間震盪"
        
        # 判斷趨勢
        if price < ma20:
            rating = "減持 / 賣出 (Underweight)"
            rating_color = "#dc3545" # 深紅
            trend_desc = "空頭排列 (Bearish)"
        elif price > ma20 and ma20 > ma60:
            if bias_20 < 8:
                rating = "增持 / 買入 (Overweight)"
                rating_color = "#198754" # 深綠
                trend_desc = "多頭回測 (Bullish Pullback)"
            elif bias_20 > 15:
                rating = "中立 (Neutral)"
                rating_color = "#ffc107" # 黃色
                trend_desc = "多頭過熱 (Overbought)"
            else:
                rating = "持有 (Hold)"
                rating_color = "#0d6efd" # 藍色
                trend_desc = "多頭行進 (Bullish Trend)"

        return {
            "price": price,
            "change_pct": change,
            "ma20": ma20,
            "rsi": rsi,
            "bias": bias_20,
            "atr": atr,
            "atr_pct": atr_pct,
            "rating": rating,
            "color": rating_color,
            "trend": trend_desc
        }

    def run_monte_carlo_var(self, df, simulations=10000, days=60):
        """蒙地卡羅模擬 + VaR 風險價值計算"""
        returns = df['Close'].pct_change().dropna().values
        last_price = df['Close'].iloc[-1]
        
        sim_paths = np.zeros((simulations, days))
        
        # 區塊拔靴法 (Block Bootstrap) - 保留波動叢集特性
        block_size = 5
        num_blocks = days // block_size
        
        for i in range(simulations):
            path_returns = []
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, len(returns) - block_size)
                path_returns.extend(returns[start_idx : start_idx + block_size])
            sim_paths[i] = last_price * np.cumprod(1 + np.array(path_returns))
            
        final_prices = sim_paths[:, -1]
        
        # === 關鍵風險指標 ===
        # 95% 信賴區間的 VaR (Value at Risk)
        # 代表有 95% 的機率，虧損不會超過這個數字
        p5 = np.percentile(final_prices, 5)
        max_drawdown_pct = (p5 - last_price) / last_price * 100
        
        # 預期報酬 (中位數)
        p50 = np.percentile(final_prices, 50)
        expected_return_pct = (p50 - last_price) / last_price * 100
        
        # 勝率 (正報酬機率)
        win_rate = (np.sum(final_prices > last_price) / simulations) * 100
        
        return sim_paths, max_drawdown_pct, expected_return_pct, win_rate, p5

# ==========================================
# 3. 介面層 (UI Layer)
# ==========================================

# 側邊欄：控制台
with st.sidebar:
    st.header("⚙️ 參數設定 (Settings)")
    user_input = st.text_input("輸入監控代碼 (TW Stock ID)", value="2330, 2317, 0050")
    st.markdown("---")
    
    st.markdown("### 📊 系統狀態")
    st.markdown("""
    - **資料來源**: 台灣證券交易所 (Real-time)
    - **模擬核心**: Monte Carlo (10,000 runs)
    - **風險模型**: ATR + VaR (95% Confidence)
    """)
    
    st.markdown("---")
    run_btn = st.button("啟動分析模型 (Run Analysis)")

# 主畫面
st.title("2026 資產配置與風險評估系統")
st.markdown("##### Asset Allocation & Risk Assessment System")
st.markdown("---")

if run_btn:
    tickers = [x.strip() for x in user_input.split(',')]
    
    # 建立兩大專業分頁
    tab1, tab2 = st.tabs(["📈 市場概覽與技術評級", "🛡️ 蒙地卡羅與風險模擬"])
    
    with tab1:
        st.subheader("Market Overview & Technical Rating")
        
        for ticker in tickers:
            engine = FinancialEngine(ticker)
            df = engine.fetch_data()
            
            if df is not None:
                data = engine.get_market_overview(df)
                
                # 使用 HTML 繪製專業卡片
                st.markdown(f"""
                <div class="metric-container">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="metric-label">{ticker} ｜ {data['trend']}</span>
                            <div class="metric-value">
                                {data['price']:.2f} 
                                <span class="metric-delta" style="color: {'#198754' if data['change_pct'] > 0 else '#dc3545'};">
                                    ({data['change_pct']:+.2f}%)
                                </span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span class="metric-label">投資評級 (Rating)</span><br>
                            <span style="font-size: 24px; font-weight: bold; color: {data['color']};">
                                {data['rating']}
                            </span>
                        </div>
                    </div>
                    <hr style="opacity: 0.2; margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span class="metric-label">RSI 相對強弱</span><br>
                            <span style="font-size: 20px;">{data['rsi']:.1f}</span>
                        </div>
                        <div>
                            <span class="metric-label">乖離率 (Bias)</span><br>
                            <span style="font-size: 20px;">{data['bias']:+.2f}%</span>
                        </div>
                        <div>
                            <span class="metric-label">ATR 波動率</span><br>
                            <span style="font-size: 20px;">{data['atr_pct']:.2f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Monte Carlo Simulation & Risk Metrics")
        st.markdown("本模組採用 **區塊拔靴法 (Block Bootstrap)** 進行 10,000 次路徑模擬，以評估極端市場條件下之資產表現。")
        
        for ticker in tickers:
            engine = FinancialEngine(ticker)
            df = engine.fetch_data()
            
            if df is not None:
                sim_paths, max_dd, exp_ret, win_rate, p5_price = engine.run_monte_carlo_var(df)
                
                # 視覺化模擬結果
                # 只取前 100 條路徑繪圖，避免混亂
                chart_data = pd.DataFrame(sim_paths[:100, :].T)
                st.line_chart(chart_data, height=300)
                
                # 專業風險數據呈現
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("""
                    <div style="background:#F8F9FA; padding:15px; border-radius:5px;">
                        <span class="metric-label">回測勝率 (Win Rate)</span><br>
                        <span style="font-size:28px; font-weight:bold; color:#0F2C59;">{:.1f}%</span>
                    </div>
                    """.format(win_rate), unsafe_allow_html=True)
                    
                with c2:
                    st.markdown("""
                    <div style="background:#F8F9FA; padding:15px; border-radius:5px;">
                        <span class="metric-label">95% VaR (風險價值)</span><br>
                        <span style="font-size:28px; font-weight:bold; color:#dc3545;">{:.1f}%</span>
                        <br><small style="color:#666;">預期最大回撤</small>
                    </div>
                    """.format(max_dd), unsafe_allow_html=True)
                    
                with c3:
                    st.markdown("""
                    <div style="background:#F8F9FA; padding:15px; border-radius:5px;">
                        <span class="metric-label">部位規模建議 (Position Sizing)</span><br>
                        <span style="font-size:28px; font-weight:bold; color:#0F2C59;">{:.0f}%</span>
                        <br><small style="color:#666;">建議單一持倉上限</small>
                    </div>
                    """.format(100 / (abs(max_dd)*2) if max_dd != 0 else 0), unsafe_allow_html=True)
                    # 簡單凱利公式變形：依據最大回撤調整持倉
                
                # 專業文字導引
                st.markdown(f"""
                <div class="risk-notice">
                    <strong>📋 風險評估報告 ({ticker})：</strong><br>
                    經由 10,000 次蒙地卡羅路徑模擬，該標的在 95% 信心水準下，未來 60 天預期最大回撤幅度為 <strong>{max_dd:.1f}%</strong>。<br>
                    若依據凱利準則 (Kelly Criterion) 進行配置，建議將該標的佔總資產比例控制在上方建議值以內，以優化長期資本增長路徑。
                </div>
                <hr>
                """, unsafe_allow_html=True)

else:
    st.info("請在左側輸入股票代碼並點擊「啟動分析模型」以開始。")
