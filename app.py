import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. 頁面配置與專業 CSS 注入
# ==========================================
st.set_page_config(
    page_title="2026 資產配置與風險評估系統",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 專業級 CSS：高對比、大字體、信任感配色
st.markdown("""
    <style>
    /* 全局設定 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Microsoft JhengHei', sans-serif;
        color: #333333;
    }
    
    /* 標題層級 */
    h1, h2, h3 {
        color: #0F2C59 !important; 
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* 內文優化 */
    p, div, label, .stMarkdown {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    
    /* 關鍵數據卡片 */
    .metric-container {
        background-color: #F8F9FA;
        border-left: 6px solid #0F2C59;
        padding: 25px;
        margin-bottom: 25px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #0F2C59;
        margin: 10px 0;
    }
    
    /* 分析師建議區塊 (新增) */
    .analyst-note {
        background-color: #E8F4F8; /* 專業淡藍 */
        border: 1px solid #D1E7ED;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 18px;
    }
    .analyst-title {
        color: #0056b3;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    
    /* 按鈕優化 */
    .stButton>button {
        background-color: #0F2C59;
        color: white;
        border: none;
        border-radius: 6px;
        height: 60px;
        font-size: 22px;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #163A72;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 金融運算核心 (含分析師邏輯)
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
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean().iloc[-1]

    def get_market_overview(self, df):
        """生成市場數據與分析師建議"""
        close = df['Close']
        price = close.iloc[-1]
        prev_price = close.iloc[-2]
        change = (price - prev_price) / prev_price * 100
        
        # 均線
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        
        # 乖離與 RSI
        bias_20 = ((price - ma20) / ma20) * 100
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # ATR 風險
        atr = self.calculate_atr(df)
        atr_pct = (atr / price) * 100
        
        # === 分析師邏輯引擎 (The Brain) ===
        rating = "持有 (Hold)"
        rating_color = "#6c757d"
        trend_desc = "區間震盪"
        
        # 預設建議
        strategy = "中立觀望"
        execution = "暫時不動作，觀察月線支撐。"
        defense = f"停損設於月線 {ma20:.1f} 元"

        if price < ma20:
            # 空頭情境
            rating = "減持 / 賣出 (Underweight)"
            rating_color = "#dc3545" # 深紅
            trend_desc = "空頭排列 (Bearish)"
            
            strategy = "防禦優先 (Capital Preservation)"
            execution = "建議降低持股水位，反彈至月線不過時應站在賣方。"
            defense = "嚴格執行停損，保留現金。"
            
        elif price > ma20 and ma20 > ma60:
            # 多頭情境
            if bias_20 < 8:
                rating = "增持 / 買入 (Overweight)"
                rating_color = "#198754" # 深綠
                trend_desc = "多頭回測 (Bullish Pullback)"
                
                strategy = "積極佈局 (Accumulate)"
                execution = f"股價回測月線有撐，建議於 {price:.1f} 元附近分批建立部位。"
                defense = f"若收盤跌破月線 {ma20:.1f} 元則短線止損。"
                
            elif bias_20 > 15:
                rating = "中立 / 止盈 (Neutral)"
                rating_color = "#ffc107" # 黃色
                trend_desc = "多頭過熱 (Overbought)"
                
                strategy = "部分獲利了結 (Profit Taking)"
                execution = "乖離過大，不建議追價。持有者可調節 30% 部位落袋為安。"
                defense = f"移動停利點上移至 10日線。"
            else:
                rating = "持有 (Hold)"
                rating_color = "#0d6efd" # 藍色
                trend_desc = "多頭行進 (Bullish Trend)"
                
                strategy = "續抱讓獲利奔跑 (Trend Following)"
                execution = "趨勢穩健，無需頻繁進出，續抱即可。"
                defense = f"波段停損守季線 {ma60:.1f} 元。"

        return {
            "price": price,
            "change_pct": change,
            "ma20": ma20,
            "rsi": rsi,
            "bias": bias_20,
            "atr_pct": atr_pct,
            "rating": rating,
            "color": rating_color,
            "trend": trend_desc,
            # 新增分析師建議包
            "advice": {
                "strategy": strategy,
                "execution": execution,
                "defense": defense
            }
        }

    def run_monte_carlo_var(self, df, simulations=10000, days=60):
        # 區塊拔靴法 (Block Bootstrap)
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
            
        final_prices = sim_paths[:, -1]
        p5 = np.percentile(final_prices, 5)
        max_dd = (p5 - last_price) / last_price * 100
        win_rate = (np.sum(final_prices > last_price) / simulations) * 100
        
        return sim_paths, max_dd, win_rate, p5

# ==========================================
# 3. 介面層 (UI Layer)
# ==========================================

# 側邊欄
with st.sidebar:
    st.header("⚙️ 參數設定 (Settings)")
    user_input = st.text_input("輸入監控代碼", value="2330, 2317, 0050")
    st.markdown("---")
    st.info("系統狀態：🟢 已連線至交易所")
    run_btn = st.button("啟動分析模型")

st.title("2026 資產配置與風險評估系統")
st.markdown("##### Asset Allocation & Risk Assessment System")
st.markdown("---")

if run_btn:
    tickers = [x.strip() for x in user_input.split(',')]
    tab1, tab2 = st.tabs(["📈 市場概覽與分析師建議", "🛡️ 風險模擬與壓力測試"])
    
    with tab1:
        st.subheader("Market Overview & Analyst Recommendations")
        
        for ticker in tickers:
            engine = FinancialEngine(ticker)
            df = engine.fetch_data()
            
            if df is not None:
                data = engine.get_market_overview(df)
                adv = data['advice']
                
                # HTML 卡片渲染
                st.markdown(f"""
                <div class="metric-container">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span class="metric-label">{ticker} ｜ {data['trend']}</span>
                            <div class="metric-value">
                                {data['price']:.2f} 
                                <span style="font-size:24px; color: {'#198754' if data['change_pct'] > 0 else '#dc3545'};">
                                    ({data['change_pct']:+.2f}%)
                                </span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span class="metric-label">綜合評級 (Rating)</span><br>
                            <span style="font-size: 26px; font-weight: bold; color: {data['color']};">
                                {data['rating']}
                            </span>
                        </div>
                    </div>
                    
                    <hr style="opacity: 0.15; margin: 20px 0;">
                    
                    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                        <div><span class="metric-label">乖離率 (Bias)</span><br><b>{data['bias']:+.2f}%</b></div>
                        <div><span class="metric-label">RSI 強弱</span><br><b>{data['rsi']:.1f}</b></div>
                        <div><span class="metric-label">ATR 波動</span><br><b>{data['atr_pct']:.2f}%</b></div>
                    </div>

                    <div class="analyst-note">
                        <div class="analyst-title">👨‍💼 首席分析師操作建議 (Chief Analyst's Note)</div>
                        <ul style="margin: 0; padding-left: 20px;">
                            <li><strong>核心策略：</strong> {adv['strategy']}</li>
                            <li style="margin-top:8px;"><strong>執行戰術：</strong> {adv['execution']}</li>
                            <li style="margin-top:8px; color:#dc3545;"><strong>風控防線：</strong> {adv['defense']}</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Monte Carlo Simulation & Stress Testing")
        
        for ticker in tickers:
            engine = FinancialEngine(ticker)
            df = engine.fetch_data()
            
            if df is not None:
                sim_paths, max_dd, win_rate, p5_price = engine.run_monte_carlo_var(df)
                
                # 繪圖
                chart_data = pd.DataFrame(sim_paths[:100, :].T)
                st.line_chart(chart_data, height=300)
                
                # 風險數據矩陣
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"""
                    <div style="background:#F8F9FA; padding:15px; border-radius:6px; border:1px solid #ddd;">
                        <span class="metric-label">歷史勝率 (Win Rate)</span><br>
                        <span style="font-size:28px; font-weight:bold; color:#0F2C59;">{win_rate:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div style="background:#F8F9FA; padding:15px; border-radius:6px; border:1px solid #ddd;">
                        <span class="metric-label">95% 風險價值 (VaR)</span><br>
                        <span style="font-size:28px; font-weight:bold; color:#dc3545;">{max_dd:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

                with c3:
                    st.markdown(f"""
                    <div style="background:#F8F9FA; padding:15px; border-radius:6px; border:1px solid #ddd;">
                        <span class="metric-label">極端支撐 (P5 Price)</span><br>
                        <span style="font-size:28px; font-weight:bold; color:#333;">{p5_price:.1f}</span>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background-color:#FFF3CD; border:1px solid #FFEEBA; color:#856404; padding:15px; border-radius:4px; margin-top:15px;">
                    <strong>風險揭露：</strong> 基於 10,000 次模擬，{ticker} 在未來 60 天內有 5% 的機率跌至 <strong>{p5_price:.1f} 元</strong> ({max_dd:.1f}%)。
                    請確保您的資產配置能承受此波動風險。
                </div>
                <hr>
                """, unsafe_allow_html=True)
else:
    st.info("系統待命模式。請在左側輸入代碼並啟動分析。")
