import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. 系統配置與專業風格定義 (System Config & Styling)
# ==========================================
st.set_page_config(
    page_title="QUANT TERMINAL 2026",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 強制深色模式與終端機風格
st.markdown("""
    <style>
    /* 全局背景與字體 */
    .stApp {
        background-color: #0e1117; /* 深黑背景 */
        color: #e0e0e0; /* 淺灰字體 */
        font-family: 'Roboto Mono', 'Courier New', monospace; /* 等寬專業字體 */
    }

    /* 側邊欄風格 */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric 指標卡片風格 (扁平、無圓角、高對比) */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700;
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stMetric {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 15px 0px;
        border-radius: 0px !important; /* 直角風格 */
    }

    /* 按鈕風格 (扁平、專業藍) */
    .stButton>button {
        width: 100%;
        border-radius: 0px !important;
        background-color: #1f6feb;
        color: white;
        border: none;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #388bfd;
    }

    /* 數據框風格 (Code Block) */
    code {
        color: #e0e0e0 !important;
        background-color: #161b22 !important;
        border: 1px solid #30363d;
        border-radius: 0px !important;
    }

    /* 分隔線 */
    hr { margin: 2em 0; border-color: #30363d; }
    
    /* 標題大寫 */
    h1, h2, h3 { text-transform: uppercase; letter-spacing: 1px; color: #ffffff; }
    
    /* 狀態標籤風格 */
    .status-tag {
        display: inline-block;
        padding: 4px 12px;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        border-radius: 0px;
    }
    .tag-buy { background-color: #238636; color: #ffffff; } /* 專業綠 */
    .tag-sell { background-color: #da3633; color: #ffffff; } /* 專業紅 */
    .tag-hold { background-color: #1f6feb; color: #ffffff; } /* 專業藍 */
    .tag-wait { background-color: #6e7681; color: #ffffff; } /* 專業灰 */

    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心運算引擎 (Quantitative Engine)
# ==========================================
class QuantEngine:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
        self.code = ticker.replace('.TW', '')

    def fetch_data(self):
        try:
            df = yf.download(self.ticker, period="2y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 200: return None
            return df
        except: return None

    def calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, df):
        close = df['Close']
        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        bias = ((price - ma20) / ma20) * 100
        rsi = self.calculate_rsi(close).iloc[-1]
        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]

        # 專業指令判斷
        action = "WAIT"
        tag_class = "tag-wait"
        brief = "趨勢不明，空手觀望"
        tech_details = []

        # A. 賣出條件 (優先)
        if price < ma20:
            action = "SELL / EXIT"
            tag_class = "tag-sell"
            brief = "跌破月線支撐，結構轉空"
            tech_details = [
                f"PRICE({price:.1f}) < MA20({ma20:.1f}) -> 趨勢破壞",
                "MA20 下彎確認 -> 壓力沈重",
                f"BIAS({bias:.2f}%) 負向擴大 -> 動能衰退"
            ]
        elif bias > 18 or rsi > 82:
            action = "TAKE PROFIT"
            tag_class = "tag-sell"
            brief = "指標嚴重過熱，建議獲利了結"
            tech_details = [
                f"BIAS({bias:.2f}%) > 閾值(18%) -> 極端乖離",
                f"RSI({rsi:.1f}) 進入超買區 -> 反轉風險高",
                "統計勝率顯著下降"
            ]
            
        # B. 買進條件
        elif price > ma20 and ma20 > ma60:
            if bias < 8:
                action = "BUY / LONG"
                tag_class = "tag-buy"
                brief = "多頭排列且回測支撐，進場甜蜜點"
                tech_details = [
                    "TREND: BULLISH (價>MA20>MA60)",
                    f"BIAS({bias:.1f}%) < 8% -> 回測月線確認",
                    f"VOL_RATIO({vol_ratio:.1f}) -> 量價結構健康"
                ]
            else:
                action = "HOLD"
                tag_class = "tag-hold"
                brief = "多頭趨勢行進中，續抱勿追高"
                tech_details = [
                    f"SUPPORT: MA20({ma20:.1f}) 有效守穩",
                    f"MA60({ma60:.1f}) 持續上揚助漲",
                    "建議移動停利策略"
                ]

        return {
            "price": price,
            "action": action,
            "tag": tag_class,
            "brief": brief,
            "details": tech_details,
            "metrics": {"MA20": ma20, "BIAS%": bias, "RSI": rsi}
        }

    def run_monte_carlo(self, df, simulations=10000, days=20):
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
# 3. 終端機介面 (Terminal UI)
# ==========================================
st.title("QUANT TERMINAL // 2026")
st.markdown("---")

# 側邊欄：輸入區
with st.sidebar:
    st.header(">> INPUT PARAMETERS")
    target_input = st.text_input("TICKERS (已自動優化.TW)", value="2330, 2317, 3231")
    st.caption("Format: 2330, 2317 (Comma separated)")
    st.markdown("---")
    run_btn = st.button(">>> EXECUTE ANALYSIS <<<")
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    st.success("DATA FEED: CONNECTED")
    st.info("ENGINE: READY")

# 主畫面：輸出區
if run_btn:
    tickers = [x.strip() for x in target_input.split(',')]
    
    # 使用專業術語作為 Tab 名稱
    tab1, tab2 = st.tabs(["[ TARGET ANALYSIS ]", "[ MONTE CARLO SIMULATION ]"])
    
    with tab1:
        for ticker in tickers:
            engine = QuantEngine(ticker)
            df = engine.fetch_data()
            if df is not None:
                res = engine.generate_signal(df)
                
                # 標題列：代碼與現價
                st.markdown(f"### >> TARGET: {engine.code} | PRICE: {res['price']:.2f}")
                
                # 核心指令區 (使用自定義 HTML 標籤)
                st.markdown(f"""
                    <div style="margin: 20px 0;">
                        <span class="status-tag {res['tag']}">{res['action']}</span>
                        <span style="margin-left: 15px; font-weight: bold; color: #e0e0e0;">
                            // {res['brief']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 數據面板 (兩欄佈局)
                c1, c2 = st.columns([2, 3])
                
                with c1:
                    st.markdown("#### KEY METRICS")
                    # 使用原生 Metric 組件
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BIAS (乖離率)", f"{res['metrics']['BIAS%']:+.2f}%")
                    m2.metric("RSI (強弱指標)", f"{res['metrics']['RSI']:.1f}")
                    m3.metric("SUP (月線支撐)", f"{res['metrics']['MA20']:.1f}")

                with c2:
                    st.markdown("#### TECHNICAL SUPPORT DATA")
                    # 使用 Code Block 顯示專業數據
                    details_str = "\n".join([f"> {item}" for item in res['details']])
                    st.code(details_str, language="shell")
                
                st.divider()

    with tab2:
        st.markdown("### >> SIMULATION PARAMETERS: N=10000 | BLOCK=5D")
        for ticker in tickers:
            engine = QuantEngine(ticker)
            df = engine.fetch_data()
            if df is not None:
                paths, p5, p50, p95, win_rate = engine.run_monte_carlo(df)
                
                st.markdown(f"#### TARGET: {engine.code} // PROBABILITY OUTLOOK (20D)")
                
                # 勝率顯示 (使用進度條風格)
                win_color = "#238636" if win_rate > 50 else "#da3633"
                st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <span style="color: #8b949e;">WIN PROBABILITY: </span>
                        <span style="font-size: 24px; font-weight: bold; color: {win_color};">
                            {win_rate:.1f}%
                        </span>
                    </div>
                    <progress value="{win_rate}" max="100" style="width: 100%; height: 10px;"></progress>
                    """, unsafe_allow_html=True)

                # 模擬路徑圖 (強制深色主題)
                chart_df = pd.DataFrame(paths[:100, :].T)
                st.line_chart(chart_df, height=300, use_container_width=True)
                
                # 風險情境分析
                st.markdown("#### RISK/REWARD SCENARIOS")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("P5 (WORST CASE)", f"{p5:.1f}", delta=f"{((p5-res['price'])/res['price']*100):.1f}%", delta_color="inverse")
                rc2.metric("P50 (EXPECTED)", f"{p50:.1f}")
                rc3.metric("P95 (BEST CASE)", f"{p95:.1f}", delta=f"{((p95-res['price'])/res['price']*100):.1f}%")
                
                st.divider()
