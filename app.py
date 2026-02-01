import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, linregress

# ==========================================
# 1. 機構級配置 (Dark Mode / High Density)
# ==========================================
st.set_page_config(
    page_title="Institutional Quant Terminal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* 專業暗色系風格 - Bloomberg Terminal 風格 */
    .stApp {background-color: #0e1117;}
    h1, h2, h3, h4, p, div, span {color: #e6e6e6 !important; font-family: 'Roboto Mono', monospace;}
    
    /* 數據表格優化 */
    .quant-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .metric-label {font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value {font-size: 28px; font-weight: bold; color: #60a5fa;}
    .metric-sub {font-size: 14px; color: #d1d5db;}
    
    /* 回測曲線圖背景 */
    .stLineChart {background-color: #1f2937; padding: 10px; border-radius: 4px;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 量化運算引擎 (Quant Engine)
# ==========================================
class QuantEngine:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
        
    def fetch_data(self, period="5y"): # 分析師至少看 3-5 年
        try:
            df = yf.download(self.ticker, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df if len(df) > 250 else None
        except: return None

    def calculate_hurst(self, ts):
        """赫斯特指數：判斷是均值回歸 (<0.5) 還是趨勢延續 (>0.5)"""
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def calculate_metrics(self, df):
        returns = df['Close'].pct_change().dropna()
        log_returns = np.log(1 + returns)
        
        # 統計因子
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_return - 0.02) / ann_vol # 假設無風險利率 2%
        skew_val = skew(returns)
        kurt_val = kurtosis(returns)
        hurst = self.calculate_hurst(np.log(df['Close'].values))
        
        # 凱利公式 (Kelly Criterion)
        # 簡化版：K = p - (1-p)/b (假設 b=1, 即賠率1:1時, K = 2p-1)
        # 這裡使用歷史勝率推算建議倉位
        win_days = returns[returns > 0].count()
        total_days = returns.count()
        win_rate = win_days / total_days
        kelly_pct = (2 * win_rate - 1) * 100 
        
        return {
            "Annual_Ret": ann_return * 100,
            "Volatility": ann_vol * 100,
            "Sharpe": sharpe,
            "Skewness": skew_val,
            "Hurst": hurst,
            "Kelly": max(0, kelly_pct) # 負值代表不該下注
        }

    def run_backtest(self, df):
        """向量化回測引擎 (Vectorized Backtester)"""
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        
        # 策略邏輯：雙均線交叉 (Golden Cross)
        data['SMA20'] = data['Close'].rolling(20).mean()
        data['SMA60'] = data['Close'].rolling(60).mean()
        
        # 訊號生成 (1=持倉, 0=空手)
        data['Signal'] = np.where(data['SMA20'] > data['SMA60'], 1, 0)
        
        # 計算策略報酬 (Shift 1 代表訊號出現後隔天進場)
        data['Strategy_Ret'] = data['Signal'].shift(1) * data['Returns']
        
        # 計算權益曲線 (Equity Curve)
        data['BuyHold_Cum'] = (1 + data['Returns']).cumprod()
        data['Strategy_Cum'] = (1 + data['Strategy_Ret']).cumprod()
        
        # 回測績效指標
        total_ret = (data['Strategy_Cum'].iloc[-1] - 1) * 100
        bh_ret = (data['BuyHold_Cum'].iloc[-1] - 1) * 100
        
        # 最大回撤 (Max Drawdown)
        cum_roll_max = data['Strategy_Cum'].cummax()
        drawdown = (data['Strategy_Cum'] - cum_roll_max) / cum_roll_max
        max_dd = drawdown.min() * 100
        
        return data, total_ret, bh_ret, max_dd

# ==========================================
# 3. 介面層 (Analyst Dashboard)
# ==========================================
st.title("🧬 AlphaSeeker: Institutional Quant Terminal")
st.markdown("_Engineered for High-Frequency Decision Making_")
st.markdown("---")

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker = st.text_input("Tickers (Comma separated)", "2330, 2317, 2454, 3231", label_visibility="collapsed")
with col_btn:
    run = st.button("EXECUTE ANALYSIS", use_container_width=True)

if run:
    tickers = [x.strip() for x in ticker.split(',')]
    
    for t in tickers:
        engine = QuantEngine(t)
        df = engine.fetch_data()
        
        if df is not None:
            metrics = engine.calculate_metrics(df)
            bt_data, strat_ret, bh_ret, max_dd = engine.run_backtest(df)
            
            # --- Header: 標的與核心績效 ---
            st.markdown(f"### 📌 {t} - Quantitative Profile")
            
            # 第一排：因子分析 (Factor Analysis)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="quant-card">
                    <div class="metric-label">Sharpe Ratio (Risk-Adj)</div>
                    <div class="metric-value" style="color: {'#10b981' if metrics['Sharpe']>1 else '#ef4444'};">{metrics['Sharpe']:.2f}</div>
                    <div class="metric-sub">Target > 1.0</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="quant-card">
                    <div class="metric-label">Hurst Exponent</div>
                    <div class="metric-value" style="color: {'#f59e0b' if 0.45 < metrics['Hurst'] < 0.55 else '#10b981'};">{metrics['Hurst']:.2f}</div>
                    <div class="metric-sub">{'>0.5 Trend / <0.5 Mean Rev'}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="quant-card">
                    <div class="metric-label">Kelly Criterion</div>
                    <div class="metric-value">{metrics['Kelly']:.1f}%</div>
                    <div class="metric-sub">Optimal Allocation</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="quant-card">
                    <div class="metric-label">Skewness (Tail Risk)</div>
                    <div class="metric-value" style="color: {'#ef4444' if metrics['Skewness'] < -0.5 else '#10b981'};">{metrics['Skewness']:.2f}</div>
                    <div class="metric-sub">Neg = Black Swan Risk</div>
                </div>""", unsafe_allow_html=True)

            # 第二排：回測驗證 (Backtest Verification)
            c_chart, c_stats = st.columns([3, 1])
            
            with c_chart:
                st.markdown("**Equity Curve (Strategy vs Buy & Hold)**")
                chart_df = bt_data[['BuyHold_Cum', 'Strategy_Cum']]
                st.line_chart(chart_df, color=["#6b7280", "#3b82f6"]) # 灰=大盤, 藍=策略
                
            with c_stats:
                st.markdown("""<div class="quant-card" style="height: 300px;">
                    <div class="metric-label">Strategy Total Return</div>
                    <div class="metric-value" style="font-size: 22px;">{:.1f}%</div>
                    <br>
                    <div class="metric-label">Benchmark Return</div>
                    <div class="metric-value" style="font-size: 22px; color: #9ca3af;">{:.1f}%</div>
                    <br>
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value" style="font-size: 22px; color: #ef4444;">{:.1f}%</div>
                    <br>
                    <div class="metric-label">Alpha (Excess Ret)</div>
                    <div class="metric-value" style="font-size: 22px; color: #10b981;">{:.1f}%</div>
                </div>""".format(strat_ret, bh_ret, max_dd, strat_ret - bh_ret), unsafe_allow_html=True)

            st.markdown("---")
