import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress

# ==========================================
# 1. 介面設定
# ==========================================
st.set_page_config(page_title="2026 多因子量化決策系統", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    /* 專業金融終端機風格 */
    html, body, [class*="css"] {font-family: 'Microsoft JhengHei', sans-serif; color: #333;}
    
    .status-box {
        padding: 20px; border-radius: 8px; border: 1px solid #ddd;
        margin-bottom: 20px; background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .factor-tag {
        background: #e3f2fd; color: #1565c0; padding: 5px 10px; 
        border-radius: 4px; font-size: 14px; font-weight: bold; border: 1px solid #bbdefb;
        margin-right: 5px;
    }
    .regime-tag {
        font-size: 18px; font-weight: bold; padding: 5px 15px; border-radius: 20px;
    }
    .kelly-warning {
        background-color: #fff3e0; color: #e65100; padding: 15px; 
        border-left: 5px solid #ff9800; font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 量化運算核心 (Math Heavy)
# ==========================================
class QuantEngine:
    def __init__(self, ticker):
        self.ticker = f"{ticker}.TW" if not ticker.endswith('.TW') else ticker
        
    def fetch_data(self):
        try:
            # 抓取 5 年數據以計算長期回撤與 Hurst
            df = yf.download(self.ticker, period="5y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 250: return None
            return df
        except: return None

    # --- 數學模型區 ---

    def calculate_adx(self, df, window=14):
        """計算 ADX (趨勢強度指標)"""
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(df['High'] - df['Low'])
        tr2 = pd.DataFrame(abs(df['High'] - df['Close'].shift(1)))
        tr3 = pd.DataFrame(abs(df['Low'] - df['Close'].shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(window).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/window).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window).mean().iloc[-1]
        return adx

    def calculate_obv_slope(self, df, window=20):
        """計算 OBV 斜率 (判斷資金是否進場)"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        # 計算最近 20 天 OBV 的線性回歸斜率
        y = obv.iloc[-window:].values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        return slope

    def calculate_hurst(self, ts):
        """赫斯特指數 (0.5=隨機, >0.5=趨勢, <0.5=均值回歸)"""
        lags = range(2, 20)
        # 防止 log(0) 錯誤
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def calculate_max_drawdown(self, df):
        """計算歷史最大回撤 (用於凱利公式分母)"""
        roll_max = df['Close'].cummax()
        daily_drawdown = df['Close'] / roll_max - 1.0
        max_dd = daily_drawdown.min()
        return abs(max_dd) # 回傳正數，例如 0.45 代表跌 45%

    # --- 綜合決策區 ---

    def analyze(self, df):
        close = df['Close']
        price = close.iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        # 1. 環境濾網 (Regime Filter)
        adx = self.calculate_adx(df)
        hurst = self.calculate_hurst(np.log(close.values))
        
        # 判定市場狀態
        market_regime = "噪音盤整 (Noise)"
        if adx > 25 and hurst > 0.55:
            market_regime = "強趨勢 (Trending)"
        elif adx < 20 and hurst < 0.45:
            market_regime = "均值回歸 (Mean Reversion)"
            
        # 2. 動能因子 (Momentum)
        obv_slope = self.calculate_obv_slope(df)
        price_slope = (price - close.iloc[-20]) / close.iloc[-20]
        
        # 判斷價量背離
        divergence = False
        if price_slope > 0 and obv_slope < 0: divergence = True # 價漲量縮 (危險)
        
        # 3. 風險因子 (Risk)
        max_dd = self.calculate_max_drawdown(df)
        
        return {
            "price": price,
            "ma20": ma20,
            "adx": adx,
            "hurst": hurst,
            "regime": market_regime,
            "obv_slope": obv_slope,
            "divergence": divergence,
            "max_dd": max_dd
        }

    def kelly_sizing(self, win_rate, reward_risk, max_dd):
        """
        優化版凱利公式：
        1. 使用 1/4 Kelly (Fractional)
        2. 風險分母使用 Max Drawdown (更保守)
        """
        w = win_rate / 100
        r = reward_risk
        
        # 原始凱利
        raw_kelly = w - (1 - w) / r
        
        # 調整 1: 槓桿懲罰 (若歷史回撤很大，凱利值要縮小)
        # 如果這檔股票歷史曾經腰斬 (MDD=0.5)，你的部位不該超過 1/MDD 的一小部分
        risk_adj_factor = 1.0
        if max_dd > 0.3: risk_adj_factor = 0.5
        if max_dd > 0.5: risk_adj_factor = 0.25
        
        # 調整 2: 使用 1/4 Kelly
        final_size = max(0, raw_kelly * 0.25 * risk_adj_factor) * 100
        
        return final_size

    def run_simulation(self, df, simulations=5000):
        # 區塊抽樣
        returns = df['Close'].pct_change().dropna().values
        last_price = df['Close'].iloc[-1]
        days = 20
        sim_paths = np.zeros((simulations, days))
        block_size = 10
        
        for i in range(simulations):
            path = []
            for _ in range(days // block_size):
                start = np.random.randint(0, len(returns) - block_size)
                path.extend(returns[start : start+block_size])
            sim_paths[i] = last_price * np.cumprod(1 + np.array(path))
            
        final = sim_paths[:, -1]
        win_rate = np.sum(final > last_price) / simulations * 100
        exp_ret = (np.median(final) - last_price) / last_price
        # 模擬 VaR (僅供參考，不入凱利公式)
        var = abs((np.percentile(final, 5) - last_price) / last_price)
        
        return win_rate, exp_ret, var

# ==========================================
# 3. 前端介面
# ==========================================
st.title("🧠 2026 多因子量化決策系統")
st.markdown("**App 8.0: Regime Filter + Volume Confirmation + Fractional Kelly**")

with st.sidebar:
    tickers_input = st.text_input("輸入代碼", "2330, 2317, 2603")
    run = st.button("執行量化運算")

if run:
    tickers = [x.strip() for x in tickers_input.split(',')]
    
    for t in tickers:
        eng = QuantEngine(t)
        df = eng.fetch_data()
        
        if df is not None:
            # 1. 執行運算
            metrics = eng.analyze(df)
            win_rate, exp_ret, var = eng.run_simulation(df)
            
            # 2. 計算凱利部位
            # 賠率 = 預期獲利 / 模擬虧損 (這裡還是得用模擬的下檔，但凱利本身會被歷史MDD修正)
            odds = exp_ret / var if var > 0 else 1
            kelly = eng.kelly_sizing(win_rate, odds, metrics['max_dd'])
            
            # --- 決策邏輯 (Regime Filter) ---
            decision = "🚫 NO TRADE (觀望)"
            color = "#757575" # 灰
            reason = "市場雜訊過多，或多空訊號衝突。"
            
            # 濾網 1: 市場狀態
            if metrics['regime'] == "強趨勢 (Trending)":
                # 濾網 2: 價量結構
                if metrics['price'] > metrics['ma20']:
                    if not metrics['divergence']:
                        # 濾網 3: OBV 動能
                        if metrics['obv_slope'] > 0:
                            decision = "✅ LONG (做多)"
                            color = "#2e7d32" # 綠
                            reason = "趨勢形成 (ADX>25) + 價漲量增 (OBV向上) + 均線多頭。"
                        else:
                            reason = "雖有趨勢，但資金動能不足 (OBV 疲軟)，建議觀察。"
                    else:
                        decision = "⚠️ WARNING (背離)"
                        color = "#f9a825" # 黃
                        reason = "價格創新高但量能跟不上 (價量背離)，小心假突破。"
                else:
                    if metrics['obv_slope'] < 0:
                        decision = "🔻 SHORT (做空/避險)"
                        color = "#c62828" # 紅
                        reason = "趨勢向下 + 資金撤離。"
            else:
                reason = "目前為盤整/噪音盤 (ADX低, Hurst<0.5)，趨勢策略失效，不動作。"

            # --- 顯示卡片 ---
            st.markdown(f"""
            <div class="status-box" style="border-left: 8px solid {color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3>{t} ｜ {decision}</h3>
                    <div class="regime-tag" style="background:{'#e8f5e9' if '強趨勢' in metrics['regime'] else '#eee'}; color:{'#2e7d32' if '強趨勢' in metrics['regime'] else '#666'};">
                        {metrics['regime']}
                    </div>
                </div>
                
                <div style="margin: 15px 0;">
                    <span class="factor-tag">ADX: {metrics['adx']:.1f}</span>
                    <span class="factor-tag">Hurst: {metrics['hurst']:.2f}</span>
                    <span class="factor-tag">OBV斜率: {metrics['obv_slope']:.2f}</span>
                    <span class="factor-tag">MaxDD: -{metrics['max_dd']*100:.1f}%</span>
                </div>
                
                <p><strong>👨‍💻 量化解讀：</strong> {reason}</p>
                
                <div class="kelly-warning">
                    <strong>💰 1/4 Kelly 部位建議： {kelly:.1f}%</strong><br>
                    <small>計算基礎：勝率 {win_rate:.1f}% ｜ 歷史最大回撤 -{metrics['max_dd']*100:.1f}% (風險懲罰因子)</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Debug 區 (給你看數值用的)
            with st.expander("查看詳細因子數據"):
                st.json(metrics)
