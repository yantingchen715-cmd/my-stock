import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(page_title="2026 股市操盤指揮所", page_icon="💰", layout="wide")

# CSS 美化 (大字體、按鈕優化)
st.markdown("""
    <style>
    .stMetric {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d0d0d0;}
    .stButton>button {width: 100%; height: 60px; font-size: 24px !important; background-color: #d32f2f; color: white; border-radius: 10px;}
    .big-font {font-size: 24px !important; font-weight: bold;}
    .highlight {background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107;}
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
            # 抓取過去 5 年數據 (為了讓模擬更準)
            df = yf.download(self.ticker, period="5y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) < 200: return None
            return df
        except: return None

    def analyze_signal(self, df):
        # 1. 準備數據
        close = df['Close']
        price = close.iloc[-1]
        
        # 均線系統
        ma20 = close.rolling(20).mean().iloc[-1]   # 月線 (短線生命線)
        ma60 = close.rolling(60).mean().iloc[-1]   # 季線 (大趨勢)
        ma5_curr = close.rolling(5).mean().iloc[-1] # 5日線 (攻擊訊號)
        ma5_prev = close.rolling(5).mean().iloc[-2]
        
        # 乖離率 (判斷貴不貴)
        bias = ((price - ma20) / ma20) * 100
        
        # 2. 判斷指令 (買/賣/觀望)
        signal = "觀望 (空手)"
        action_color = "gray"
        reason = "目前趨勢不明顯，錢留著比較安全。"
        
        # --- 賣出邏輯 (優先判斷，保命要緊) ---
        # 條件：跌破月線 且 月線下彎 OR 乖離過大
        if price < ma20:
            signal = "🚨 快逃 (賣出訊號)"
            action_color = "red"
            reason = "股價已經跌破月線支撐，主力在出貨了，不要留戀！"
        elif bias > 20:
            signal = "💰 獲利了結 (太貴了)"
            action_color = "orange"
            reason = f"短線漲太多了 (乖離率 {bias:.1f}%)，隨時會崩盤，建議先賣一半放口袋。"
            
        # --- 買進邏輯 ---
        # 條件：多頭排列 (價>月>季) AND 5日線勾頭向上 AND 乖離不大
        elif price > ma20 and ma20 > ma60:
            if bias < 8: 
                # 回測支撐，且5日線向上
                if ma5_curr > ma5_prev:
                    signal = "✅ 進場買進 (黃金買點)"
                    action_color = "green"
                    reason = "趨勢向上，且股價剛整理完準備發動，現在買很安全。"
                else:
                    signal = "🔵 續抱 (安心持有)"
                    action_color = "blue"
                    reason = "趨勢還是多頭，但短線在休息，不用急著動作，繼續抱著就好。"
            else:
                signal = "⚠️ 續抱但勿追高"
                action_color = "orange"
                reason = "雖然是多頭，但現在買有點貴，手上有票的續抱，沒票的別追。"

        return {
            "現價": price,
            "指令": signal,
            "顏色": action_color,
            "理由": reason,
            "乖離": bias
        }

    def run_historical_bootstrap(self, df, simulations=10000, days=20):
        # 真實歷史重組模擬 (Block Bootstrap)
        returns = df['Close'].pct_change().dropna().values
        last_price = df['Close'].iloc[-1]
        
        # 建立矩陣
        sim_paths = np.zeros((simulations, days))
        
        # 快速區塊抽樣 (為了速度，我們簡化為隨機抽取 5 天區塊)
        block_size = 5
        num_blocks = days // block_size
        
        for i in range(simulations):
            path_returns = []
            for _ in range(num_blocks):
                # 隨機選一個起始點
                start_idx = np.random.randint(0, len(returns) - block_size)
                # 剪下這一段真實歷史
                path_returns.extend(returns[start_idx : start_idx + block_size])
            
            # 計算價格路徑
            sim_paths[i] = last_price * np.cumprod(1 + np.array(path_returns))
            
        end_prices = sim_paths[:, -1]
        
        # 統計勝率
        win_count = np.sum(end_prices > last_price)
        win_rate = (win_count / simulations) * 100
        
        p5 = np.percentile(end_prices, 5)
        p50 = np.percentile(end_prices, 50)
        p95 = np.percentile(end_prices, 95)
        
        return sim_paths, p5, p50, p95, win_rate

# ==========================================
# 前端介面
# ==========================================
st.title("💰 2026 股市操盤指揮所")
st.caption("AI 幫您算命：什麼時候買？什麼時候賣？一次看清楚")

with st.sidebar:
    st.header("👇 1. 輸入股票代碼")
    user_input = st.text_input("代碼 (例如 2330, 2317)", value="2330, 2317")
    st.write("---")
    st.header("👇 2. 按下紅色按鈕")
    run_btn = st.button("🚀 開始分析")
    st.write("---")
    st.info("💡 說明：\n\n- **進場**：趨勢剛開始，最安全。\n- **獲利了結**：漲太多了，落袋為安。\n- **快逃**：趨勢壞了，不要賠大錢。")

if run_btn:
    tickers = [x.strip() for x in user_input.split(',')]
    
    # 建立分頁
    tab1, tab2 = st.tabs(["📢 買賣指令 (現在做什麼?)", "🎲 一萬次模擬 (勝算多少?)"])
    
    with tab1:
        st.subheader("🤖 AI 操盤指令")
        for ticker in tickers:
            brain = StockBrain(ticker)
            df = brain.fetch_data()
            if df is not None:
                res = brain.analyze_signal(df)
                
                # 使用外框框起來，比較清楚
                with st.container():
                    st.markdown(f"### 【{ticker}】 現價: {res['現價']:.1f} 元")
                    
                    # 顯示超大指令
                    if res['顏色'] == 'green':
                        st.success(f"### {res['指令']}")
                    elif res['顏色'] == 'red':
                        st.error(f"### {res['指令']}")
                    elif res['顏色'] == 'orange':
                        st.warning(f"### {res['指令']}")
                    elif res['顏色'] == 'blue':
                        st.info(f"### {res['指令']}")
                    else:
                        st.write(f"### {res['指令']}")
                    
                    # 顯示白話文理由
                    st.markdown(f"<div class='highlight'><b>💬 AI 為什麼這樣說？</b><br>{res['理由']}</div>", unsafe_allow_html=True)
                    st.divider()

    with tab2:
        st.subheader("🎲 蒙地卡羅：如果歷史重演一萬次...")
        st.write("我們把這檔股票過去 5 年的走勢剪碎，重新拼湊 **10,000 次**，看看一個月後賺錢的機率有多少？")
        
        for ticker in tickers:
            brain = StockBrain(ticker)
            df = brain.fetch_data()
            if df is not None:
                sim_paths, p5, p50, p95, win_rate = brain.run_historical_bootstrap(df)
                
                st.markdown(f"### 📍 {ticker} 模擬結果")
                
                # 顯示勝率 (這是說服長輩最有力的證據)
                col_win, col_risk = st.columns(2)
                col_win.metric("勝率 (賺錢機率)", f"{win_rate:.1f}%", help="模擬一萬次中，有多少次是賺錢的")
                
                # 判斷勝率顏色
                if win_rate > 60:
                    col_win.success("✨ 勝率很高！這檔股票歷史股性很好，容易漲。")
                elif win_rate < 40:
                    col_win.error("💀 勝率很低！這檔股票很容易讓人賠錢，小心。")
                else:
                    col_win.warning("😐 勝率普通，大概一半一半。")

                # 畫出模擬圖 (只畫 100 條代表，不然網頁會卡死)
                chart_data = pd.DataFrame(sim_paths[:100, :].T)
                st.line_chart(chart_data, height=250)
                
                # 價格預測
                c1, c2, c3 = st.columns(3)
                c1.metric("運氣最差跌到", f"{p5:.1f}", delta=f"{((p5-df['Close'].iloc[-1])/df['Close'].iloc[-1]*100):.1f}%")
                c2.metric("平均會漲到", f"{p50:.1f}")
                c3.metric("運氣好漲到", f"{p95:.1f}", delta=f"{((p95-df['Close'].iloc[-1])/df['Close'].iloc[-1]*100):.1f}%")
                
                st.divider()
