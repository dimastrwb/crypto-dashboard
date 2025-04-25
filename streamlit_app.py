import streamlit as st
import requests
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(layout="wide")
st.title("📊 Crypto Price Dashboard + AI Forecast (5 Tahun)")

st.markdown("""
Selamat datang di dashboard prediksi harga crypto! 🚀  
Pilih crypto, lihat grafik harga, dan prediksi harga hingga 5 tahun ke depan dengan AI (Prophet)
""")

# === Sidebar ===
st.sidebar.header("🪙 Pilih Cryptocurrency")

@st.cache_data
def get_top5_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 5,
        'page': 1
    }
    r = requests.get(url, params=params)
    data = r.json()
    return [coin['id'] for coin in data], data

coin_list, top5_data = get_top5_coins()

coin1 = st.sidebar.selectbox("Pilih coin pertama", coin_list)
coin2 = st.sidebar.selectbox("Pilih coin kedua", ["(Tidak dibandingkan)"] + coin_list)
days = st.sidebar.slider("⏳ Berapa hari data historis?", 30, 180, 90)

# === Notifikasi perubahan harga ===
@st.cache_data

def load_price_data(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["ds"] = pd.to_datetime(prices["timestamp"], unit='ms')
    prices["y"] = prices["price"]
    return prices[["ds", "y"]]

df1 = load_price_data(coin1, days)
df2 = load_price_data(coin2, days) if coin2 != "(Tidak dibandingkan)" else None

st.subheader("📢 Notifikasi Perubahan Harga Harian")
if df1 is not None and len(df1) >= 2:
    change_pct = ((df1["y"].iloc[-1] - df1["y"].iloc[-2]) / df1["y"].iloc[-2]) * 100
    if change_pct > 0:
        color = "green"
        emoji = "📈"
    elif change_pct < 0:
        color = "red"
        emoji = "📉"
    else:
        color = "gray"
        emoji = "➖"
    st.markdown(f"""
    <div style='background-color:{color};padding:10px;border-radius:10px'>
        <span style='color:white'>{emoji} Perubahan harga {coin1} hari ini: {change_pct:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

# === Grafik Harga Historis ===
st.subheader(f"📉 Harga historis {coin1}")
if df1 is not None:
    st.line_chart(df1.set_index("ds")["y"])
else:
    st.error("Gagal ambil data dari CoinGecko.")

if coin2 != "(Tidak dibandingkan)":
    st.subheader(f"📊 Perbandingan harga historis {coin1} vs {coin2}")
    if df1 is not None and df2 is not None:
        df_merge = pd.merge(df1, df2, on="ds", suffixes=(f"_{coin1}", f"_{coin2}"))
        fig, ax = plt.subplots()
        ax.plot(df_merge["ds"], df_merge[f"y_{coin1}"], label=coin1)
        ax.plot(df_merge["ds"], df_merge[f"y_{coin2}"], label=coin2)
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (USD)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Gagal ambil data coin kedua dari CoinGecko.")

# === Prediksi harga dengan Prophet (berdasarkan historis) ===
st.subheader(f"🤖 Prediksi harga {coin1} (AI Forecast 5 Tahun)")
if df1 is not None:
    m = Prophet()
    m.fit(df1)
    future = m.make_future_dataframe(periods=1825)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    # Download ke Excel
    st.subheader("📥 Download prediksi ke Excel")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel(writer, index=False, sheet_name="Forecast")
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="forecast_{coin1}.xlsx">📁 Download File Excel</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.error("Tidak bisa melakukan prediksi karena data historis kosong.")

# === Prediksi harga tanpa historis ===
st.subheader(f"📈 Prediksi harga {coin1} selama 5 tahun ke depan (Tanpa Historis)")
if df1 is not None and not df1.empty:
    last_price = df1["y"].iloc[-1]
    future_df = pd.DataFrame({
        "ds": pd.date_range(start=pd.Timestamp.today(), periods=1825),
        "y": [last_price] * 1825
    })

    model = Prophet()
    model.fit(future_df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)

    fig_pred = model.plot(forecast)
    st.pyplot(fig_pred)
else:
    st.warning("Tidak bisa membuat prediksi karena data historis tidak tersedia.")

# === Tabel Top 5 Market Cap ===
st.subheader("🏆 Top 5 Cryptocurrency Berdasarkan Market Cap")

@st.cache_data

def get_top5_table():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 5,
        'page': 1
    }
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data)[["name", "symbol", "current_price", "market_cap", "price_change_percentage_24h"]]
    df.index = range(1, len(df) + 1)
    return df

st.dataframe(get_top5_table())
