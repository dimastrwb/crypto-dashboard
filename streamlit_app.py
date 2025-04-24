import streamlit as st
import requests
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Crypto Price Dashboard + 5-Year Forecast")

st.markdown("""
Ini adalah dashboard prediksi harga crypto. Kamu bisa:
- Pilih koin crypto
- Lihat grafik harga historis
- Prediksi harga 5 tahun ke depan (dengan Prophet)
""")

coin = st.sidebar.selectbox("🔽 Pilih Cryptocurrency", ["bitcoin", "ethereum", "dogecoin", "cardano", "solana"])
days = st.sidebar.selectbox("⏳ Pilih jumlah hari historis", [1, 7, 14, 30, 90, 180, 365, 'max'])

url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency=usd&days={days}"
r = requests.get(url)
if r.status_code != 200:
    st.error("❌ Gagal ambil data dari CoinGecko.")
    st.stop()

data = r.json()
if not data:
    st.warning("⚠️ Data kosong untuk kombinasi coin dan hari ini.")
    st.stop()

df = pd.DataFrame(data, columns=["Timestamp", "Open", "High", "Low", "Close"])
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

st.subheader(f"📈 Harga Historis: {coin.upper()}")
st.line_chart(df.set_index("Timestamp")[["Open", "Close"]])

st.subheader("🔮 Prediksi Harga 5 Tahun ke Depan")
df_prophet = df[["Timestamp", "Close"]].rename(columns={"Timestamp": "ds", "Close": "y"})
model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=1825)
forecast = model.predict(future)

fig1 = model.plot(forecast)
st.pyplot(fig1)

with st.expander("📊 Komponen Tren & Musiman"):
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
