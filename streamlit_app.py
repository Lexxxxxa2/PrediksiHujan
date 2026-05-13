# =========================================================
# APP FINAL HYBRID V5.1
# PREDIKSI CURAH HUJAN SURABAYA
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Prediksi Curah Hujan",
    page_icon="🌧️",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
}

.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION
# =========================================================

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "page" not in st.session_state:
    st.session_state.page = "hasil"

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌧️ MENU")

def nav_button(label, key):

    if st.sidebar.button(
        label,
        use_container_width=True
    ):
        st.session_state.page = key

nav_button("📋 Hasil Prediksi", "hasil")
nav_button("📈 Grafik", "grafik")
nav_button("🔥 Heatmap", "heatmap")
nav_button("📊 Evaluasi Model", "evaluasi")

menu = st.session_state.page

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("model_hybrid_v5_1.pkl")
scaler = joblib.load("scaler_hybrid_v5_1.pkl")
pca = joblib.load("pca_hybrid_v5_1.pkl")
features = joblib.load("features_hybrid_v5_1.pkl")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_excel(
    "datahujan2021-2025.xlsx"
)

df.columns = df.columns.str.lower()

# =========================================================
# RENAME KOLOM
# =========================================================

rename_map = {

    'temp_avg_c': 'TAVG',
    'temp_max_c': 'TX',
    'temp_min_c': 'TN',

    'rel_humidity_avg_pc': 'RH_AVG',

    'sunshine_24h_h': 'SS',

    'wind_speed_max_ms': 'FF_X',
    'wind_speed_avg_ms': 'FF_AVG',

    'rainfall_mm': 'RR'
}

df.rename(
    columns=rename_map,
    inplace=True
)

# =========================================================
# PASTIKAN FEATURE ADA
# =========================================================

required_cols = [
    'TAVG',
    'TX',
    'TN',
    'RH_AVG',
    'SS',
    'FF_X',
    'FF_AVG',
    'RR'
]

for col in required_cols:

    if col not in df.columns:

        df[col] = 0

# =========================================================
# TANGGAL
# =========================================================

df['tanggal'] = pd.to_datetime(
    df['data_timestamp']
)

# =========================================================
# GROUP HARIAN
# =========================================================

df = df.groupby(
    'tanggal'
).mean(
    numeric_only=True
).reset_index()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_features(df):

    df = df.copy()

    # rolling
    df['TAVG_3D'] = df['TAVG'].rolling(3).mean()
    df['RR_3D'] = df['RR'].rolling(3).sum()
    df['RH_3D'] = df['RH_AVG'].rolling(3).mean()

    # calendar
    df['DAYOFYEAR'] = df['tanggal'].dt.dayofyear
    df['MONTH'] = df['tanggal'].dt.month

    # lag
    df['RR_LAG_1'] = df['RR'].shift(1)
    df['RR_LAG_7'] = df['RR'].shift(7)
    df['RR_LAG_14'] = df['RR'].shift(14)

    # rolling stats
    df['RR_MEAN_7'] = df['RR'].rolling(7).mean()
    df['RR_STD_7'] = df['RR'].rolling(7).std()

    df['RR_MEAN_30'] = df['RR'].rolling(30).mean()
    df['RR_STD_30'] = df['RR'].rolling(30).std()

    # diff
    df['TAVG_DIFF'] = df['TAVG'].diff()
    df['RH_DIFF'] = df['RH_AVG'].diff()
    df['RR_DIFF'] = df['RR'].diff()

    return df

# =========================================================
# APPLY FEATURE ENGINEERING
# =========================================================

df = add_features(df)

df = df.ffill()
df = df.fillna(0)

# =========================================================
# METRIC MODEL
# =========================================================

BEST_MAE = 5.18
BEST_RMSE = 11.37
BEST_ACC = 69.44
BEST_PRECISION = 0.74
BEST_RECALL = 0.69
BEST_F1 = 0.70

# =========================================================
# TITLE
# =========================================================

st.title("🌧️ Prediksi Curah Hujan Kota Surabaya")

st.markdown("""
Dashboard prediksi curah hujan berbasis Machine Learning  
menggunakan algoritma K-Nearest Neighbor (K-NN)
""")

# =========================================================
# INPUT
# =========================================================

col1, col2, col3 = st.columns([1,1,3])

with col1:

    n_days = st.number_input(
        "Jumlah Hari Prediksi",
        1,
        365,
        30
    )

with col2:

    run = st.button(
        "🚀 Jalankan Prediksi",
        use_container_width=True
    )

# =========================================================
# KATEGORI HUJAN
# =========================================================

def kategori_hujan(rr):

    if rr < 0.5:
        return "Tidak Hujan"

    elif rr <= 20:
        return "Hujan Ringan"

    elif rr <= 50:
        return "Hujan Sedang"

    else:
        return "Hujan Lebat"

# =========================================================
# PREDIKSI FORECASTING
# =========================================================

if run:

    forecast_df = df.copy()

    results = []

    for i in range(n_days):

        # =====================================
        # FEATURE ENGINEERING
        # =====================================

        temp_df = add_features(
            forecast_df.copy()
        )

        temp_df = temp_df.ffill()
        temp_df = temp_df.fillna(0)

        # =====================================
        # AMBIL DATA TERAKHIR
        # =====================================

        latest = temp_df.iloc[-1]

        # =====================================
        # INPUT FEATURE
        # =====================================

        X_input = pd.DataFrame(
            [latest[features].values],
            columns=features
        )

        # =====================================
        # SCALING
        # =====================================

        X_scaled = scaler.transform(
            X_input
        )

        # =====================================
        # PCA
        # =====================================

        X_pca = pca.transform(
            X_scaled
        )

        # =====================================
        # PREDIKSI
        # =====================================

        pred_rr = model.predict(
            X_pca
        )[0]

        pred_rr = max(0, pred_rr)

        # =====================================
        # TANGGAL BARU
        # =====================================

        next_date = (
            forecast_df['tanggal'].iloc[-1]
            + pd.Timedelta(days=1)
        )

        # =====================================
        # SIMPAN HASIL
        # =====================================

        results.append({

            "Tanggal": next_date,

            "Curah Hujan Prediksi":
                round(pred_rr, 2),

            "Kategori":
                kategori_hujan(pred_rr)

        })

        # =====================================
        # ROW BARU
        # =====================================

        new_row = forecast_df.iloc[-1].copy()

        new_row['tanggal'] = next_date

        # hasil prediksi dipakai lagi
        new_row['RR'] = pred_rr

        # =====================================
        # TAMBAHKAN KE DATASET
        # =====================================

        forecast_df = pd.concat(

            [
                forecast_df,
                pd.DataFrame([new_row])
            ],

            ignore_index=True
        )

    # =====================================
    # SIMPAN HASIL
    # =====================================

    st.session_state.result_df = pd.DataFrame(
        results
    )

# =========================================================
# HASIL PREDIKSI
# =========================================================

if (
    st.session_state.result_df is not None
    and menu == "hasil"
):

    result_df = st.session_state.result_df.copy()

    st.subheader("📋 Model Hasil Prediksi")

    # metric
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", BEST_MAE)
    c2.metric("RMSE", BEST_RMSE)
    c3.metric("Accuracy", f"{BEST_ACC}%")
    c4.metric("F1 Score", BEST_F1)

    st.markdown("---")

    # warna kategori
    def style_kategori(val):

        if val == "Tidak Hujan":
            return "background-color:#2c2c2c;color:white"

        elif val == "Hujan Ringan":
            return "background-color:#2ecc71;color:black"

        elif val == "Hujan Sedang":
            return "background-color:#f39c12;color:black"

        else:
            return "background-color:#e74c3c;color:white"

    st.dataframe(
        result_df.style
        .format({
            "Curah Hujan Prediksi": "{:.2f}"
        })
        .applymap(
            style_kategori,
            subset=["Kategori"]
        ),
        use_container_width=True
    )

    # download
    csv = result_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download CSV",
        csv,
        "hasil_prediksi.csv",
        use_container_width=True
    )

# =========================================================
# GRAFIK
# =========================================================

elif (
    st.session_state.result_df is not None
    and menu == "grafik"
):

    st.subheader(
        "📈 Grafik Prediksi Curah Hujan"
    )

    gdf = st.session_state.result_df.copy()

    fig = px.line(
        gdf,
        x="Tanggal",
        y="Curah Hujan Prediksi",
        markers=True
    )

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# HEATMAP
# =========================================================

elif (
    st.session_state.result_df is not None
    and menu == "heatmap"
):

    st.subheader(
        "🔥 Heatmap Curah Hujan"
    )

    hdf = st.session_state.result_df.copy()

    hdf['month'] = hdf['Tanggal'].dt.month
    hdf['day'] = hdf['Tanggal'].dt.day

    bulan = st.selectbox(
        "Pilih Bulan",
        sorted(hdf['month'].unique())
    )

    mdf = hdf[
        hdf['month'] == bulan
    ]

    pivot = mdf.pivot(
        index='month',
        columns='day',
        values='Curah Hujan Prediksi'
    )

    fig, ax = plt.subplots(
        figsize=(16,4)
    )

    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# EVALUASI MODEL
# =========================================================

elif (
    st.session_state.result_df is not None
    and menu == "evaluasi"
):

    st.subheader("📊 Evaluasi Model")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", BEST_MAE)
    c2.metric("RMSE", BEST_RMSE)
    c3.metric("Accuracy", f"{BEST_ACC}%")

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Precision",
        BEST_PRECISION
    )

    c5.metric(
        "Recall",
        BEST_RECALL
    )

    c6.metric(
        "F1 Score",
        BEST_F1
    )

    st.markdown("---")

    st.subheader("📌 Informasi Model")

    info_df = pd.DataFrame({

        "Parameter": [
            "Model",
            "Metric",
            "Best K",
            "Ratio",
            "PCA Component"
        ],

        "Value": [
            "K-NN Hybrid V5.1",
            "Manhattan",
            "3",
            "90:10",
            "10"
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

# =========================================================
# EMPTY
# =========================================================

elif st.session_state.result_df is None:

    st.info(
        "Silakan jalankan prediksi terlebih dahulu 🚀"
    )