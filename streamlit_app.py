# =========================================================
# MODERN RAINFALL FORECAST DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Rainfall Forecast Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SPLASH SCREEN
# =========================================================

import streamlit.components.v1 as components

if "loading_done" not in st.session_state:

    components.html(
        """
        <html>
        <head>
        <style>

        body {
            margin: 0;
            background-color: #020817;
            overflow: hidden;
        }

        .splash-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial;
        }

        .logo {
            width: 220px;
            animation: float 2s ease-in-out infinite;
        }

        .title {
            margin-top: 25px;
            font-size: 42px;
            font-weight: bold;
            color: white;
        }

        .subtitle {
            margin-top: 10px;
            font-size: 18px;
            color: #94a3b8;
        }

        @keyframes float {
            0% {
                transform: translateY(0px);
            }

            50% {
                transform: translateY(-10px);
            }

            100% {
                transform: translateY(0px);
            }
        }

        </style>
        </head>

        <body>

        <div class="splash-container">

            <img
                class="logo"
                src="https://raw.githubusercontent.com/Lexxxxxa2/prediksihujan/main/logo_universitas.png"
            >

            <div class="title">
                Universitas Wijaya Kusuma Surabaya
            </div>

            <div class="subtitle">
                Rainfall Forecast Dashboard
            </div>

        </div>

        </body>
        </html>
        """,
        height=700
    )

    time.sleep(3)

    st.session_state.loading_done = True

    st.rerun()

# =========================================================
# MODERN CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent;
}

.main {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #111827 100%
    );
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ===================================================== */
/* SIDEBAR FIX */
/* ===================================================== */

section[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* sidebar terbuka */
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 260px !important;
    max-width: 260px !important;
}

/* sidebar collapse */
section[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 0px !important;
    max-width: 0px !important;
}

/* ===================================================== */
/* BUTTON */
/* ===================================================== */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 48px;
    border: none;
    font-weight: 600;
    font-size: 15px;
    background: linear-gradient(90deg,#2563eb,#3b82f6);
    color: white;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(37,99,235,0.4);
}

/* ===================================================== */
/* METRIC CARD */
/* ===================================================== */

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 18px;
}

/* ===================================================== */
/* TITLE */
/* ===================================================== */

.big-title {
    font-size: 42px;
    font-weight: 800;
    color: white;

    /* FIX TEXT KEPOTONG */
    line-height: 1.3;
    padding-top: 15px;
    padding-bottom: 8px;

    margin-bottom: 0;
    overflow: visible;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;

    /* ubah ini */
    margin-top: 0px;
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

st.sidebar.markdown("##Rainfall AI")
st.sidebar.markdown("---")

def nav_button(label, key):

    if st.sidebar.button(
        label,
        use_container_width=True
    ):
        st.session_state.page = key

nav_button("Hasil Prediksi", "hasil")
nav_button("Grafik", "grafik")
nav_button("Heatmap", "heatmap")
nav_button("Evaluasi Model", "evaluasi")
nav_button("Flowchart Penelitian", "flowchart")

st.sidebar.markdown("---")

st.sidebar.info(
    "Dashboard prediksi curah hujan berbasis Machine Learning"
)

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

df.rename(columns=rename_map, inplace=True)

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
# DATE
# =========================================================

df['tanggal'] = pd.to_datetime(
    df['data_timestamp']
)

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

    df['TAVG_3D'] = df['TAVG'].rolling(3).mean()
    df['RR_3D'] = df['RR'].rolling(3).sum()
    df['RH_3D'] = df['RH_AVG'].rolling(3).mean()

    df['DAYOFYEAR'] = df['tanggal'].dt.dayofyear
    df['MONTH'] = df['tanggal'].dt.month

    df['RR_LAG_1'] = df['RR'].shift(1)
    df['RR_LAG_7'] = df['RR'].shift(7)
    df['RR_LAG_14'] = df['RR'].shift(14)

    df['RR_MEAN_7'] = df['RR'].rolling(7).mean()
    df['RR_STD_7'] = df['RR'].rolling(7).std()

    df['RR_MEAN_30'] = df['RR'].rolling(30).mean()
    df['RR_STD_30'] = df['RR'].rolling(30).std()

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
# MODEL METRIC
# =========================================================

BEST_MAE = 5.18
BEST_RMSE = 11.37
BEST_ACC = 69.44
BEST_PRECISION = 0.74
BEST_RECALL = 0.69
BEST_F1 = 0.70

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="big-title">
Rainfall Forecast Dashboard
</div>

<div class="subtitle">
Prediksi Curah Hujan Kota Surabaya menggunakan Machine Learning K-Nearest Neighbor (K-NN)
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================================
# TOP METRIC
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Model", "K-NN Hybrid")
m2.metric("Akurasi", "69.44%")
m3.metric("PCA", "10 Component")
m4.metric("Data", "2021-2025")

st.markdown("---")

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("## Pengaturan Prediksi")

box1, box2 = st.columns([2,1])

with box1:

    n_days = st.slider(
        "Jumlah Hari Prediksi",
        1,
        365,
        30
    )

with box2:

    st.write("")
    st.write("")

    run = st.button(
        "Jalankan Prediksi"
    )

# =========================================================
# CATEGORY
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
# FORECAST
# =========================================================

if run:

    with st.spinner("Sedang melakukan prediksi..."):

        forecast_df = df.copy()

        results = []

        for i in range(n_days):

            temp_df = add_features(
                forecast_df.copy()
            )

            temp_df = temp_df.ffill()
            temp_df = temp_df.fillna(0)

            latest = temp_df.iloc[-1]

            X_input = pd.DataFrame(
                [latest[features].values],
                columns=features
            )

            X_scaled = scaler.transform(X_input)
            X_pca = pca.transform(X_scaled)

            pred_rr = model.predict(X_pca)[0]

            pred_rr = max(0, pred_rr)

            next_date = (
                forecast_df['tanggal'].iloc[-1]
                + pd.Timedelta(days=1)
            )

            results.append({

                "Tanggal": next_date,

                "Curah Hujan Prediksi": round(pred_rr, 2),

                "Kategori": kategori_hujan(pred_rr)

            })

            new_row = forecast_df.iloc[-1].copy()

            new_row['tanggal'] = next_date
            new_row['RR'] = pred_rr

            forecast_df = pd.concat(
                [
                    forecast_df,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

        st.session_state.result_df = pd.DataFrame(results)

# =========================================================
# PAGE HASIL
# =========================================================

if (
    st.session_state.result_df is not None
    and menu == "hasil"
):

    result_df = st.session_state.result_df.copy()

    st.subheader("Model Hasil Prediksi")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", BEST_MAE)
    c2.metric("RMSE", BEST_RMSE)
    c3.metric("Accuracy", f"{BEST_ACC}%")
    c4.metric("F1 Score", BEST_F1)

    st.markdown("---")

    def highlight_kategori(row):

        kategori = row["Kategori"]

        if kategori == "Tidak Hujan":
            color = "background-color: #2c2c2c; color: white"

        elif kategori == "Hujan Ringan":
            color = "background-color: #2ecc71; color: black"

        elif kategori == "Hujan Sedang":
            color = "background-color: #f39c12; color: black"

        else:
            color = "background-color: #e74c3c; color: white"

        return [""] * 2 + [color]

    styled_df = (
        result_df.style
        .format({
            "Curah Hujan Prediksi": "{:.2f}"
        })
        .apply(
            highlight_kategori,
            axis=1
        )
    )

    st.dataframe(
        styled_df,
        use_container_width=True
    )

    csv = result_df.to_csv(index=False)

    st.download_button(
        "⬇Download CSV",
        csv,
        "hasil_prediksi.csv",
        use_container_width=True
    )

# =========================================================
# PAGE GRAFIK
# =========================================================

elif (
    st.session_state.result_df is not None
    and menu == "grafik"
):

    st.subheader(
        "Grafik Prediksi Curah Hujan"
    )

    gdf = st.session_state.result_df.copy()

    fig = px.line(
        gdf,
        x="Tanggal",
        y="Curah Hujan Prediksi",
        markers=True,
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(width=4),
        marker=dict(size=7)
    )

    fig.update_layout(
        height=650,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Tanggal",
        yaxis_title="Curah Hujan (mm)",
        title="Forecast Curah Hujan"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# PAGE HEATMAP
# =========================================================

elif (
    st.session_state.result_df is not None
    and menu == "heatmap"
):

    st.subheader(
        "Heatmap Curah Hujan"
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
        figsize=(18,5)
    )

    sns.heatmap(
        pivot,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

# =========================================================
# PAGE EVALUASI
# =========================================================

elif (
    st.session_state.result_df is not None
    and menu == "evaluasi"
):

    st.subheader("Evaluasi Model")

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

# =========================================================
# PAGE FLOWCHART
# =========================================================

elif menu == "flowchart":

    st.subheader("Flowchart Penelitian")

    st.markdown(
        """
        Berikut merupakan alur penelitian sistem prediksi curah hujan
        menggunakan metode Machine Learning K-Nearest Neighbor (K-NN).
        """
    )

    c1, c2, c3 = st.columns([1,4,1])

    with c2:

        st.image(
            "flowchart.jpg",
            width=700
        )

# =========================================================
# EMPTY STATE
# =========================================================

elif st.session_state.result_df is None:

    st.info(
        "Silakan jalankan prediksi terlebih dahulu"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <span style='color:gray'>
    Didukung oleh Data BMKG
    </span>
    </center>
    """,
    unsafe_allow_html=True
)