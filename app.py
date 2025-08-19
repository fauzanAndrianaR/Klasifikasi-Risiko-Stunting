import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sistem Klasifikasi Risiko stunting berdasarkan Gizi Balita",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("random_forest_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except:
        st.error("Model tidak ditemukan!")
        return None, None

model, features = load_model_and_features()

st.markdown("""
<div class="main-header">
    <h1>🍽️ Sistem Klasifikasi Risiko stunting berdasarkan Gizi Balita</h1>
    <p>Aplikasi klasifikasi status gizi balita berdasarkan indikator antropometri</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 📊 Informasi Model")
if model is not None:
    st.sidebar.success("✅ Model berhasil dimuat")
    st.sidebar.info(f"📋 Jumlah fitur: {len(features) if features else 0}")
else:
    st.sidebar.error("❌ Model gagal dimuat")

with st.sidebar.expander("ℹ️ Tentang Aplikasi"):
    st.markdown("""
    Aplikasi ini menggunakan algoritma Random Forest untuk klasifikasi status gizi balita.

    **Kategori:**
    - Gizi Kurang
    - Gizi Baik
    - Gizi Lebih
    """)

with st.sidebar.expander("📈 Statistik Session"):
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    if 'csv_processed' not in st.session_state:
        st.session_state.csv_processed = 0

    st.metric("Prediksi Manual", st.session_state.prediction_count)
    st.metric("File CSV Diproses", st.session_state.csv_processed)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📅 Download Template")
if features:
    template_df = pd.DataFrame(columns=features)
    st.sidebar.download_button(
        "📄 Download Template CSV",
        data=template_df.to_csv(index=False),
        file_name="template_gizi_balita.csv",
        mime="text/csv"
    )

if model and features:
    tab1, tab2, tab3 = st.tabs(["📝 Input Manual", "📁 Upload CSV", "📊 Analisis Data"])

    with tab1:
        st.markdown("### 📝 Input Data Balita Secara Manual")
        col1, col2 = st.columns([2, 1])

        with col1:
            with st.form("manual_input_form"):
                input_data = {}

                if 'JK' in features:
                    input_data['JK'] = st.selectbox(
                        "👶 Jenis Kelamin",
                        options=[0, 1],
                        format_func=lambda x: "👦 Laki-laki" if x == 0 else "👧 Perempuan"
                    )

                numeric_features = [f for f in features if f != 'JK']
                colA, colB = st.columns(2)
                with colA:
                    for feature in numeric_features[:len(numeric_features)//2]:
                        input_data[feature] = st.number_input(f"📊 {feature}", step=0.1)
                with colB:
                    for feature in numeric_features[len(numeric_features)//2:]:
                        input_data[feature] = st.number_input(f"📊 {feature}", step=0.1)

                submitted = st.form_submit_button("🔍 Prediksi Status Gizi")
                if submitted:
                    try:
                        input_df = pd.DataFrame([input_data], columns=features)
                        prediction = model.predict(input_df)[0]
                        st.session_state.prediction_count += 1

                        st.markdown("### 📌 Hasil Prediksi")
                        if prediction == "Gizi Kurang":
                            st.error(f"🔴 **Status Gizi: {prediction}**")
                        elif prediction == "Gizi Baik":
                            st.success(f"🟢 **Status Gizi: {prediction}**")
                        else:
                            st.warning(f"🟡 **Status Gizi: {prediction}**")

                        if hasattr(model, 'classes_') and hasattr(model, 'predict_proba'):
                            prob_df = pd.DataFrame({
                                'Kategori': model.classes_,
                                'Probabilitas': model.predict_proba(input_df)[0]
                            })
                            fig = px.bar(prob_df, x='Kategori', y='Probabilitas',
                                         title="Distribusi Probabilitas Prediksi",
                                         color='Probabilitas',
                                         color_continuous_scale='Viridis')
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")

        with col2:
            st.markdown("### 💡 Tips Penggunaan")
            st.markdown("""
            <div class="info-box">
            <strong>Panduan Input:</strong><br>
            • Isi semua field<br>
            • Pastikan data akurat<br>
            • Klik tombol prediksi untuk hasil
            </div>
            """, unsafe_allow_html=True)
            if st.button("👁️ Preview Input Data"):
                if len(input_data) > 0:
                    st.dataframe(pd.DataFrame([input_data], columns=features), use_container_width=True)
                else:
                    st.warning("Belum ada data yang diinput")

    with tab2:
        col_upload = st.columns([2, 1])
        with col_upload[0]:
            st.markdown("### 📁 Upload dan Prediksi File CSV")
            uploaded_file = st.file_uploader("📂 Pilih file CSV", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File berhasil diupload ({df.shape[0]} baris)")

                    missing_cols = [col for col in features if col not in df.columns]
                    extra_cols = [col for col in df.columns if col not in features]

                    c1, c2 = st.columns(2)
                    with c1:
                        if missing_cols:
                            st.error(f"❌ Kolom hilang: {missing_cols}")
                        else:
                            st.success("✅ Semua kolom tersedia")
                    with c2:
                        if extra_cols:
                            st.warning(f"⚠️ Kolom tambahan akan diabaikan: {extra_cols}")

                    with st.expander("👁️ Preview Data (5 baris)"):
                        st.dataframe(df.head(), use_container_width=True)

                    with st.expander("📊 Statistik Data"):
                        st.dataframe(df.describe(), use_container_width=True)

                    if not missing_cols:
                        if st.button("🔍 Prediksi Semua Data", use_container_width=True):
                            try:
                                df_result = df.copy()
                                df_result['Prediksi_Status_Gizi'] = model.predict(df[features])
                                st.session_state.csv_processed += 1

                                st.markdown("### 📌 Hasil Prediksi")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Data", len(df_result))
                                with col2:
                                    st.metric("Jumlah Prediksi", df_result['Prediksi_Status_Gizi'].count())
                                with col3:
                                    st.metric("Kategori Ditemukan", df_result['Prediksi_Status_Gizi'].nunique())

                                pred_counts = df_result['Prediksi_Status_Gizi'].value_counts()
                                fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index,
                                                 title="Distribusi Prediksi Status Gizi")
                                st.plotly_chart(fig_pie, use_container_width=True)

                                st.dataframe(df_result, use_container_width=True)

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.download_button(
                                    "📅 Download Hasil Prediksi",
                                    data=df_result.to_csv(index=False).encode("utf-8"),
                                    file_name=f"hasil_prediksi_{timestamp}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
                except Exception as e:
                    st.error(f"Gagal membaca file: {str(e)}")

        with col_upload[1]:
            st.markdown("### 📋 Format File CSV")
            st.markdown("""
            <div class="info-box">
            <strong>Format Wajib:</strong><br>
            • CSV dengan koma (,)<br>
            • Kolom sesuai template<br>
            • Header wajib<br>
            • Encoding UTF-8
            </div>
            """, unsafe_allow_html=True)
            if features:
                st.markdown("**Kolom yang diperlukan:**")
                for i, feature in enumerate(features, 1):
                    st.write(f"{i}. `{feature}`")

    with tab3:
        st.markdown("### 📊 Analisis dan Visualisasi Data")
        st.markdown("""
        <div class="info-box">
        <strong>Fitur Analisis:</strong><br>
        • Importance fitur<br>
        • Distribusi kategori<br>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, 'feature_importances_'):
            st.markdown("#### 🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Fitur': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig = px.bar(importance_df, x='Importance', y='Fitur', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 Fitur tambahan akan tersedia di versi berikutnya.")
else:
    st.error("❌ Aplikasi tidak dapat dimuat karena model atau fitur tidak ditemukan!")
    st.markdown("""
    **Troubleshooting:**
    1. Pastikan file `random_forest_model.pkl` dan `model_features.pkl` tersedia
    2. Letakkan di direktori yang sama
    3. Restart aplikasi setelah memastikan file tersedia
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🍽️ Sistem Klasifikasi Status Gizi Balita | Dibuat dengan ❤️ menggunakan Streamlit</p>
    <p><small>Versi 2.0 | © 2024 - Untuk keperluan edukasi dan penelitian</small></p>
</div>
""", unsafe_allow_html=True)
