import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Pengelompokan Barang", layout="wide")

# Navigasi Halaman
page = st.sidebar.radio("Navigasi", ["📊 EDA", "📈 K-Means Clustering"])

# Fungsi untuk memuat data
@st.cache_data

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df[df['satuan_item'] == 'kg']
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['tahun'] = df['tanggal'].dt.year
    df['bulan'] = df['tanggal'].dt.month
    df = df.dropna(subset=['jumlah', 'kebutuhan'])
    df['jumlah'] = df['jumlah'].astype(int)
    df['kebutuhan'] = df['kebutuhan'].astype(int)
    return df

# Halaman EDA
if page == "📊 EDA":
    st.title("📊 Dashboard Analisis Barang di Pasar Tradisional Sumedang")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        
        # Filter Sidebar
        st.sidebar.header("Filter")
        pasar_filter = st.sidebar.multiselect("Pilih Pasar:", df['nama_pasar'].unique(), default=df['nama_pasar'].unique())
        barang_filter = st.sidebar.multiselect("Pilih Barang:", df['item_barang'].unique(), default=df['item_barang'].unique())
        tahun_filter = st.sidebar.multiselect("Pilih Tahun:", sorted(df['tahun'].unique()), default=sorted(df['tahun'].unique()))
        bulan_filter = st.sidebar.multiselect("Pilih Bulan:", sorted(df['bulan'].unique()), default=sorted(df['bulan'].unique()))

        # Terapkan Filter
        filtered_df = df[
            (df['nama_pasar'].isin(pasar_filter)) &
            (df['item_barang'].isin(barang_filter)) &
            (df['tahun'].isin(tahun_filter)) &
            (df['bulan'].isin(bulan_filter))
        ]

        # Chart 1
        st.subheader("Distribusi Jumlah dan Kebutuhan Barang")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['jumlah'], bins=50, color='blue', label='Jumlah', kde=True, ax=ax1)
        sns.histplot(filtered_df['kebutuhan'], bins=50, color='orange', label='Kebutuhan', kde=True, ax=ax1)
        ax1.set_xlabel("Jumlah / Kebutuhan")
        ax1.set_ylabel("Frekuensi")
        ax1.legend()
        st.pyplot(fig1)

        avg_jumlah = filtered_df['jumlah'].mean()
        avg_kebutuhan = filtered_df['kebutuhan'].mean()
        median_jumlah = filtered_df['jumlah'].median()
        median_kebutuhan = filtered_df['kebutuhan'].median()
        st.markdown("### 🔍 Insight Distribusi")
        st.markdown(f"- Rata-rata stok: **{avg_jumlah:.0f}**, Median: **{median_jumlah:.0f}**")
        st.markdown(f"- Rata-rata kebutuhan: **{avg_kebutuhan:.0f}**, Median: **{median_kebutuhan:.0f}**")

        if avg_jumlah > avg_kebutuhan:
            st.info("📦 Rata-rata stok lebih tinggi dari kebutuhan. Potensi overstock.")
        elif avg_jumlah < avg_kebutuhan:
            st.warning("⚠️ Rata-rata kebutuhan lebih tinggi dari stok. Potensi kekurangan barang.")
        else:
            st.success("✅ Stok dan kebutuhan seimbang.")

        # Chart 2
        st.subheader("Hubungan Jumlah vs Kebutuhan")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.scatterplot(data=filtered_df, x='jumlah', y='kebutuhan', alpha=0.4, ax=ax2)
        st.pyplot(fig2)

        correlation = filtered_df['jumlah'].corr(filtered_df['kebutuhan'])
        st.markdown("### 🔍 Insight Korelasi")
        st.markdown(f"- Korelasi: **{correlation:.2f}**")
        if correlation > 0.7:
            st.success("📈 Hubungan kuat: stok cenderung sesuai kebutuhan.")
        elif correlation > 0.4:
            st.info("📉 Korelasi sedang.")
        else:
            st.warning("❗ Korelasi lemah: stok tidak berbasis permintaan.")

        # Chart 3
        st.subheader("Barang dengan Kebutuhan Tertinggi")
        top_items = filtered_df.groupby('item_barang')[['jumlah', 'kebutuhan']].mean().sort_values(by='kebutuhan', ascending=False).head(10)
        fig3, ax3 = plt.subplots(figsize=(10,6))
        top_items.plot(kind='bar', ax=ax3, color=['skyblue', 'salmon'])
        st.pyplot(fig3)

        most_needed = top_items.index[0]
        second_needed = top_items.index[1]
        st.markdown("### 🔍 Insight Kebutuhan Barang")
        st.markdown(f"- Paling dibutuhkan: **{most_needed}**, disusul **{second_needed}**")

        # Chart 4
        st.subheader("Rata-rata Jumlah dan Kebutuhan per Pasar")
        pasar_stats = filtered_df.groupby('nama_pasar')[['jumlah', 'kebutuhan']].mean().sort_values(by='kebutuhan', ascending=False).head(5)
        fig4, ax4 = plt.subplots(figsize=(10,6))
        pasar_stats.plot(kind='barh', ax=ax4, color=['mediumseagreen', 'tomato'])
        st.pyplot(fig4)

        st.markdown("### 🔍 Insight Pasar")
        st.markdown(f"- Kebutuhan tertinggi: **{pasar_stats.index[0]}**")
        st.markdown(f"- Terendah di antara top 5: **{pasar_stats.index[-1]}**")

# Halaman K-Means Clustering
elif page == "📈 K-Means Clustering":
    st.title("📈 K-Means Clustering: Pengelompokan Barang Berdasarkan Jumlah & Kebutuhan")
    uploaded_file = st.file_uploader("Unggah dataset", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)

        # Sidebar Filter
        st.sidebar.header("Filter")
        pasar_filter = st.sidebar.multiselect("Pilih Pasar:", df['nama_pasar'].unique(), default=df['nama_pasar'].unique())
        tahun_filter = st.sidebar.multiselect("Pilih Tahun:", sorted(df['tahun'].unique()), default=sorted(df['tahun'].unique()))
        bulan_filter = st.sidebar.multiselect("Pilih Bulan:", sorted(df['bulan'].unique()), default=sorted(df['bulan'].unique()))

        # Terapkan filter
        filtered_df = df[
            (df['nama_pasar'].isin(pasar_filter)) &
            (df['tahun'].isin(tahun_filter)) &
            (df['bulan'].isin(bulan_filter))
        ]

        if filtered_df.empty:
            st.warning("⚠️ Tidak ada data yang cocok dengan filter. Silakan ubah filter Anda.")
        else:
            # Pilih fitur untuk clustering
            clustering_data = filtered_df[['jumlah', 'kebutuhan']].copy().fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clustering_data)

            k = st.slider("Pilih jumlah cluster (K):", 2, 10, 3)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            filtered_df['cluster'] = clusters
            clustering_data['cluster'] = clusters

            st.subheader("Visualisasi Cluster")
            fig = px.scatter(clustering_data, x='jumlah', y='kebutuhan', color='cluster',
                             title='Hasil Clustering Jumlah vs Kebutuhan',
                             labels={'jumlah': 'Jumlah Barang', 'kebutuhan': 'Kebutuhan Barang'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📌 Insight Otomatis dari Clustering")
            for c in sorted(filtered_df['cluster'].unique()):
                subset = filtered_df[filtered_df['cluster'] == c]
                avg_jumlah = int(subset['jumlah'].mean())
                avg_kebutuhan = int(subset['kebutuhan'].mean())
                top_items = subset['item_barang'].value_counts().head(3).index.tolist()

                st.markdown(f"""
                - **Cluster {c}**
                  - Rata-rata Jumlah: **{avg_jumlah}**
                  - Rata-rata Kebutuhan: **{avg_kebutuhan}**
                  - Contoh Barang Teratas: {', '.join(top_items)}
                  - 📌 Interpretasi: Barang dalam cluster ini cenderung memiliki pola {'tinggi' if avg_kebutuhan > 1000 else 'rendah'} kebutuhan.
                """)
    else:
        st.warning("Silakan unggah file dataset terlebih dahulu.")
