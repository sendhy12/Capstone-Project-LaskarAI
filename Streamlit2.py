import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Pasar Sumedang", layout="wide")

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

        # Chart Time Series - Tren Kebutuhan dan Ketersediaan Barang
        st.subheader("📈 Tren Total Kebutuhan vs Ketersediaan Barang Seiring Waktu")

        # Kelompokkan data berdasarkan tanggal untuk mendapatkan total kebutuhan dan total ketersediaan
        filtered_df_grouped = filtered_df.groupby('tanggal')[['kebutuhan', 'jumlah']].sum().reset_index()

        # Buat grafik dengan dua garis (kebutuhan dan ketersediaan barang)
        fig_ts = px.line(filtered_df_grouped, x='tanggal', y=['kebutuhan', 'jumlah'],
                        title="Tren Total Kebutuhan vs Ketersediaan Barang Harian",
                        labels={'tanggal': 'Tanggal', 'value': 'Jumlah Barang', 'variable': 'Kategori'},
                        line_shape='linear')

        # Tampilkan grafik
        st.plotly_chart(fig_ts, use_container_width=True)

        # Insight Otomatis dari Tren Kebutuhan vs Ketersediaan
        st.subheader("📌 Insight Otomatis dari Tren")

        # Hitung Rata-rata dan Korelasi
        avg_kebutuhan = filtered_df_grouped['kebutuhan'].mean()
        avg_ketersediaan = filtered_df_grouped['jumlah'].mean()
        correlation = filtered_df_grouped['kebutuhan'].corr(filtered_df_grouped['jumlah'])

        # Tampilkan Rata-rata
        st.markdown(f"- 📦 **Rata-rata Ketersediaan Barang per Bulan**: {avg_ketersediaan:.2f}")
        st.markdown(f"- 📊 **Rata-rata Kebutuhan Barang per Bulan**: {avg_kebutuhan:.2f}")
        st.markdown(f"- 🔗 **Korelasi antara Ketersediaan dan Kebutuhan**: {correlation:.2f}")

        # Interpretasi Berdasarkan Korelasi
        if correlation > 0.7:
            st.success("✅ Hubungan kuat antara kebutuhan dan stok, distribusi cenderung sesuai permintaan.")
        elif correlation > 0.4:
            st.info("📉 Korelasi sedang, ada beberapa ketidakseimbangan distribusi barang.")
        else:
            st.warning("❗ Korelasi lemah, stok tidak selalu mencerminkan kebutuhan pasar.")

        # Analisis Tren Stok vs Kebutuhan
        if avg_ketersediaan > avg_kebutuhan:
            st.info("📦 **Rata-rata stok lebih tinggi dari kebutuhan** → Potensi overstock.")
        elif avg_ketersediaan < avg_kebutuhan:
            st.warning("⚠️ **Kebutuhan lebih tinggi dari stok** → Potensi kelangkaan barang.")
        else:
            st.success("✅ **Stok dan kebutuhan relatif seimbang**.")

        # Analisis Tren Perubahan
        trend_kebutuhan = filtered_df_grouped['kebutuhan'].diff().mean()
        trend_ketersediaan = filtered_df_grouped['jumlah'].diff().mean()

        if trend_kebutuhan > 0:
            st.info("📈 **Kebutuhan barang cenderung meningkat dari bulan ke bulan.**")
        elif trend_kebutuhan < 0:
            st.warning("📉 **Kebutuhan barang menunjukkan tren penurunan.**")

        if trend_ketersediaan > 0:
            st.success("📈 **Stok barang mengalami peningkatan secara bertahap.**")
        elif trend_ketersediaan < 0:
            st.warning("📉 **Stok barang mengalami penurunan dari bulan ke bulan.**")

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

        # Pilih fitur untuk clustering
        clustering_data = df[['jumlah', 'kebutuhan']].copy().fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clustering_data)

        k = st.slider("Pilih jumlah cluster (K):", 2, 10, 3)

        # Penerapan K-Means Clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(X_scaled)

        st.subheader("🔍 Visualisasi Cluster")
        fig = px.scatter(df, x='jumlah', y='kebutuhan', color=df['cluster'].astype(str),
                         title="Hasil Clustering Jumlah vs Kebutuhan",
                         labels={'jumlah': 'Jumlah Barang', 'kebutuhan': 'Kebutuhan Barang', 'cluster': 'Cluster'})
        st.plotly_chart(fig, use_container_width=True)

        # 📌 Insight Otomatis dari Clustering
        st.subheader("📌 Insight Otomatis dari Hasil Clustering")
        for c in sorted(df['cluster'].unique()):
            subset = df[df['cluster'] == c]
            avg_jumlah = int(subset['jumlah'].mean())
            avg_kebutuhan = int(subset['kebutuhan'].mean())
            top_items = subset['item_barang'].value_counts().head(3).index.tolist()

            st.markdown(f"""
            - **Cluster {c}**
              - Rata-rata Jumlah Barang: **{avg_jumlah}**
              - Rata-rata Kebutuhan Barang: **{avg_kebutuhan}**
              - Barang Teratas dalam Cluster: {', '.join(top_items)}
              - 📌 Interpretasi: Barang dalam cluster ini cenderung memiliki pola kebutuhan {'tinggi' if avg_kebutuhan > 1000 else 'rendah'}.
            """)

        # 📊 Analisis Lanjutan Setelah Clustering
        st.subheader("📊 Analisis Lanjutan Setelah Clustering")

        # **Distribusi Klaster**
        st.subheader("📈 Distribusi Barang Berdasarkan Klaster")
        fig_dist_cluster = px.bar(df.groupby('cluster')[['jumlah', 'kebutuhan']].mean().reset_index(),
                                  x='cluster', y=['jumlah', 'kebutuhan'], barmode='group',
                                  title="Rata-rata Jumlah & Kebutuhan Barang per Klaster",
                                  labels={'cluster': 'Cluster', 'value': 'Rata-rata Jumlah', 'variable': 'Kategori'})
        st.plotly_chart(fig_dist_cluster)

        # **Tren Kebutuhan dan Ketersediaan Barang per Klaster**
        st.subheader("📊 Tren Rata-rata Kebutuhan & Ketersediaan per Bulan Berdasarkan Klaster")
        df_clustered_grouped = df.groupby(['cluster', 'tahun', 'bulan'])[['jumlah', 'kebutuhan']].mean().reset_index()
        df_clustered_grouped['tanggal'] = pd.to_datetime(df_clustered_grouped[['tahun', 'bulan']].astype(str).agg('-'.join, axis=1) + '-01')


        fig_ts_cluster = px.line(df_clustered_grouped, x='tanggal', y=['jumlah', 'kebutuhan'], color='cluster',
                                 title="Tren Rata-rata Kebutuhan vs Ketersediaan per Bulan Berdasarkan Klaster",
                                 labels={'tanggal': 'Bulan', 'value': 'Rata-rata Jumlah', 'variable': 'Kategori', 'cluster': 'Cluster'})
        st.plotly_chart(fig_ts_cluster, use_container_width=True)

        # **Proporsi Data dalam Tiap Klaster**
        st.subheader("📊 Proporsi Data dalam Tiap Klaster")
        cluster_counts = df['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'count']
        fig_pie_cluster = px.pie(cluster_counts, names='cluster', values='count', title="Proporsi Data dalam Tiap Klaster")
        st.plotly_chart(fig_pie_cluster)

        # **Analisis Lebih Lanjut**
        st.subheader("📌 Analisis Lebih Lanjut")

        highest_demand_cluster = df.groupby('cluster')['kebutuhan'].mean().idxmax()
        highest_stock_cluster = df.groupby('cluster')['jumlah'].mean().idxmax()

        st.markdown(f"- Cluster dengan **kebutuhan tertinggi**: **Cluster {highest_demand_cluster}**")
        st.markdown(f"- Cluster dengan **stok tertinggi**: **Cluster {highest_stock_cluster}**")

        if highest_demand_cluster != highest_stock_cluster:
            st.warning("⚠️ Distribusi barang mungkin tidak seimbang antara kebutuhan dan stok. Pertimbangkan strategi redistribusi barang.")
        else:
            st.success("✅ Cluster dengan kebutuhan tertinggi juga memiliki stok tinggi, distribusi relatif seimbang.")

    else:
        st.warning("Silakan unggah file dataset terlebih dahulu.")

