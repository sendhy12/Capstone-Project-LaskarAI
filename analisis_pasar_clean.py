import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration constants for the application"""
    PAGE_TITLE = "Analisis Pasar Sumedang"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    
    # Chart settings
    FIGSIZE_LARGE = (12, 8)
    FIGSIZE_MEDIUM = (10, 6)
    FIGSIZE_SMALL = (8, 5)
    
    # Colors
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#17becf'
    }
    
    # PDF settings
    PDF_MARGIN = 10
    PDF_FONT_SIZE_TITLE = 12
    PDF_FONT_SIZE_NORMAL = 11
    PDF_FONT_SIZE_SMALL = 8

# ==================== DATA PROCESSING ====================
class DataProcessor:
    """Handles all data processing operations"""
    
    @staticmethod
    @st.cache_data
    def load_and_clean_data(uploaded_file) -> pd.DataFrame:
        """Load and clean the uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data cleaning pipeline
            df = df[df['satuan_item'] == 'kg'].copy()
            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
            df['tahun'] = df['tanggal'].dt.year
            df['bulan'] = df['tanggal'].dt.month
            df = df.dropna(subset=['jumlah', 'kebutuhan', 'tanggal'])
            
            # Convert to appropriate data types
            df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce').astype('Int64')
            df['kebutuhan'] = pd.to_numeric(df['kebutuhan'], errors='coerce').astype('Int64')
            
            # Remove any remaining NaN values
            df = df.dropna(subset=['jumlah', 'kebutuhan'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to the dataframe"""
        filtered_df = df.copy()
        
        for column, values in filters.items():
            if values and column in df.columns:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        return filtered_df
    
    @staticmethod
    def get_statistics(df: pd.DataFrame) -> Dict:
        """Calculate basic statistics for the dataset"""
        return {
            'total_records': len(df),
            'avg_jumlah': df['jumlah'].mean(),
            'avg_kebutuhan': df['kebutuhan'].mean(),
            'median_jumlah': df['jumlah'].median(),
            'median_kebutuhan': df['kebutuhan'].median(),
            'correlation': df['jumlah'].corr(df['kebutuhan'])
        }

# ==================== VISUALIZATION ====================
class Visualizer:
    """Handles all visualization operations"""
    
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame) -> plt.Figure:
        """Create distribution plot for jumlah and kebutuhan"""
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        
        sns.histplot(df['jumlah'], bins=50, color=Config.COLORS['primary'], 
                    label='Jumlah', kde=True, alpha=0.7, ax=ax)
        sns.histplot(df['kebutuhan'], bins=50, color=Config.COLORS['secondary'], 
                    label='Kebutuhan', kde=True, alpha=0.7, ax=ax)
        
        ax.set_xlabel("Jumlah / Kebutuhan")
        ax.set_ylabel("Frekuensi")
        ax.set_title("Distribusi Jumlah dan Kebutuhan Barang")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame) -> plt.Figure:
        """Create scatter plot for jumlah vs kebutuhan"""
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_SMALL)
        
        sns.scatterplot(data=df, x='jumlah', y='kebutuhan', 
                       alpha=0.6, color=Config.COLORS['primary'], ax=ax)
        ax.set_title("Hubungan Jumlah vs Kebutuhan")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_top_items_plot(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
        """Create bar plot for top items by kebutuhan"""
        top_items = (df.groupby('item_barang')[['jumlah', 'kebutuhan']]
                    .mean()
                    .sort_values(by='kebutuhan', ascending=False)
                    .head(top_n))
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        top_items.plot(kind='bar', ax=ax, 
                      color=[Config.COLORS['info'], Config.COLORS['warning']])
        ax.set_title(f"Top {top_n} Barang dengan Kebutuhan Tertinggi")
        ax.set_xlabel("Item Barang")
        ax.set_ylabel("Rata-rata")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_market_comparison_plot(df: pd.DataFrame, top_n: int = 5) -> plt.Figure:
        """Create horizontal bar plot for market comparison"""
        pasar_stats = (df.groupby('nama_pasar')[['jumlah', 'kebutuhan']]
                      .mean()
                      .sort_values(by='kebutuhan', ascending=False)
                      .head(top_n))
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        pasar_stats.plot(kind='barh', ax=ax, 
                        color=[Config.COLORS['success'], Config.COLORS['warning']])
        ax.set_title(f"Top {top_n} Pasar - Rata-rata Jumlah dan Kebutuhan")
        ax.set_xlabel("Rata-rata")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ==================== ANALYSIS ====================
class Analyzer:
    """Handles analysis and insights generation"""
    
    @staticmethod
    def generate_distribution_insights(stats: Dict) -> List[str]:
        """Generate insights from distribution statistics"""
        insights = []
        insights.append(f"Rata-rata stok: **{stats['avg_jumlah']:.0f}**, Median: **{stats['median_jumlah']:.0f}**")
        insights.append(f"Rata-rata kebutuhan: **{stats['avg_kebutuhan']:.0f}**, Median: **{stats['median_kebutuhan']:.0f}**")
        
        return insights
    
    @staticmethod
    def get_stock_status(avg_jumlah: float, avg_kebutuhan: float) -> Tuple[str, str]:
        """Determine stock status based on averages"""
        if avg_jumlah > avg_kebutuhan:
            return "info", "📦 Rata-rata stok lebih tinggi dari kebutuhan. Potensi overstock."
        elif avg_jumlah < avg_kebutuhan:
            return "warning", "⚠️ Rata-rata kebutuhan lebih tinggi dari stok. Potensi kekurangan barang."
        else:
            return "success", "✅ Stok dan kebutuhan seimbang."
    
    @staticmethod
    def get_correlation_insight(correlation: float) -> Tuple[str, str]:
        """Generate correlation insights"""
        if correlation > 0.7:
            return "success", f"📈 Hubungan kuat (r={correlation:.2f}): stok cenderung sesuai kebutuhan."
        elif correlation > 0.4:
            return "info", f"📉 Korelasi sedang (r={correlation:.2f})."
        else:
            return "warning", f"❗ Korelasi lemah (r={correlation:.2f}): stok tidak berbasis permintaan."

# ==================== CLUSTERING ====================
class ClusterAnalyzer:
    """Handles K-Means clustering operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = StandardScaler()
        self.kmeans = None
        self.X_scaled = None
    
    def prepare_data(self) -> bool:
        """Prepare data for clustering"""
        try:
            X = self.df[['jumlah', 'kebutuhan']]
            self.X_scaled = self.scaler.fit_transform(X)
            return True
        except Exception as e:
            st.error(f"Error preparing data for clustering: {str(e)}")
            return False
    
    def perform_clustering(self, n_clusters: int) -> np.ndarray:
        """Perform K-Means clustering"""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = self.kmeans.fit_predict(self.X_scaled)
        return clusters
    
    def generate_cluster_insights(self, df_with_clusters: pd.DataFrame) -> List[Dict]:
        """Generate insights for each cluster"""
        insights = []
        
        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            subset = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            insight = {
                'cluster_id': cluster_id,
                'avg_jumlah': int(subset['jumlah'].mean()),
                'avg_kebutuhan': int(subset['kebutuhan'].mean()),
                'total_items': len(subset),
                'top_items': subset['item_barang'].value_counts().head(3).index.tolist(),
                'interpretation': 'tinggi' if subset['kebutuhan'].mean() > 1000 else 'rendah'
            }
            insights.append(insight)
        
        return insights

# ==================== PDF GENERATOR ====================
class PDFGenerator:
    """Handles PDF report generation"""
    
    class ReportPDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', Config.PDF_FONT_SIZE_TITLE)
            self.cell(0, 6, "PEMERINTAH KABUPATEN SUMEDANG", ln=True, align='C')
            self.cell(0, 6, "DINAS KOPERASI, UKM, PERDAGANGAN DAN PERINDUSTRIAN", ln=True, align='C')
            
            self.set_font('Arial', '', Config.PDF_FONT_SIZE_SMALL + 2)
            self.cell(0, 6, "Jl. Raya Sumedang No.123, Sumedang, Jawa Barat", ln=True, align='C')
            self.cell(0, 6, "Telp: (0261) 123456 | Email: dinaskopdag@sumedang.go.id", ln=True, align='C')
            
            self.ln(5)
            self.line(Config.PDF_MARGIN, self.get_y(), 200, self.get_y())
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', Config.PDF_FONT_SIZE_SMALL)
            self.cell(0, 10, f"Halaman {self.page_no()} | Dicetak: {datetime.today().strftime('%d-%m-%Y')}", 
                     0, 0, 'C')
    
    @staticmethod
    def generate_clustering_report(filters: Dict, insights: List[Dict]) -> BytesIO:
        """Generate PDF report for clustering analysis"""
        pdf = PDFGenerator.ReportPDF()
        pdf.add_page()
        
        # Filter information
        pdf.set_font("Arial", 'B', Config.PDF_FONT_SIZE_TITLE)
        pdf.cell(0, 10, "Laporan Analisis Clustering", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', Config.PDF_FONT_SIZE_NORMAL)
        pdf.cell(0, 10, "Keterangan Filter:", ln=True)
        pdf.set_font("Arial", '', Config.PDF_FONT_SIZE_NORMAL)
        
        for key, value in filters.items():
            if value:
                pdf.cell(0, 8, f"{key.replace('_', ' ').title()}: {', '.join(map(str, value))}", ln=True)
        
        pdf.ln(5)
        
        # Clustering results
        pdf.set_font("Arial", 'B', Config.PDF_FONT_SIZE_TITLE)
        pdf.cell(0, 10, "Hasil Clustering & Insight", ln=True)
        
        for insight in insights:
            pdf.set_font("Arial", 'B', Config.PDF_FONT_SIZE_NORMAL)
            pdf.cell(0, 8, f"Cluster {insight['cluster_id']}", ln=True)
            pdf.set_font("Arial", '', Config.PDF_FONT_SIZE_NORMAL)
            
            pdf.cell(0, 8, f"  - Rata-rata Jumlah     : {insight['avg_jumlah']}", ln=True)
            pdf.cell(0, 8, f"  - Rata-rata Kebutuhan  : {insight['avg_kebutuhan']}", ln=True)
            pdf.cell(0, 8, f"  - Total Item           : {insight['total_items']}", ln=True)
            pdf.cell(0, 8, f"  - Contoh Barang Teratas: {', '.join(insight['top_items'])}", ln=True)
            pdf.cell(0, 8, f"  - Interpretasi         : Pola kebutuhan {insight['interpretation']}", ln=True)
            pdf.ln(2)
            pdf.line(Config.PDF_MARGIN, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
        
        # Save to BytesIO
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        
        return pdf_output

# ==================== UI COMPONENTS ====================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def create_filters_sidebar(df: pd.DataFrame) -> Dict:
        """Create filter sidebar and return filter values"""
        st.sidebar.header("🔍 Filter Data")
        
        filters = {}
        
        if 'nama_pasar' in df.columns:
            filters['nama_pasar'] = st.sidebar.multiselect(
                "Pilih Pasar:", 
                options=sorted(df['nama_pasar'].unique()),
                default=sorted(df['nama_pasar'].unique())
            )
        
        if 'item_barang' in df.columns:
            filters['item_barang'] = st.sidebar.multiselect(
                "Pilih Barang:", 
                options=sorted(df['item_barang'].unique()),
                default=sorted(df['item_barang'].unique())[:10]  # Limit default selection
            )
        
        if 'tahun' in df.columns:
            filters['tahun'] = st.sidebar.multiselect(
                "Pilih Tahun:", 
                options=sorted(df['tahun'].unique()),
                default=sorted(df['tahun'].unique())
            )
        
        if 'bulan' in df.columns:
            filters['bulan'] = st.sidebar.multiselect(
                "Pilih Bulan:", 
                options=list(range(1, 13)),
                default=sorted(df['bulan'].unique())
            )
        
        return filters
    
    @staticmethod
    def display_metrics(stats: Dict):
        """Display key metrics in columns"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{stats['total_records']:,}")
        with col2:
            st.metric("Avg Jumlah", f"{stats['avg_jumlah']:.0f}")
        with col3:
            st.metric("Avg Kebutuhan", f"{stats['avg_kebutuhan']:.0f}")
        with col4:
            st.metric("Correlation", f"{stats['correlation']:.2f}")

# ==================== MAIN APPLICATION ====================
class MarketAnalysisApp:
    """Main application class"""
    
    def __init__(self):
        self.setup_page()
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.analyzer = Analyzer()
        self.ui = UIComponents()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT,
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the main application"""
        st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
        
        # Navigation
        page = st.sidebar.radio(
            "📍 Navigasi", 
            ["📊 Exploratory Data Analysis", "🎯 K-Means Clustering"],
            index=0
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Unggah file CSV Anda", 
            type=["csv"],
            help="Pastikan file CSV berisi kolom: nama_pasar, item_barang, jumlah, kebutuhan, tanggal, satuan_item"
        )
        
        if uploaded_file is not None:
            df = self.data_processor.load_and_clean_data(uploaded_file)
            
            if df.empty:
                st.error("❌ Data tidak dapat dimuat atau tidak valid.")
                return
            
            st.success(f"✅ Data berhasil dimuat: {len(df):,} records")
            
            if page == "📊 Exploratory Data Analysis":
                self.render_eda_page(df)
            else:
                self.render_clustering_page(df)
        else:
            st.info("👆 Silakan unggah file CSV untuk memulai analisis.")
    
    def render_eda_page(self, df: pd.DataFrame):
        """Render EDA page"""
        st.header("📊 Exploratory Data Analysis")
        
        # Filters
        filters = self.ui.create_filters_sidebar(df)
        filtered_df = self.data_processor.apply_filters(df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ Tidak ada data yang cocok dengan filter yang dipilih.")
            return
        
        # Statistics
        stats = self.data_processor.get_statistics(filtered_df)
        self.ui.display_metrics(stats)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribusi Data")
            fig1 = self.visualizer.create_distribution_plot(filtered_df)
            st.pyplot(fig1)
            
            # Insights
            with st.expander("🔍 Insight Distribusi"):
                insights = self.analyzer.generate_distribution_insights(stats)
                for insight in insights:
                    st.markdown(f"• {insight}")
                
                status_type, status_msg = self.analyzer.get_stock_status(
                    stats['avg_jumlah'], stats['avg_kebutuhan']
                )
                getattr(st, status_type)(status_msg)
        
        with col2:
            st.subheader("🔄 Korelasi Jumlah vs Kebutuhan")
            fig2 = self.visualizer.create_scatter_plot(filtered_df)
            st.pyplot(fig2)
            
            # Correlation insight
            with st.expander("🔍 Insight Korelasi"):
                corr_type, corr_msg = self.analyzer.get_correlation_insight(stats['correlation'])
                getattr(st, corr_type)(corr_msg)
        
        # Additional charts
        st.subheader("🏆 Top Barang dengan Kebutuhan Tertinggi")
        fig3 = self.visualizer.create_top_items_plot(filtered_df)
        st.pyplot(fig3)
        
        st.subheader("🏪 Perbandingan Pasar")
        fig4 = self.visualizer.create_market_comparison_plot(filtered_df)
        st.pyplot(fig4)
    
    def render_clustering_page(self, df: pd.DataFrame):
        """Render clustering page"""
        st.header("🎯 K-Means Clustering Analysis")
        
        # Filters (simplified for clustering)
        filters = {
            'nama_pasar': st.sidebar.multiselect(
                "Pilih Pasar:", 
                options=sorted(df['nama_pasar'].unique()),
                default=sorted(df['nama_pasar'].unique())
            ),
            'tahun': st.sidebar.multiselect(
                "Pilih Tahun:", 
                options=sorted(df['tahun'].unique()),
                default=sorted(df['tahun'].unique())
            ),
            'bulan': st.sidebar.multiselect(
                "Pilih Bulan:", 
                options=sorted(df['bulan'].unique()),
                default=sorted(df['bulan'].unique())
            )
        }
        
        filtered_df = self.data_processor.apply_filters(df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ Tidak ada data yang cocok dengan filter yang dipilih.")
            return
        
        # Clustering parameters
        k = st.slider("🎛️ Pilih jumlah cluster (K):", min_value=2, max_value=10, value=3)
        
        # Perform clustering
        cluster_analyzer = ClusterAnalyzer(filtered_df)
        if not cluster_analyzer.prepare_data():
            return
        
        clusters = cluster_analyzer.perform_clustering(k)
        filtered_df_with_clusters = filtered_df.copy()
        filtered_df_with_clusters['cluster'] = clusters
        
        # Visualization
        st.subheader("📊 Visualisasi Cluster")
        fig = px.scatter(
            filtered_df_with_clusters, 
            x='jumlah', 
            y='kebutuhan', 
            color=filtered_df_with_clusters['cluster'].astype(str),
            title='Hasil Clustering: Jumlah vs Kebutuhan',
            labels={
                'jumlah': 'Jumlah Barang', 
                'kebutuhan': 'Kebutuhan Barang', 
                'color': 'Cluster'
            },
            hover_data=['item_barang', 'nama_pasar']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        insights = cluster_analyzer.generate_cluster_insights(filtered_df_with_clusters)
        
        st.subheader("🔍 Insight Clustering")
        for insight in insights:
            with st.expander(f"Cluster {insight['cluster_id']} ({insight['total_items']} items)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rata-rata Jumlah", f"{insight['avg_jumlah']:,}")
                    st.metric("Rata-rata Kebutuhan", f"{insight['avg_kebutuhan']:,}")
                
                with col2:
                    st.write("**Top 3 Barang:**")
                    for i, item in enumerate(insight['top_items'], 1):
                        st.write(f"{i}. {item}")
                
                interpretation_color = "green" if insight['interpretation'] == 'tinggi' else "blue"
                st.markdown(f"**Interpretasi:** <span style='color:{interpretation_color}'>Pola kebutuhan {insight['interpretation']}</span>", 
                           unsafe_allow_html=True)
        
        # PDF Export
        st.subheader("📄 Export Laporan")
        if st.button("📥 Generate PDF Report", type="primary"):
            with st.spinner("Membuat laporan PDF..."):
                pdf_output = PDFGenerator.generate_clustering_report(filters, insights)
                
                st.download_button(
                    label="📄 Download Laporan PDF",
                    data=pdf_output,
                    file_name=f"laporan_clustering_pasar_sumedang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    app = MarketAnalysisApp()
    app.run()