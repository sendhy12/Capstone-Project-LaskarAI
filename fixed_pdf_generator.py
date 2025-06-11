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
    PAGE_TITLE = "Smart Market Analytics"
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
    """Handles PDF report generation with professional formatting"""
    
    class ReportPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=20)
            
        def header(self):
            # Only add header if a page exists
            if self.page_no() > 0:
                # Background color for header
                self.set_fill_color(240, 248, 255)  # Light blue background
                self.rect(10, 10, 190, 45, 'F')
                
                # Logo placeholder (you can add actual logo later)
                self.set_fill_color(70, 130, 180)  # Steel blue for logo placeholder
                self.rect(15, 15, 20, 20, 'F')
                self.set_font('Arial', 'B', 8)
                self.set_text_color(255, 255, 255)
                self.text(20, 27, 'LOGO')
                
                # Reset text color
                self.set_text_color(0, 0, 0)
                
                # Government header
                self.set_font('Arial', 'B', 14)
                self.set_xy(40, 15)
                self.cell(0, 6, "PEMERINTAH KABUPATEN SUMEDANG", ln=True, align='L')
                
                self.set_font('Arial', 'B', 12)
                self.set_x(40)
                self.cell(0, 5, "DINAS KOPERASI, UKM, PERDAGANGAN DAN PERINDUSTRIAN", ln=True, align='L')
                
                # Contact information
                self.set_font('Arial', '', 9)
                self.set_x(40)
                self.cell(0, 4, "Jl. Raya Sumedang No.123, Sumedang 45300, Jawa Barat", ln=True, align='L')
                self.set_x(40)
                self.cell(0, 4, "Telp: (0261) 123456 | Fax: (0261) 123457", ln=True, align='L')
                self.set_x(40)
                self.cell(0, 4, "Email: dinaskopdag@sumedang.go.id | Website: www.sumedang.go.id", ln=True, align='L')
                
                # Decorative line
                self.set_line_width(0.5)
                self.set_draw_color(70, 130, 180)
                self.line(10, 58, 200, 58)
                self.set_line_width(0.2)
                self.line(10, 60, 200, 60)
                
                self.ln(20)
        
        def footer(self):
            # Background for footer
            self.set_fill_color(240, 248, 255)
            self.rect(10, -25, 190, 15, 'F')
            
            self.set_y(-20)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            
            # Left side - document info
            self.set_x(15)
            self.cell(0, 5, f"Dokumen dibuat secara otomatis oleh Smart Market Analytics", ln=True, align='L')
            
            # Right side - page and date
            self.set_y(-15)
            self.set_x(15)
            self.cell(0, 5, f"Halaman {self.page_no()} | Dicetak pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", 
                     ln=True, align='R')
            
            # Reset text color
            self.set_text_color(0, 0, 0)
        
        def add_title_page(self, report_title: str, filters: Dict):
            """Add a professional title page"""
            # First, add a page
            self.add_page()
            
            # Document title
            self.ln(10)
            self.set_font('Arial', 'B', 18)
            self.set_text_color(70, 130, 180)
            self.cell(0, 12, report_title.upper(), ln=True, align='C')
            
            # Subtitle
            self.set_font('Arial', '', 12)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Analisis Data Pasar Tradisional Kabupaten Sumedang", ln=True, align='C')
            
            self.ln(15)
            
            # Information box
            self.set_fill_color(248, 249, 250)
            self.set_draw_color(200, 200, 200)
            self.rect(20, self.get_y(), 170, 60, 'DF')
            
            self.set_text_color(0, 0, 0)
            self.ln(5)
            
            # Filter information in a neat table format
            self.set_font('Arial', 'B', 11)
            self.cell(0, 8, "PARAMETER ANALISIS", ln=True, align='C')
            self.ln(3)
            
            self.set_font('Arial', '', 10)
            y_start = self.get_y()
            
            filter_labels = {
                'nama_pasar': 'Pasar yang Dianalisis',
                'tahun': 'Periode Tahun',
                'bulan': 'Periode Bulan',
                'item_barang': 'Jenis Barang'
            }
            
            for key, value in filters.items():
                if value and key in filter_labels:
                    self.set_x(25)
                    self.set_font('Arial', 'B', 10)
                    self.cell(50, 6, f"{filter_labels[key]}:", ln=False, align='L')
                    
                    self.set_font('Arial', '', 10)
                    # Handle long lists by truncating
                    value_str = ', '.join(map(str, value))
                    if len(value_str) > 80:
                        value_str = value_str[:77] + "..."
                    self.cell(0, 6, value_str, ln=True, align='L')
            
            # Generation info
            self.ln(10)
            self.set_font('Arial', 'I', 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", 
                     ln=True, align='C')
            self.cell(0, 6, f"Petugas: Sistem Otomatis Analisis Data", ln=True, align='C')
            
            # Add new page for content
            self.add_page()
        
        def add_section_header(self, title: str, subtitle: str = ""):
            """Add a styled section header"""
            self.ln(5)
            
            # Background for section header
            self.set_fill_color(70, 130, 180)
            self.rect(10, self.get_y(), 190, 12, 'F')
            
            # Section title
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 12)
            self.cell(0, 12, f"  {title}", ln=True, align='L')
            
            if subtitle:
                self.set_text_color(100, 100, 100)
                self.set_font('Arial', 'I', 10)
                self.cell(0, 6, f"  {subtitle}", ln=True, align='L')
            
            self.set_text_color(0, 0, 0)
            self.ln(3)
        
        def add_cluster_box(self, cluster_info: Dict):
            """Add a styled box for cluster information"""
            # Calculate box height based on content
            box_height = 45
            
            # Check if new page is needed
            if self.get_y() + box_height > 270:
                self.add_page()
            
            y_start = self.get_y()
            
            # Cluster box background
            colors = [
                (230, 247, 255),  # Light blue
                (255, 245, 230),  # Light orange  
                (240, 255, 240),  # Light green
                (255, 240, 245),  # Light pink
                (245, 245, 255),  # Light purple
            ]
            color = colors[cluster_info['cluster_id'] % len(colors)]
            
            self.set_fill_color(*color)
            self.set_draw_color(200, 200, 200)
            self.rect(15, y_start, 180, box_height, 'DF')
            
            # Cluster header
            self.set_xy(20, y_start + 3)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(70, 130, 180)
            self.cell(0, 6, f"CLUSTER {cluster_info['cluster_id']}", ln=True, align='L')
            
            # Statistics in two columns
            self.set_font('Arial', '', 10)
            self.set_text_color(0, 0, 0)
            
            # Left column
            self.set_xy(25, y_start + 12)
            self.cell(80, 5, f"Rata-rata Jumlah Stok", ln=False, align='L')
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, f": {cluster_info['avg_jumlah']:,} unit", ln=True, align='L')
            
            self.set_xy(25, y_start + 18)
            self.set_font('Arial', '', 10)
            self.cell(80, 5, f"Rata-rata Kebutuhan", ln=False, align='L')
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, f": {cluster_info['avg_kebutuhan']:,} unit", ln=True, align='L')
            
            self.set_xy(25, y_start + 24)
            self.set_font('Arial', '', 10)
            self.cell(80, 5, f"Total Item dalam Cluster", ln=False, align='L')
            self.set_font('Arial', 'B', 10)
            self.cell(0, 5, f": {cluster_info['total_items']:,} item", ln=True, align='L')
            
            # Top items
            self.set_xy(25, y_start + 32)
            self.set_font('Arial', '', 10)
            self.cell(80, 5, f"Barang Teratas", ln=False, align='L')
            self.set_font('Arial', 'B', 10)
            top_items_str = ', '.join(cluster_info['top_items'][:3])
            if len(top_items_str) > 60:
                top_items_str = top_items_str[:57] + "..."
            self.cell(0, 5, f": {top_items_str}", ln=True, align='L')
            
            # Interpretation badge
            interpretation_color = (34, 139, 34) if cluster_info['interpretation'] == 'tinggi' else (30, 144, 255)
            self.set_fill_color(*interpretation_color)
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 9)
            
            badge_text = f"KEBUTUHAN {cluster_info['interpretation'].upper()}"
            badge_width = self.get_string_width(badge_text) + 8
            
            self.set_xy(190 - badge_width, y_start + 5)
            self.cell(badge_width, 8, badge_text, ln=False, align='C', fill=True)
            
            self.set_text_color(0, 0, 0)
            self.set_y(y_start + box_height + 5)
        
        def add_summary_table(self, insights: List[Dict]):
            """Add a summary table of all clusters"""
            self.add_section_header("RINGKASAN ANALISIS CLUSTERING", 
                                  "Tabel perbandingan semua cluster yang terbentuk")
            
            # Table header
            self.set_fill_color(70, 130, 180)
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 9)
            
            col_widths = [25, 35, 35, 25, 70]
            headers = ['Cluster', 'Avg Jumlah', 'Avg Kebutuhan', 'Total Item', 'Karakteristik']
            
            x_start = 15
            self.set_x(x_start)
            
            for i, (header, width) in enumerate(zip(headers, col_widths)):
                self.cell(width, 8, header, 1, 0, 'C', True)
            self.ln()
            
            # Table rows
            self.set_fill_color(248, 249, 250)
            self.set_text_color(0, 0, 0)
            self.set_font('Arial', '', 9)
            
            for i, insight in enumerate(insights):
                fill = i % 2 == 0
                self.set_x(x_start)
                
                # Cluster ID
                self.cell(col_widths[0], 7, str(insight['cluster_id']), 1, 0, 'C', fill)
                
                # Avg Jumlah
                self.cell(col_widths[1], 7, f"{insight['avg_jumlah']:,}", 1, 0, 'R', fill)
                
                # Avg Kebutuhan  
                self.cell(col_widths[2], 7, f"{insight['avg_kebutuhan']:,}", 1, 0, 'R', fill)
                
                # Total Items
                self.cell(col_widths[3], 7, f"{insight['total_items']:,}", 1, 0, 'R', fill)
                
                # Karakteristik
                karakteristik = f"Kebutuhan {insight['interpretation']}"
                self.cell(col_widths[4], 7, karakteristik, 1, 0, 'L', fill)
                
                self.ln()
        
        def add_recommendations(self, insights: List[Dict]):
            """Add recommendations based on clustering results"""
            self.add_section_header("REKOMENDASI STRATEGIS", 
                                  "Saran tindakan berdasarkan hasil analisis clustering")
            
            # General recommendations
            recommendations = [
                "Optimalkan stok barang pada cluster dengan kebutuhan tinggi untuk mencegah kehabisan stok",
                "Evaluasi strategi procurement untuk cluster dengan pola kebutuhan rendah",  
                "Lakukan monitoring berkala terhadap perubahan pola kebutuhan di setiap cluster",
                "Pertimbangkan diversifikasi produk pada pasar dengan cluster kebutuhan stabil",
                "Koordinasikan distribusi antar pasar untuk efisiensi supply chain"
            ]
            
            self.set_font('Arial', '', 10)
            for i, rec in enumerate(recommendations, 1):
                self.cell(10, 6, f"{i}.", ln=False, align='L')
                # Multi-line text handling
                lines = self.split_text(rec, 170)
                for j, line in enumerate(lines):
                    if j > 0:
                        self.cell(10, 6, "", ln=False, align='L')  # Indent
                    self.cell(0, 6, line, ln=True, align='L')
                self.ln(2)
        
        def split_text(self, text: str, max_width: int) -> List[str]:
            """Split text into multiple lines to fit within max_width"""
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                if self.get_string_width(test_line) <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            return lines
    
    @staticmethod
    def generate_clustering_report(filters: Dict, insights: List[Dict], stats: Dict = None) -> BytesIO:
        """Generate comprehensive PDF report for clustering analysis"""
        pdf = PDFGenerator.ReportPDF()
        
        # Title page - add_page() is called inside add_title_page()
        pdf.add_title_page("Laporan Analisis Clustering K-Means", filters)
        
        # Executive Summary (if stats provided)
        if stats:
            pdf.add_section_header("RINGKASAN EKSEKUTIF", 
                                 "Gambaran umum hasil analisis data pasar")
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f"Total data yang dianalisis: {stats.get('total_records', 0):,} record", ln=True)
            pdf.cell(0, 6, f"Korelasi jumlah vs kebutuhan: {stats.get('correlation', 0):.3f}", ln=True)
            pdf.cell(0, 6, f"Rata-rata stok keseluruhan: {stats.get('avg_jumlah', 0):.0f} unit", ln=True)
            pdf.cell(0, 6, f"Rata-rata kebutuhan keseluruhan: {stats.get('avg_kebutuhan', 0):.0f} unit", ln=True)
            pdf.ln(5)
        
        # Detailed cluster analysis
        pdf.add_section_header("HASIL ANALISIS CLUSTERING", 
                             f"Pembagian data menjadi {len(insights)} cluster berdasarkan pola jumlah dan kebutuhan")
        
        for insight in insights:
            pdf.add_cluster_box(insight)
        
        # Summary table
        pdf.add_summary_table(insights)
        
        # Recommendations
        pdf.add_recommendations(insights)
        
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
            with st.spinner("Membuat laporan PDF profesional..."):
                stats = self.data_processor.get_statistics(filtered_df_with_clusters)
                pdf_output = PDFGenerator.generate_clustering_report(filters, insights, stats)
                
                st.success("✅ Laporan PDF berhasil dibuat!")
                st.download_button(
                    label="📄 Download Laporan PDF",
                    data=pdf_output,
                    file_name=f"laporan_clustering_pasar_sumedang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Klik untuk mengunduh laporan dalam format PDF"
                )

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    app = MarketAnalysisApp()
    app.run()