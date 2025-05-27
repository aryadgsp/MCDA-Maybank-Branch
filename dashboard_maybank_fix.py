import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Branch Analysis Dashboard - MCDA Enhanced",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Service Specification Weights for Branch Analysis
SERVICE_WEIGHTS = {
    'industrial_business': 0.15,      # Pabrik/Industri - KUR, Cash Management
    'residential_consumer': 0.15,     # Pemukiman - KPR, Tabungan, Consumer
    'youth_digital': 0.12,           # Mall/Cafe/Restaurant - Digital Banking Focus
    'corporate_office': 0.12,        # Perkantoran - Corporate Banking, Payroll
    'wealthy_premium': 0.10,         # Area elit - Wealth Management, Private Banking
    'education_student': 0.08,       # Sekolah/Universitas - Tabungan Pelajar
    'healthcare_insurance': 0.08,    # Rumah Sakit - Asuransi Kesehatan
    'competitor_threat': 0.10,       # Kompetitor - Defensive Strategy
    'accessibility': 0.10            # Traffic & Transport - Service Hours & Method
}

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .mcda-card {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        padding: 1.8rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.3);
    }
    
    .traffic-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.8rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(116, 185, 255, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(253, 121, 168, 0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 184, 148, 0.3);
    }
    
    .priority-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .priority-medium {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .priority-low {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border-left: 5px solid #fdcb6e;
    }
    
    .branch-detail-card {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(45, 52, 54, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'poi_data' not in st.session_state:
    st.session_state.poi_data = None
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = None
if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = None
if 'service_results' not in st.session_state:
    st.session_state.service_results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1>🏦 Branch Service Specification Analysis</h1>
    <p style="font-size: 1.2em;">Analisis Spesifikasi Layanan untuk Optimalisasi Efisiensi Cabang Bank</p>
    <p style="opacity: 0.9;">📍 POI Analysis • 🎯 Service Specification • 💰 Cost Optimization • 📱 Digital Transformation</p>
</div>
""", unsafe_allow_html=True)

# Branch coordinates
branches = {
    "KCI ROXY MAS": (-6.1664558, 106.8038417),
    "KCP GREEN VILLE": (-6.1718394, 106.7768208),
    "KCP TAMAN ANGGREK": (-6.178956, 106.792761),
    "KCP TOMANG": (-6.1786007, 106.7983146),
    "KCP CENTRAL PARK": (-6.1757957, 106.789946),
    "KCP JEMBATAN LIMA": (-6.1538156, 106.8077951),
    "KCP DAAN MOGOT": (-6.166103, 106.7861734)
}

@st.cache_data
def load_poi_data():
    """Load POI data from uploaded file"""
    try:
        # Expected columns: name, enhanced_category, branch, lat, lon, distance_to_branch_m
        poi_data = pd.read_csv('poi_data.csv')  # You'll need to upload this file
        
        # Validate required columns
        required_cols = ['name', 'enhanced_category', 'branch', 'lat', 'lon', 'distance_to_branch_m']
        missing_cols = [col for col in required_cols if col not in poi_data.columns]
        
        if missing_cols:
            st.error(f"Missing columns in POI data: {missing_cols}")
            return None
            
        return poi_data
        
    except FileNotFoundError:
        st.error("POI data file not found. Please upload 'poi_data.csv'")
        return None
    except Exception as e:
        st.error(f"Error loading POI data: {str(e)}")
        return None

@st.cache_data
def load_traffic_data():
    """Load traffic data from uploaded file"""
    try:
        # Expected columns: nama_jalan, jenis_jalan, kategori_traffic, lat, lon, branch
        traffic_data = pd.read_csv('jalan_semua_cabang.csv')
        
        # Validate required columns
        required_cols = ['nama_jalan', 'jenis_jalan', 'kategori_traffic', 'lat', 'lon', 'branch']
        missing_cols = [col for col in required_cols if col not in traffic_data.columns]
        
        if missing_cols:
            st.error(f"Missing columns in traffic data: {missing_cols}")
            return None
            
        return traffic_data
        
    except FileNotFoundError:
        st.error("Traffic data file not found. Please upload 'jalan_semua_cabang.csv'")
        return None
    except Exception as e:
        st.error(f"Error loading traffic data: {str(e)}")
        return None

def analyze_traffic_data(traffic_df):
    """Analyze traffic data and create summary"""
    if traffic_df is None or traffic_df.empty:
        return pd.DataFrame()
    
    # Traffic scoring weights
    traffic_weights = {
        'padat': 5,     # Heavy traffic (motorway, trunk, primary)
        'sedang': 3,    # Medium traffic (secondary, tertiary)
        'lancar': 1,    # Light traffic (residential, etc)
        'tidak diketahui': 0
    }
    
    # Calculate traffic scores by branch
    traffic_summary = []
    
    for branch in branches.keys():
        branch_roads = traffic_df[traffic_df['branch'] == branch]
        
        if branch_roads.empty:
            # Default values for branches with no road data
            traffic_summary.append({
                'cabang': branch,
                'jumlah_jalan': 0,
                'jumlah_jalan_padat': 0,
                'jumlah_jalan_sedang': 0,
                'jumlah_jalan_lancar': 0,
                'skor_potensi_traffic': 10,  # Low default score
                'kategori_traffic': 'Rendah',
                'traffic_level': 'Low'
            })
            continue
        
        # Count roads by traffic category
        traffic_counts = branch_roads['kategori_traffic'].value_counts()
        
        jumlah_padat = traffic_counts.get('padat', 0)
        jumlah_sedang = traffic_counts.get('sedang', 0)
        jumlah_lancar = traffic_counts.get('lancar', 0)
        jumlah_total = len(branch_roads)
        
        # Calculate weighted traffic score
        weighted_score = (
            jumlah_padat * traffic_weights['padat'] +
            jumlah_sedang * traffic_weights['sedang'] +
            jumlah_lancar * traffic_weights['lancar']
        )
        
        # Normalize score (0-100 scale)
        if jumlah_total > 0:
            base_score = weighted_score / jumlah_total * 10  # Scale to 0-50
            density_bonus = min(jumlah_total / 20 * 30, 30)  # Density bonus up to 30
            total_score = min(base_score + density_bonus, 100)
        else:
            total_score = 0
        
        # Categorize traffic level
        if total_score >= 60:
            kategori = 'Tinggi'
            level = 'High'
        elif total_score >= 30:
            kategori = 'Sedang'
            level = 'Medium'
        else:
            kategori = 'Rendah'
            level = 'Low'
        
        traffic_summary.append({
            'cabang': branch,
            'jumlah_jalan': jumlah_total,
            'jumlah_jalan_padat': jumlah_padat,
            'jumlah_jalan_sedang': jumlah_sedang,
            'jumlah_jalan_lancar': jumlah_lancar,
            'skor_potensi_traffic': round(total_score, 2),
            'kategori_traffic': kategori,
            'traffic_level': level
        })
    
    return pd.DataFrame(traffic_summary)

def analyze_poi_for_service_specification(df):
    """Analyze POI data for Branch Service Specification"""
    if df is None or df.empty:
        return None, None
    
    # Service-oriented categories with detailed keywords
    service_categories = {
        'Industrial_Business': {
            'keywords': [
                'industrial', 'factory', 'plant', 'warehouse', 'works', 'workshop', 'manufacture',
                'pabrik', 'industri', 'gudang', 'bengkel', 'manufacturing', 'industrial_area',
                'logistics', 'distribution', 'production', 'assembly', 'processing'
            ],
            'categories': ['Industrial', 'Commercial'],
            'services': ['Kredit Usaha Rakyat (KUR)', 'Cash Management', 'Trade Finance', 'Working Capital']
        },
        
        'Residential_Consumer': {
            'keywords': [
                'residential', 'house', 'apartments', 'terrace', 'detached', 'semidetached',
                'rumah', 'apartemen', 'perumahan', 'housing', 'kondominium', 'cluster',
                'villa', 'mansion', 'residence', 'real_estate', 'komplek_perumahan'
            ],
            'categories': ['Residential'],
            'services': ['KPR', 'Tabungan Keluarga', 'Asuransi Jiwa', 'Consumer Loan']
        },
        
        'Youth_Digital': {
            'keywords': [
                'mall', 'shopping_mall', 'plaza', 'cafe', 'coffee_shop', 'restaurant', 
                'fast_food', 'bar', 'pub', 'cinema', 'entertainment', 'game', 'karaoke',
                'gym', 'fitness', 'salon', 'beauty', 'fashion', 'clothing', 'boutique',
                'coworking', 'internet_cafe', 'warnet'
            ],
            'categories': ['F&B', 'Retail', 'Recreation'],
            'services': ['Digital Banking', 'E-wallet', 'Mobile Payment', 'Student Account']
        },
        
        'Corporate_Office': {
            'keywords': [
                'office', 'business_park', 'commercial', 'trading', 'wholesale', 'corporate',
                'gedung_perkantoran', 'ruko', 'kawasan_bisnis', 'office_building', 'headquarters',
                'company', 'firm', 'consultant', 'law', 'accounting', 'finance'
            ],
            'categories': ['Office'],
            'services': ['Corporate Banking', 'Payroll Services', 'Business Loan', 'Trade Services']
        },
        
        'Wealthy_Premium': {
            'keywords': [
                'luxury', 'premium', 'exclusive', 'private', 'villa', 'mansion', 'resort',
                'golf', 'country_club', 'yacht', 'marina', 'spa', 'wellness', 'high_end',
                'expensive', 'elite', 'upscale', 'five_star', 'mewah', 'eksklusif'
            ],
            'categories': ['Tourism'],
            'services': ['Wealth Management', 'Private Banking', 'Investment Advisory', 'Premium Credit Card']
        },
        
        'Education_Student': {
            'keywords': [
                'school', 'college', 'university', 'kindergarten', 'language_school',
                'sekolah', 'kampus', 'universitas', 'institut', 'akademi', 'bimbel',
                'student', 'education', 'learning', 'training', 'course', 'academy'
            ],
            'categories': ['Education'],
            'services': ['Tabungan Pelajar', 'Beasiswa Program', 'Student Loan', 'Educational Insurance']
        },
        
        'Healthcare_Insurance': {
            'keywords': [
                'hospital', 'clinic', 'doctors', 'pharmacy', 'dentist', 'laboratory',
                'rumah_sakit', 'klinik', 'apotek', 'dokter', 'puskesmas', 'medical',
                'health', 'medicine', 'treatment', 'therapy', 'rehabilitation'
            ],
            'categories': ['Healthcare'],
            'services': ['Asuransi Kesehatan', 'Medical Financing', 'Healthcare Investment', 'Insurance Claims']
        },
        
        'Competitor_Banks': {
            'keywords': [
                'bca', 'mandiri', 'bni', 'bri', 'cimb', 'ocbc', 'panin', 'permata',
                'danamon', 'uob', 'hsbc', 'btn', 'mega', 'bukopin', 'sinarmas',
                'commonwealth', 'standard_chartered', 'citibank', 'anz', 'btpn'
            ],
            'categories': ['Bank_Competitor'],
            'exclude_keywords': ['maybank'],
            'services': ['Competitive Strategy', 'Differentiation Focus', 'Defensive Positioning', 'Market Share Protection']
        },
        
        'Transportation_Hub': {
            'keywords': [
                'station', 'bus_station', 'taxi', 'terminal', 'halte', 'bandara', 'airport',
                'kereta', 'stasiun', 'busway', 'transjakarta', 'mrt', 'lrt', 'commuter',
                'transport', 'transit', 'mobility'
            ],
            'categories': ['Transport'],
            'services': ['Extended Hours', 'Quick Service', 'Mobile Banking', 'Travel Services']
        }
    }
    
    analysis_results = {}
    service_recommendations = {}
    
    for category_name, category_config in service_categories.items():
        try:
            mask = pd.Series(False, index=df.index)
            
            # Apply category-based filtering
            categories = category_config.get('categories', [])
            if categories:
                cat_mask = df['enhanced_category'].isin(categories)
                mask |= cat_mask
            
            # Apply keyword-based filtering
            keywords = category_config.get('keywords', [])
            if keywords:
                keyword_pattern = '|'.join(keywords)
                name_mask = df['name'].str.lower().str.contains(
                    keyword_pattern, na=False, case=False
                )
                mask |= name_mask
            
            # Apply exclusions (important for competitor banks)
            exclude_keywords = category_config.get('exclude_keywords', [])
            if exclude_keywords:
                exclude_pattern = '|'.join(exclude_keywords)
                exclude_mask = df['name'].str.lower().str.contains(
                    exclude_pattern, na=False, case=False
                )
                mask &= ~exclude_mask
            
            category_data = df[mask].copy()
            
            if not category_data.empty:
                # Remove duplicates and count by branch
                category_data = category_data.drop_duplicates(
                    subset=['lat', 'lon', 'name'], keep='first'
                )
                poi_counts = category_data.groupby('branch').size()
                analysis_results[category_name] = poi_counts
                
                # Store service recommendations
                service_recommendations[category_name] = category_config.get('services', [])
            else:
                # Initialize with zeros for all branches
                branch_list = df['branch'].unique()
                analysis_results[category_name] = pd.Series(0, index=branch_list)
                service_recommendations[category_name] = category_config.get('services', [])
                
        except Exception as e:
            st.warning(f"Error analyzing category {category_name}: {str(e)}")
            branch_list = df['branch'].unique()
            analysis_results[category_name] = pd.Series(0, index=branch_list)
            service_recommendations[category_name] = category_config.get('services', [])
    
    if analysis_results:
        result_df = pd.DataFrame(analysis_results).fillna(0)
        all_branches = list(branches.keys())
        
        # Ensure all branches are included
        for branch in all_branches:
            if branch not in result_df.index:
                result_df.loc[branch] = 0
        
        return result_df.loc[all_branches], service_recommendations
    
    return None, None

def calculate_distance_matrix():
    """Calculate comprehensive distance matrix between branches"""
    branch_names = list(branches.keys())
    dist_matrix = pd.DataFrame(index=branch_names, columns=branch_names, dtype=float)
    
    for b1 in branch_names:
        for b2 in branch_names:
            if b1 != b2:
                coord1 = branches[b1]
                coord2 = branches[b2]
                dist_matrix.loc[b1, b2] = geodesic(coord1, coord2).km
            else:
                dist_matrix.loc[b1, b2] = 0.0
    
    return dist_matrix.round(2)

def calculate_service_specification_scores(poi_analysis, traffic_summary, distance_matrix, weights=None):
    """Calculate Service Specification scores using weighted criteria"""
    
    if weights is None:
        weights = SERVICE_WEIGHTS
    
    # Initialize results dataframe
    service_results = pd.DataFrame(index=poi_analysis.index)
    
    # Normalize function (0-1 scale)
    def normalize_series(series, reverse=False):
        if series.max() == series.min():
            return pd.Series(0.5, index=series.index)
        
        normalized = (series - series.min()) / (series.max() - series.min())
        if reverse:
            normalized = 1 - normalized
        return normalized
    
    # 1. Industrial Business Score (KUR, Cash Management)
    industrial_score = normalize_series(poi_analysis['Industrial_Business'])
    service_results['industrial_business_score'] = industrial_score
    
    # 2. Residential Consumer Score (KPR, Consumer Banking)
    residential_score = normalize_series(poi_analysis['Residential_Consumer'])
    service_results['residential_consumer_score'] = residential_score
    
    # 3. Youth Digital Score (Digital Banking Focus)
    youth_score = normalize_series(poi_analysis['Youth_Digital'])
    service_results['youth_digital_score'] = youth_score
    
    # 4. Corporate Office Score (Corporate Banking)
    corporate_score = normalize_series(poi_analysis['Corporate_Office'])
    service_results['corporate_office_score'] = corporate_score
    
    # 5. Wealthy Premium Score (Private Banking, Wealth Management)
    wealthy_score = normalize_series(poi_analysis['Wealthy_Premium'])
    service_results['wealthy_premium_score'] = wealthy_score
    
    # 6. Education Student Score (Student Services)
    education_score = normalize_series(poi_analysis['Education_Student'])
    service_results['education_student_score'] = education_score
    
    # 7. Healthcare Insurance Score (Health Insurance, Medical Financing)
    healthcare_score = normalize_series(poi_analysis['Healthcare_Insurance'])
    service_results['healthcare_insurance_score'] = healthcare_score
    
    # 8. Competitor Threat Score (higher competitors = more defensive strategy needed)
    competitor_score = normalize_series(poi_analysis['Competitor_Banks'])
    service_results['competitor_threat_score'] = competitor_score
    
    # 9. Accessibility Score (Extended hours, Drive-thru needs)
    if not traffic_summary.empty:
        traffic_scores = traffic_summary.set_index('cabang')['skor_potensi_traffic']
        traffic_score_norm = normalize_series(traffic_scores)
        transport_score = normalize_series(poi_analysis['Transportation_Hub'])
        accessibility_score = (traffic_score_norm.reindex(service_results.index).fillna(0) + transport_score) / 2
    else:
        accessibility_score = normalize_series(poi_analysis['Transportation_Hub'])
    service_results['accessibility_score'] = accessibility_score
    
    # Calculate weighted Service Specification score
    service_results['weighted_score'] = (
        service_results['industrial_business_score'] * weights['industrial_business'] +
        service_results['residential_consumer_score'] * weights['residential_consumer'] +
        service_results['youth_digital_score'] * weights['youth_digital'] +
        service_results['corporate_office_score'] * weights['corporate_office'] +
        service_results['wealthy_premium_score'] * weights['wealthy_premium'] +
        service_results['education_student_score'] * weights['education_student'] +
        service_results['healthcare_insurance_score'] * weights['healthcare_insurance'] +
        service_results['competitor_threat_score'] * weights['competitor_threat'] +
        service_results['accessibility_score'] * weights['accessibility']
    )
    
    # Scale to 0-100
    service_results['final_score'] = service_results['weighted_score'] * 100
    
    # Add rankings
    service_results['rank'] = service_results['final_score'].rank(method='dense', ascending=False).astype(int)
    
    # Determine dominant service type for each branch
    service_columns = [
        'industrial_business_score', 'residential_consumer_score', 'youth_digital_score',
        'corporate_office_score', 'wealthy_premium_score', 'education_student_score',
        'healthcare_insurance_score'
    ]
    
    # service_results['dominant_service'] = service_results[service_columns].idxmax(axis=1).str.replace('_score', '')
    service_results['dominant_service'] = service_results[service_columns].apply(
    lambda row: row.nlargest(2).index.str.replace('_score', '').tolist(),
    axis=1
    )
    
    # Service priority classification
    def classify_service_priority(row):
        max_score = max([row[col] for col in service_columns])
        if max_score >= 0.7:
            return 'Specialized'  # Fokus layanan khusus
        elif max_score >= 0.4:
            return 'Targeted'     # Layanan target dengan dukungan umum
        else:
            return 'General'      # Layanan umum
    
    service_results['service_priority'] = service_results.apply(classify_service_priority, axis=1)
    
    return service_results

def create_poi_map(poi_df, selected_branch=None):
    """Create interactive map showing POI distribution"""
    
    # Center map on Jakarta
    center_lat = -6.1751
    center_lon = 106.8650
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Color mapping for categories
    category_colors = {
        'Industrial': '#e74c3c',
        'Residential': '#3498db', 
        'F&B': '#f39c12',
        'Office': '#9b59b6',
        'Healthcare': '#e67e22',
        'Education': '#27ae60',
        'Bank_Competitor': '#c0392b',
        'Transport': '#16a085',
        'Retail': '#f1c40f',
        'Recreation': '#8e44ad',
        'Tourism': '#2980b9',
        'Commercial': '#34495e',
        'Other': '#95a5a6'
    }
    
    # Add branch locations
    for branch_name, (lat, lon) in branches.items():
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{branch_name}</b><br>Maybank Branch",
            icon=folium.Icon(color='red', icon='info-sign', prefix='glyphicon')
        ).add_to(m)
    
    # Filter POI data if specific branch selected
    if selected_branch and selected_branch != "All Branches":
        poi_filtered = poi_df[poi_df['branch'] == selected_branch]
    else:
        poi_filtered = poi_df
    
    # Add POI points
    for idx, row in poi_filtered.iterrows():
        try:
            category = row['enhanced_category']
            color = category_colors.get(category, '#95a5a6')
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                popup=f"<b>{row['name']}</b><br>Category: {category}<br>Branch: {row['branch']}",
                color=color,
                fill=True,
                weight=2
            ).add_to(m)
        except:
            continue
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>POI Categories</b><br>
    '''
    
    for category, color in category_colors.items():
        legend_html += f'<i class="fa fa-circle" style="color:{color}"></i> {category}<br>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_branch_breakdown(poi_analysis, traffic_summary, service_results, selected_branch):
    """Create detailed breakdown for selected branch"""
    
    if selected_branch not in poi_analysis.index:
        st.error(f"Branch {selected_branch} not found in analysis data")
        return
    
    st.markdown(f"""
    <div class="branch-detail-card">
        <h2>📊 {selected_branch} - Detailed Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Branch coordinates
    branch_lat, branch_lon = branches[selected_branch]
    
    # Get branch data
    branch_poi = poi_analysis.loc[selected_branch]
    branch_service = service_results.loc[selected_branch]
    branch_traffic = traffic_summary[traffic_summary['cabang'] == selected_branch].iloc[0] if not traffic_summary.empty else None
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_poi = int(branch_poi.sum())
        st.markdown(f"""
        <div class="metric-card">
            <h3>📍 Total POI</h3>
            <h2>{total_poi}</h2>
            <p>Points of Interest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        service_score = branch_service['final_score']
        rank = int(branch_service['rank'])
        st.markdown(f"""
        <div class="metric-card">
            <h3>⭐ Service Score</h3>
            <h2>{service_score:.1f}</h2>
            <p>Rank #{rank}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # dominant_service = branch_service['dominant_service'].replace('_', ' ').title()
        dominant_service = ', '.join(s.replace('_', ' ') for s in branch_service['dominant_service'])
        priority = branch_service['service_priority']
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Focus Area</h3>
            <h2 style="font-size: 1.2em;">{dominant_service}</h2>
            <p>{priority} Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if branch_traffic is not None:
            traffic_score = branch_traffic['skor_potensi_traffic']
            traffic_cat = branch_traffic['kategori_traffic']
            st.markdown(f"""
            <div class="traffic-card">
                <h3>🚦 Traffic Level</h3>
                <h2>{traffic_score:.1f}</h2>
                <p>{traffic_cat}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="traffic-card">
                <h3>🚦 Traffic Level</h3>
                <h2>N/A</h2>
                <p>No Data</p>
            </div>
            """, unsafe_allow_html=True)
    
    # POI Category Analysis
    st.markdown("#### 📊 POI Category Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # POI category bar chart
        poi_categories = branch_poi[branch_poi > 0].sort_values(ascending=True)
        
        if not poi_categories.empty:
            fig_poi = px.bar(
                x=poi_categories.values,
                y=[cat.replace('_', ' ').title() for cat in poi_categories.index],
                orientation='h',
                title=f"POI Categories - {selected_branch}",
                color=poi_categories.values,
                color_continuous_scale="viridis"
            )
            fig_poi.update_layout(
                height=400,
                showlegend=False,
                xaxis_title="Number of POIs",
                yaxis_title="Category"
            )
            st.plotly_chart(fig_poi, use_container_width=True)
        else:
            st.info("No POI data available for this branch")
    
    with col2:
        # Top 3 categories
        st.markdown("##### 🏆 Top 3 Categories")
        top_3 = branch_poi.nlargest(3)
        
        for i, (category, count) in enumerate(top_3.items()):
            if count > 0:
                rank_emoji = ["🥇", "🥈", "🥉"][i]
                category_name = category.replace('_', ' ').title()
                st.markdown(f"""
                <div class="insight-card" style="margin: 0.5rem 0; padding: 1rem;">
                    <h4>{rank_emoji} {category_name}</h4>
                    <h3>{int(count)} POIs</h3>
                </div>
                """, unsafe_allow_html=True)
    
    # Service Scores Breakdown
    st.markdown("#### 🎯 Service Specification Scores")
    
    service_categories = [
        ('industrial_business_score', 'Industrial/Business', '🏭'),
        ('residential_consumer_score', 'Residential/Consumer', '🏠'),
        ('youth_digital_score', 'Youth/Digital', '📱'),
        ('corporate_office_score', 'Corporate/Office', '🏢'),
        ('wealthy_premium_score', 'Wealthy/Premium', '💎'),
        ('education_student_score', 'Education/Student', '🎓'),
        ('healthcare_insurance_score', 'Healthcare/Insurance', '🏥'),
        ('competitor_threat_score', 'Competitor Threat', '⚔️'),
        ('accessibility_score', 'Accessibility', '🚗')
    ]
    
    scores_data = []
    for score_col, label, emoji in service_categories:
        score_value = branch_service[score_col]
        scores_data.append({
            'Category': f"{emoji} {label}",
            'Score': score_value,
            'Percentage': score_value * 100
        })
    
    scores_df = pd.DataFrame(scores_data)
    
    # Service scores radar chart
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=scores_df['Score'].tolist() + [scores_df['Score'].iloc[0]],
        theta=scores_df['Category'].tolist() + [scores_df['Category'].iloc[0]],
        fill='toself',
        name=selected_branch,
        line_color='#667eea'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f"Service Specification Profile - {selected_branch}",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Service scores table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📋 Detailed Service Scores")
        scores_display = scores_df.copy()
        scores_display['Score'] = scores_display['Score'].round(3)
        scores_display['Percentage'] = scores_display['Percentage'].round(1)
        st.dataframe(scores_display, use_container_width=True, hide_index=True)
    
    with col2:
        # Traffic breakdown if available
        if branch_traffic is not None:
            st.markdown("##### 🚦 Traffic Analysis Details")
            traffic_details = {
                'Metric': [
                    'Total Roads',
                    'High Traffic Roads',
                    'Medium Traffic Roads', 
                    'Low Traffic Roads',
                    'Traffic Score',
                    'Category'
                ],
                'Value': [
                    branch_traffic['jumlah_jalan'],
                    branch_traffic['jumlah_jalan_padat'],
                    branch_traffic['jumlah_jalan_sedang'],
                    branch_traffic['jumlah_jalan_lancar'],
                    f"{branch_traffic['skor_potensi_traffic']:.1f}",
                    branch_traffic['kategori_traffic']
                ]
            }
            traffic_df = pd.DataFrame(traffic_details)
            st.dataframe(traffic_df, use_container_width=True, hide_index=True)
        else:
            st.info("No traffic data available")
    
    # Service Recommendations
    st.markdown("#### 💡 Branch-Specific Service Recommendations")
    
    # Generate specific recommendations based on scores
    recommendations = []
    
    # Primary service focus
    if branch_service['industrial_business_score'] >= 0.6:
        recommendations.append("🏭 **INDUSTRIAL FOCUS**: Prioritize KUR, Cash Management, and Trade Finance services")
        recommendations.append("👔 **STAFF**: Assign dedicated relationship managers for corporate clients")
    
    if branch_service['residential_consumer_score'] >= 0.6:
        recommendations.append("🏠 **CONSUMER BANKING**: Focus on KPR, family savings, and consumer loans")
        recommendations.append("👨‍👩‍👧‍👦 **FAMILY SERVICES**: Implement weekend banking and financial planning services")
    
    if branch_service['youth_digital_score'] >= 0.6:
        recommendations.append("📱 **DIGITAL TRANSFORMATION**: Prioritize mobile banking and reduce physical services")
        recommendations.append("🎮 **YOUTH ENGAGEMENT**: Implement student accounts and social media banking")
    
    if branch_service['corporate_office_score'] >= 0.6:
        recommendations.append("🏢 **CORPORATE BANKING**: Focus on payroll services and business accounts")
        recommendations.append("💼 **B2B SERVICES**: Develop corporate consultation and trade services")
    
    if branch_service['wealthy_premium_score'] >= 0.6:
        recommendations.append("💎 **WEALTH MANAGEMENT**: Implement private banking and investment advisory")
        recommendations.append("🏆 **PREMIUM EXPERIENCE**: Create VIP banking areas and priority services")
    
    # Competition and accessibility
    if branch_service['competitor_threat_score'] >= 0.7:
        recommendations.append("⚔️ **HIGH COMPETITION**: Implement differentiation strategy and competitive pricing")
    
    if branch_service['accessibility_score'] >= 0.7:
        recommendations.append("🚗 **HIGH ACCESSIBILITY**: Consider drive-thru banking and extended hours")
    elif branch_service['accessibility_score'] <= 0.3:
        recommendations.append("🚶 **LIMITED ACCESS**: Focus on appointment banking and personal service")
    
    # Cost optimization based on service priority
    if branch_service['service_priority'] == 'General':
        recommendations.append("💰 **COST OPTIMIZATION**: Consider service consolidation or branch merger analysis")
    elif branch_service['service_priority'] == 'Specialized':
        recommendations.append("🎯 **SPECIALIZATION**: Focus resources on dominant service area")
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="insight-card">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)
    
    if not recommendations:
        st.markdown("""
        <div class="recommendation-box">
            <h4>📊 Balanced Service Profile</h4>
            <p>This branch shows a balanced service profile across multiple categories. 
            Consider maintaining current service mix while monitoring market changes.</p>
        </div>
        """, unsafe_allow_html=True)

def generate_service_recommendations(service_results, poi_analysis, traffic_summary, service_map):
    """Generate service specification recommendations based on POI analysis"""
    recommendations = {}
    strategic_insights = []
    
    for branch in service_results.index:
        recs = []
        branch_data = service_results.loc[branch]
        dominant_service = branch_data['dominant_service']
        service_priority = branch_data['service_priority']
        final_score = branch_data['final_score']
        
        # Get detailed scores for each service category
        industrial_score = branch_data['industrial_business_score']
        residential_score = branch_data['residential_consumer_score']
        youth_score = branch_data['youth_digital_score']
        corporate_score = branch_data['corporate_office_score']
        wealthy_score = branch_data['wealthy_premium_score']
        education_score = branch_data['education_student_score']
        healthcare_score = branch_data['healthcare_insurance_score']
        competitor_score = branch_data['competitor_threat_score']
        accessibility_score = branch_data['accessibility_score']
        
        # Service Specification Recommendations
        # recs.append(f"🎯 **PRIMARY FOCUS**: {dominant_service.replace('_', ' ').title()} Services ({service_priority} Priority)")
        if isinstance(dominant_service, list):
            formatted_dominant = ', '.join(s.replace('_', ' ') for s in dominant_service)
        else:
            formatted_dominant = dominant_service.replace('_', ' ')

        recs.append(f"🎯 **PRIMARY FOCUS**: {formatted_dominant}")

        
        # Detailed service recommendations based on scores
        if industrial_score >= 0.6:
            recs.append("🏭 **INDUSTRIAL SERVICES**: Prioritaskan KUR, Cash Management, Trade Finance, Working Capital Loans")
            recs.append("📋 **OPERATIONAL**: Tambah relationship manager untuk corporate, extended business hours")
            
        if residential_score >= 0.6:
            recs.append("🏠 **CONSUMER SERVICES**: Focus KPR, Tabungan Keluarga, Asuransi Jiwa, Consumer Loans")
            recs.append("👨‍👩‍👧‍👦 **FAMILY BANKING**: Weekend service, family financial planning, child education savings")
            
        if youth_score >= 0.6:
            recs.append("📱 **DIGITAL FIRST**: Reduce physical services, prioritas mobile banking, e-wallet integration")
            recs.append("🎮 **YOUTH ENGAGEMENT**: Student accounts, digital payment solutions, social media banking")
            recs.append("⚡ **SERVICE REDUCTION**: Consider removing traditional services, focus on self-service")
            
        if corporate_score >= 0.6:
            recs.append("🏢 **CORPORATE BANKING**: Payroll services, business accounts, corporate loans, trade services")
            recs.append("💼 **B2B FOCUS**: Relationship banking, business consultation, corporate credit cards")
            
        if wealthy_score >= 0.6:
            recs.append("💎 **WEALTH MANAGEMENT**: Private banking, investment advisory, premium credit cards")
            recs.append("🏆 **PREMIUM SERVICES**: Priority banking, wealth planning, exclusive investment products")
            
        if education_score >= 0.6:
            recs.append("🎓 **EDUCATION SERVICES**: Tabungan pelajar, beasiswa program, student loans, educational insurance")
            recs.append("📚 **STUDENT BANKING**: Campus partnerships, dormitory banking, education financing")
            
        if healthcare_score >= 0.6:
            recs.append("🏥 **HEALTHCARE FINANCE**: Asuransi kesehatan, medical financing, healthcare investment products")
            recs.append("💊 **MEDICAL PARTNERSHIPS**: Hospital partnerships, medical insurance claims processing")
            
        # Competitor and accessibility strategies
        if competitor_score >= 0.7:
            recs.append("⚔️ **HIGH COMPETITION**: Implement differentiation strategy, competitive pricing, unique value propositions")
            recs.append("🛡️ **DEFENSIVE STRATEGY**: Customer retention programs, loyalty rewards, service excellence")
            
        if accessibility_score >= 0.7:
            recs.append("🚗 **HIGH ACCESSIBILITY**: Consider drive-thru banking, extended hours, mobile banking units")
            recs.append("⏰ **EXTENDED SERVICE**: 24/7 ATM, weekend banking, express service lanes")
        elif accessibility_score <= 0.3:
            recs.append("🚶 **LIMITED ACCESS**: Focus on appointment banking, personal service, relationship building")
            
        # Service reduction recommendations (cost optimization)
        services_to_reduce = []
        if youth_score >= 0.5:
            services_to_reduce.extend(["Traditional teller services", "Paper-based transactions", "Physical document storage"])
        if industrial_score <= 0.2 and corporate_score <= 0.2:
            services_to_reduce.extend(["Business loan processing", "Trade finance", "Corporate relationship manager"])
        if wealthy_score <= 0.2:
            services_to_reduce.extend(["Private banking suite", "Investment advisory", "Wealth management services"])
        if residential_score <= 0.2:
            services_to_reduce.extend(["KPR processing", "Family financial planning", "Consumer loan services"])
            
        if services_to_reduce:
            recs.append(f"❌ **SERVICES TO REDUCE/REMOVE**: {', '.join(services_to_reduce[:3])}")
            
        # Cost optimization recommendations
        if service_priority == 'General':
            recs.append("💰 **COST OPTIMIZATION**: Consider branch consolidation, shared services, reduced operating hours")
        elif service_priority == 'Specialized':
            recs.append("🎯 **SPECIALIZATION**: Focus budget on dominant service, reduce non-core services")
            
        # Branch consolidation analysis
        nearby_branches = []
        for other_branch in service_results.index:
            if other_branch != branch:
                if abs(service_results.loc[branch, 'final_score'] - service_results.loc[other_branch, 'final_score']) < 10:
                    # Similar service profiles
                    nearby_branches.append(other_branch)
                    
        if nearby_branches:
            recs.append(f"🔄 **CONSOLIDATION OPPORTUNITY**: Similar service profile to {nearby_branches[0]} - consider service integration")
        
        recommendations[branch] = {
            'recommendations': recs,
            'dominant_service': dominant_service,
            'service_priority': service_priority,
            'final_score': round(final_score, 1),
            'rank': int(branch_data['rank']),
            'cost_optimization_potential': 'High' if service_priority == 'General' else 'Medium' if youth_score >= 0.5 else 'Low'
        }
    
    # Generate strategic insights
    specialized_branches = [b for b, r in recommendations.items() if r['service_priority'] == 'Specialized']
    general_branches = [b for b, r in recommendations.items() if r['service_priority'] == 'General']
    high_digital_branches = [b for b in service_results.index if service_results.loc[b, 'youth_digital_score'] >= 0.6]
    
    if specialized_branches:
        strategic_insights.append(f"🎯 **SPECIALIZED BRANCHES**: {', '.join(specialized_branches)} have clear service focus - optimize for dominant services")
    
    if general_branches:
        strategic_insights.append(f"⚠️ **CONSOLIDATION CANDIDATES**: {', '.join(general_branches)} lack clear specialization - consider merger or service reduction")
        
    if high_digital_branches:
        strategic_insights.append(f"📱 **DIGITAL TRANSFORMATION**: {', '.join(high_digital_branches)} should prioritize digital services and reduce physical operations")
    
    # Service distribution analysis
    dominant_services = [r['dominant_service'] for r in recommendations.values()]
    service_distribution = pd.Series(dominant_services).value_counts()
    
    if service_distribution.iloc[0] > len(recommendations) * 0.4:
        strategic_insights.append(f"📊 **SERVICE CONCENTRATION**: Network over-focused on {service_distribution.index[0]} - consider diversification")
    
    return recommendations, strategic_insights

# Main Dashboard Layout
def main_dashboard():
    global SERVICE_WEIGHTS  # Declare global at the beginning of the function
    
    # File Upload Section
    st.markdown("## 📁 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 POI Data Upload")
        poi_file = st.file_uploader(
            "Upload POI Data CSV",
            type=['csv'],
            help="Required columns: name, enhanced_category, branch, lat, lon, distance_to_branch_m"
        )
        
        if poi_file is not None:
            try:
                poi_data = pd.read_csv(poi_file)
                required_cols = ['name', 'enhanced_category', 'branch', 'lat', 'lon', 'distance_to_branch_m']
                missing_cols = [col for col in required_cols if col not in poi_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    st.session_state.poi_data = poi_data
                    st.success(f"✅ POI data loaded: {len(poi_data)} records")
                    st.info(f"📊 Branches: {poi_data['branch'].nunique()}, Categories: {poi_data['enhanced_category'].nunique()}")
            except Exception as e:
                st.error(f"❌ Error loading POI data: {str(e)}")
    
    with col2:
        st.markdown("### 🚦 Traffic Data Upload")
        traffic_file = st.file_uploader(
            "Upload Traffic Data CSV", 
            type=['csv'],
            help="Required columns: nama_jalan, jenis_jalan, kategori_traffic, lat, lon, branch"
        )
        
        if traffic_file is not None:
            try:
                traffic_data = pd.read_csv(traffic_file)
                required_cols = ['nama_jalan', 'jenis_jalan', 'kategori_traffic', 'lat', 'lon', 'branch']
                missing_cols = [col for col in required_cols if col not in traffic_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                else:
                    st.session_state.traffic_data = analyze_traffic_data(traffic_data)
                    st.success(f"✅ Traffic data loaded: {len(traffic_data)} records")
                    st.info(f"📊 Branches: {traffic_data['branch'].nunique()}, Road types: {traffic_data['jenis_jalan'].nunique()}")
            except Exception as e:
                st.error(f"❌ Error loading traffic data: {str(e)}")
    
    # Service Specification Weight Configuration
    st.markdown("## ⚖️ Service Specification Weight Configuration")
    
    with st.expander("🔧 Customize Service Weights", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            industrial_weight = st.slider("Industrial/Business", 0.0, 0.3, SERVICE_WEIGHTS['industrial_business'], 0.01)
            residential_weight = st.slider("Residential/Consumer", 0.0, 0.3, SERVICE_WEIGHTS['residential_consumer'], 0.01)
            youth_weight = st.slider("Youth/Digital", 0.0, 0.25, SERVICE_WEIGHTS['youth_digital'], 0.01)
        
        with col2:
            corporate_weight = st.slider("Corporate/Office", 0.0, 0.25, SERVICE_WEIGHTS['corporate_office'], 0.01)
            wealthy_weight = st.slider("Wealthy/Premium", 0.0, 0.2, SERVICE_WEIGHTS['wealthy_premium'], 0.01)
            education_weight = st.slider("Education/Student", 0.0, 0.15, SERVICE_WEIGHTS['education_student'], 0.01)
            
        with col3:
            healthcare_weight = st.slider("Healthcare/Insurance", 0.0, 0.15, SERVICE_WEIGHTS['healthcare_insurance'], 0.01)
            competitor_weight = st.slider("Competitor Threat", 0.0, 0.2, SERVICE_WEIGHTS['competitor_threat'], 0.01)
            accessibility_weight = st.slider("Accessibility", 0.0, 0.15, SERVICE_WEIGHTS['accessibility'], 0.01)
        
        # Update weights
        custom_weights = {
            'industrial_business': industrial_weight,
            'residential_consumer': residential_weight,
            'youth_digital': youth_weight,
            'corporate_office': corporate_weight,
            'wealthy_premium': wealthy_weight,
            'education_student': education_weight,
            'healthcare_insurance': healthcare_weight,
            'competitor_threat': competitor_weight,
            'accessibility': accessibility_weight
        }
        
        total_weight = sum(custom_weights.values())
        st.markdown(f"**Total Weight: {total_weight:.2f}** {'✅' if abs(total_weight - 1.0) < 0.01 else '⚠️ Should sum to 1.0'}")
        
        if st.button("🔄 Update Service Weights"):
            if abs(total_weight - 1.0) < 0.01:
                SERVICE_WEIGHTS = custom_weights
                st.success("✅ Service weights updated successfully!")
                if st.session_state.analysis_complete:
                    st.session_state.analysis_complete = False  # Force recalculation
            else:
                st.error("❌ Weights must sum to 1.0")

    # Data Processing Section
    if st.session_state.poi_data is not None and st.session_state.traffic_data is not None:
        if not st.session_state.analysis_complete:
            st.markdown("## 🔄 Service Specification Analysis Processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Service Analysis", type="primary", use_container_width=True):
                    with st.spinner("🔄 Processing service specification analysis..."):
                        
                        # Distance Matrix
                        st.info("📏 Calculating distance matrix...")
                        distance_matrix = calculate_distance_matrix()
                        st.session_state.distance_matrix = distance_matrix
                        st.success("✅ Distance matrix calculated")
                        
                        # POI Analysis for Service Specification
                        st.info("🎯 Analyzing POI for service specification...")
                        poi_analysis, service_map = analyze_poi_for_service_specification(st.session_state.poi_data)
                        if poi_analysis is not None:
                            st.success("✅ POI analysis for service specification completed")
                            
                            # Service Specification Calculation
                            st.info("⚖️ Calculating service specification scores...")
                            service_results = calculate_service_specification_scores(
                                poi_analysis, 
                                st.session_state.traffic_data, 
                                distance_matrix,
                                SERVICE_WEIGHTS
                            )
                            st.session_state.service_results = service_results
                            st.success("✅ Service specification analysis completed")
                            
                            st.session_state.analysis_complete = True
                            st.rerun()
                        else:
                            st.error("❌ Failed to analyze POI for service specification")
            
            with col2:
                st.markdown(f"""
                <div class="mcda-card">
                    <h3>🎯 Service Specification Configuration</h3>
                    <p><strong>Current Weights:</strong></p>
                    <ul>
                        <li>Industrial/Business: {SERVICE_WEIGHTS['industrial_business']:.2f}</li>
                        <li>Residential/Consumer: {SERVICE_WEIGHTS['residential_consumer']:.2f}</li>
                        <li>Youth/Digital: {SERVICE_WEIGHTS['youth_digital']:.2f}</li>
                        <li>Corporate/Office: {SERVICE_WEIGHTS['corporate_office']:.2f}</li>
                        <li>Wealthy/Premium: {SERVICE_WEIGHTS['wealthy_premium']:.2f}</li>
                        <li>Education/Student: {SERVICE_WEIGHTS['education_student']:.2f}</li>
                        <li>Healthcare/Insurance: {SERVICE_WEIGHTS['healthcare_insurance']:.2f}</li>
                        <li>Competitor Threat: {SERVICE_WEIGHTS['competitor_threat']:.2f}</li>
                        <li>Accessibility: {SERVICE_WEIGHTS['accessibility']:.2f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload both POI and Traffic data files to begin analysis")
    
    # Analysis Results
    if st.session_state.analysis_complete and st.session_state.service_results is not None:
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🎯 Service Specification", 
            "📊 POI Analysis", 
            "🚦 Traffic Analysis", 
            "🗺️ Interactive Maps",
            "📈 EDA Dashboard",
            "🏦 Branch Breakdown",
            "💡 Service Recommendations"
        ])
        
        with tab1:
            st.markdown("### 🎯 Branch Service Specification Results")
            
            service_results = st.session_state.service_results
            
            # Service Specification overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = service_results['final_score'].mean()
                st.markdown(f"""
                <div class="mcda-card">
                    <h3>⭐ Average Score</h3>
                    <h2>{avg_score:.1f}</h2>
                    <p>Service Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                specialized_count = (service_results['service_priority'] == 'Specialized').sum()
                st.markdown(f"""
                <div class="mcda-card">
                    <h3>🎯 Specialized</h3>
                    <h2>{specialized_count}</h2>
                    <p>Focused Branches</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                top_score = service_results['final_score'].max()
                st.markdown(f"""
                <div class="mcda-card">
                    <h3>🏆 Top Score</h3>
                    <h2>{top_score:.1f}</h2>
                    <p>Best performer</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                digital_focus = (service_results['youth_digital_score'] >= 0.6).sum()
                st.markdown(f"""
                <div class="mcda-card">
                    <h3>📱 Digital Focus</h3>
                    <h2>{digital_focus}</h2>
                    <p>Youth-oriented</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Service Specification Results Table
            st.markdown("#### 🏆 Branch Service Specification Ranking")
            
            display_cols = [
                'rank', 'final_score', 'dominant_service', 'service_priority', 
                 'industrial_business_score', 'residential_consumer_score',
                 'youth_digital_score', 'corporate_office_score',
                 'wealthy_premium_score', 'education_student_score',
                 'healthcare_insurance_score', 'competitor_threat_score',
                 'accessibility_score', 'weighted_score'
            ]
            
            service_display = service_results[display_cols].round(3)
            service_display.columns = [
                'Rank', 'Final Score', 'Dominant Service', 'Service Priority', 
                 'Industrial Business Score', 'Residential Consumer Score',
                 'Youth Digital Score', 'Corporate Office Score',
                 'Wealthy Premium Score', 'Education Student Score',
                 'Healthcare Insurance Score', 'Competitor Threat Score',
                 'Accessibility Score', 'Weighted Score'
            ]
            
            st.dataframe(
                service_display,
                use_container_width=True,
                height=300
            )
            
            # Service Specification Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Service scores by branch
                fig_scores = px.bar(
                    x=service_results.index,
                    y=service_results['final_score'],
                    color=service_results['service_priority'],
                    title="Service Specification Scores by Branch",
                    color_discrete_map={
                        'Specialized': '#00b894',
                        'Targeted': '#fdcb6e', 
                        'General': '#ff6b6b'
                    }
                )
                fig_scores.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    xaxis_title="Branch",
                    yaxis_title="Service Score"
                )
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with col2:
                # Dominant service distribution
                service_dist = service_results['dominant_service'].value_counts()
                fig_pie = px.pie(
                    values=service_dist.values,
                    names=service_dist.index,
                    title="Dominant Service Distribution"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Service category heatmap
            st.markdown("#### 🔍 Service Category Analysis")
            
            service_cols = [
                'industrial_business_score', 'residential_consumer_score',
                 'youth_digital_score', 'corporate_office_score',
                 'wealthy_premium_score', 'education_student_score',
                 'healthcare_insurance_score', 'competitor_threat_score',
                 'accessibility_score', 'weighted_score'
            ]
            
            service_matrix = service_results[service_cols].T
            service_matrix.index = [col.replace('_score', '').replace('_', ' ').title() for col in service_matrix.index]
            
            fig_heatmap = px.imshow(
                service_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Service Category Strength by Branch"
            )
            fig_heatmap.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab2:
            st.markdown("### 📊 POI Analysis for Service Specification")
            
            if st.session_state.poi_data is not None:
                poi_analysis, service_map = analyze_poi_for_service_specification(st.session_state.poi_data)
                
                if poi_analysis is not None:
                    # POI overview
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_poi = len(st.session_state.poi_data)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📍 Total POI</h3>
                            <h2>{total_poi:,}</h2>
                            <p>Points analyzed</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        competitor_total = poi_analysis['Competitor_Banks'].sum()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🏦 Competitor Banks</h3>
                            <h2>{competitor_total}</h2>
                            <p>Excluding Maybank</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        industrial_total = poi_analysis['Industrial_Business'].sum()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>🏭 Industrial/Business</h3>
                            <h2>{industrial_total}</h2>
                            <p>Business centers</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        youth_total = poi_analysis['Youth_Digital'].sum()
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📱 Youth/Digital</h3>
                            <h2>{youth_total}</h2>
                            <p>Cafes, Malls, etc</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # POI Analysis Table
                    st.markdown("#### 📋 Service-Oriented POI Analysis by Branch")
                    
                    poi_display = poi_analysis.copy()
                    poi_display['Total'] = poi_display.sum(axis=1)
                    poi_display['Rank'] = poi_display['Total'].rank(method='dense', ascending=False).astype(int)
                    
                    # Reorder columns
                    cols = ['Rank', 'Total'] + [col for col in poi_display.columns if col not in ['Rank', 'Total']]
                    poi_display = poi_display[cols]
                    
                    st.dataframe(
                        poi_display,
                        use_container_width=True,
                        height=300
                    )
                    
                    # POI Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # POI heatmap
                        fig_heatmap = px.imshow(
                            poi_analysis.T,
                            labels=dict(x="Branches", y="Service Categories", color="Count"),
                            aspect="auto",
                            color_continuous_scale="Blues",
                            title="Service-Oriented POI Distribution"
                        )
                        fig_heatmap.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    with col2:
                        # Category totals
                        category_totals = poi_analysis.sum().sort_values(ascending=True)
                        fig_cat = px.bar(
                            x=category_totals.values,
                            y=category_totals.index,
                            orientation='h',
                            title="Total POI by Service Category",
                            color=category_totals.values,
                            color_continuous_scale="Viridis"
                        )
                        fig_cat.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_cat, use_container_width=True)
                    
                    # Service implications
                    st.markdown("#### 🎯 Service Implications by POI Category")
                    
                    for category, services in service_map.items():
                        if services:
                            category_name = category.replace('_', ' ').title()
                            services_text = ", ".join(services)
                            st.markdown(f"""
                            <div class="insight-card">
                                <h4>{category_name}</h4>
                                <p><strong>Recommended Services:</strong> {services_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                else:
                    st.error("❌ Failed to analyze POI data for service specification")
            else:
                st.warning("⚠️ No POI data available.")
        
        with tab3:
            st.markdown("### 🚦 Enhanced Traffic Analysis")
            
            if st.session_state.traffic_data is not None and not st.session_state.traffic_data.empty:
                traffic_summary = st.session_state.traffic_data
                
                # Traffic overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_traffic = traffic_summary['skor_potensi_traffic'].mean()
                    st.markdown(f"""
                    <div class="traffic-card">
                        <h3>🚦 Avg Traffic Score</h3>
                        <h2>{avg_traffic:.1f}</h2>
                        <p>Network average</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    high_traffic = (traffic_summary['kategori_traffic'] == 'Tinggi').sum()
                    st.markdown(f"""
                    <div class="traffic-card">
                        <h3>⚡ High Traffic</h3>
                        <h2>{high_traffic}</h2>
                        <p>Branches</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    medium_traffic = (traffic_summary['kategori_traffic'] == 'Sedang').sum()
                    st.markdown(f"""
                    <div class="traffic-card">
                        <h3>📊 Medium Traffic</h3>
                        <h2>{medium_traffic}</h2>
                        <p>Branches</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    low_traffic = (traffic_summary['kategori_traffic'] == 'Rendah').sum()
                    st.markdown(f"""
                    <div class="traffic-card">
                        <h3>🚶 Low Traffic</h3>
                        <h2>{low_traffic}</h2>
                        <p>Branches</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Traffic analysis table
                st.markdown("#### 📋 Traffic Analysis Results")
                
                traffic_display = traffic_summary.sort_values('skor_potensi_traffic', ascending=False).copy()
                traffic_display['rank'] = range(1, len(traffic_display) + 1)
                
                display_cols = [
                    'rank', 'cabang', 'kategori_traffic', 'skor_potensi_traffic',
                    'jumlah_jalan_padat', 'jumlah_jalan_sedang', 'jumlah_jalan_lancar'
                ]
                
                st.dataframe(
                    traffic_display[display_cols].round(2),
                    use_container_width=True,
                    height=300,
                    column_config={
                        "rank": "Rank",
                        "cabang": "Branch", 
                        "kategori_traffic": "Category",
                        "skor_potensi_traffic": "Traffic Score (0-100)",
                        "jumlah_jalan_padat": "Heavy Traffic Roads",
                        "jumlah_jalan_sedang": "Medium Traffic Roads",
                        "jumlah_jalan_lancar": "Light Traffic Roads"
                    }
                )
                
                # Traffic distribution visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Traffic score distribution
                    fig_traffic = px.bar(
                        traffic_display,
                        x='cabang',
                        y='skor_potensi_traffic',
                        color='kategori_traffic',
                        title="Traffic Scores by Branch",
                        color_discrete_map={
                            'Tinggi': '#00b894',
                            'Sedang': '#fdcb6e',
                            'Rendah': '#ff6b6b'
                        }
                    )
                    fig_traffic.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        xaxis_title="Branch",
                        yaxis_title="Traffic Score (0-100)"
                    )
                    st.plotly_chart(fig_traffic, use_container_width=True)
                
                with col2:
                    # Traffic category pie chart
                    category_counts = traffic_summary['kategori_traffic'].value_counts()
                    fig_pie = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Traffic Category Distribution",
                        color_discrete_map={
                            'Tinggi': '#00b894',
                            'Sedang': '#fdcb6e',
                            'Rendah': '#ff6b6b'
                        }
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            else:
                st.warning("⚠️ No traffic data available.")
        
        with tab4:
            st.markdown("### 🗺️ Interactive Maps")
            
            # Branch selector for maps
            selected_branch_map = st.selectbox(
                "Select Branch for Map View:",
                ["All Branches"] + list(branches.keys()),
                key="map_branch_selector"
            )
            
            if st.session_state.poi_data is not None:
                # Create and display POI map
                st.markdown("#### 📍 POI Distribution Map")
                poi_map = create_poi_map(st.session_state.poi_data, selected_branch_map)
                
                # Display map
                map_data = st_folium(poi_map, width=700, height=500)
                
                # Map statistics
                if selected_branch_map != "All Branches":
                    branch_poi = st.session_state.poi_data[st.session_state.poi_data['branch'] == selected_branch_map]
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 {selected_branch_map} Map Statistics</h4>
                        <p><strong>Total POIs:</strong> {len(branch_poi)}</p>
                        <p><strong>Categories:</strong> {branch_poi['enhanced_category'].nunique()}</p>
                        <p><strong>Average Distance:</strong> {branch_poi['distance_to_branch_m'].mean():.0f}m</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No POI data available for mapping.")
        
        with tab5:
            st.markdown("### 📈 Exploratory Data Analysis Dashboard")
            
            if st.session_state.poi_data is not None:
                poi_analysis, _ = analyze_poi_for_service_specification(st.session_state.poi_data)
                
                # EDA Overview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📊 POI Category Distribution (Network-wide)")
                    
                    # Category distribution across all branches
                    category_totals = st.session_state.poi_data['enhanced_category'].value_counts()
                    
                    fig_cat_dist = px.bar(
                        x=category_totals.index,
                        y=category_totals.values,
                        title="POI Categories Across All Branches",
                        color=category_totals.values,
                        color_continuous_scale="Viridis"
                    )
                    fig_cat_dist.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        showlegend=False
                    )
                    st.plotly_chart(fig_cat_dist, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🏦 POI Distribution by Branch")
                    
                    # POI count by branch
                    branch_totals = st.session_state.poi_data['branch'].value_counts()
                    
                    fig_branch_dist = px.bar(
                        x=branch_totals.values,
                        y=branch_totals.index,
                        orientation='h',
                        title="Total POIs by Branch",
                        color=branch_totals.values,
                        color_continuous_scale="Blues"
                    )
                    fig_branch_dist.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_branch_dist, use_container_width=True)
                
                with col3:
                    st.markdown("#### 📏 Distance Analysis")
                    
                    # Distance distribution
                    fig_distance = px.histogram(
                        st.session_state.poi_data,
                        x='distance_to_branch_m',
                        nbins=30,
                        title="POI Distance Distribution",
                        color_discrete_sequence=['#667eea']
                    )
                    fig_distance.update_layout(
                        height=400,
                        xaxis_title="Distance to Branch (meters)",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_distance, use_container_width=True)
                
                # Detailed Category Analysis
                st.markdown("#### 🎯 Detailed Category Analysis")
                
                # Category breakdown by branch
                category_branch = pd.crosstab(
                    st.session_state.poi_data['branch'], 
                    st.session_state.poi_data['enhanced_category']
                )
                
                # Interactive heatmap
                fig_heatmap_detailed = px.imshow(
                    category_branch,
                    labels=dict(x="Category", y="Branch", color="Count"),
                    aspect="auto",
                    color_continuous_scale="YlOrRd",
                    title="POI Category Distribution by Branch (Detailed)"
                )
                fig_heatmap_detailed.update_layout(height=500)
                st.plotly_chart(fig_heatmap_detailed, use_container_width=True)
                
                # Top categories per branch
                st.markdown("#### 🏆 Top POI Categories by Branch")
                
                branch_cols = st.columns(2)
                
                for i, branch in enumerate(branches.keys()):
                    with branch_cols[i % 2]:
                        branch_data = st.session_state.poi_data[st.session_state.poi_data['branch'] == branch]
                        
                        if not branch_data.empty:
                            top_categories = branch_data['enhanced_category'].value_counts().head(5)
                            
                            fig_top_cat = px.bar(
                                x=top_categories.values,
                                y=top_categories.index,
                                orientation='h',
                                title=f"Top Categories - {branch}",
                                color=top_categories.values,
                                color_continuous_scale="Plasma"
                            )
                            fig_top_cat.update_layout(
                                height=300,
                                showlegend=False,
                                margin=dict(t=50, b=30, l=10, r=10)
                            )
                            st.plotly_chart(fig_top_cat, use_container_width=True)
                        else:
                            st.info(f"No POI data for {branch}")
                
                # Service-oriented analysis
                if poi_analysis is not None:
                    st.markdown("#### 🎯 Service-Oriented Category Analysis")
                    
                    # Service category comparison
                    service_comparison = poi_analysis.T
                    
                    fig_service_comp = px.bar(
                        service_comparison,
                        title="Service Categories Comparison Across Branches",
                        barmode='group',
                        height=500
                    )
                    fig_service_comp.update_layout(
                        xaxis_title="Service Category",
                        yaxis_title="Count",
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_service_comp, use_container_width=True)
                    
                    # Service category correlation
                    correlation_matrix = poi_analysis.corr()
                    
                    fig_corr = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        title="Service Category Correlation Matrix"
                    )
                    fig_corr.update_layout(height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            else:
                st.warning("⚠️ No POI data available for EDA.")
        
        with tab6:
            st.markdown("### 🏦 Branch Breakdown Analysis")
            
            # Branch selector
            selected_branch = st.selectbox(
                "Select Branch for Detailed Analysis:",
                list(branches.keys()),
                key="breakdown_branch_selector"
            )
            
            if st.session_state.poi_data is not None and st.session_state.service_results is not None:
                poi_analysis, _ = analyze_poi_for_service_specification(st.session_state.poi_data)
                
                if poi_analysis is not None:
                    create_branch_breakdown(
                        poi_analysis, 
                        st.session_state.traffic_data, 
                        st.session_state.service_results, 
                        selected_branch
                    )
            else:
                st.warning("⚠️ Complete the analysis first to view branch breakdown.")
        
        with tab7:
            st.markdown("### 💡 Service Specification Recommendations")
            
            if st.session_state.service_results is not None:
                # Generate Service Specification recommendations
                poi_analysis, service_map = analyze_poi_for_service_specification(st.session_state.poi_data)
                recommendations, strategic_insights = generate_service_recommendations(
                    st.session_state.service_results,
                    poi_analysis,
                    st.session_state.traffic_data,
                    service_map
                )
                
                # Strategic insights
                st.markdown("#### 🎯 Strategic Service Insights")
                for insight in strategic_insights:
                    st.markdown(f"""
                    <div class="insight-card">
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Branch recommendations
                st.markdown("#### 🏦 Branch-Specific Service Recommendations")
                
                # Service priority filter
                priority_filter = st.selectbox(
                    "Filter by Service Priority:",
                    ["All", "Specialized", "Targeted", "General"]
                )
                
                # Cost optimization filter
                cost_filter = st.selectbox(
                    "Filter by Cost Optimization Potential:",
                    ["All", "High", "Medium", "Low"]
                )
                
                for branch, rec_data in recommendations.items():
                    service_priority = rec_data['service_priority']
                    cost_potential = rec_data['cost_optimization_potential']
                    
                    # Apply filters
                    if priority_filter != "All" and service_priority != priority_filter:
                        continue
                    if cost_filter != "All" and cost_potential != cost_filter:
                        continue
                    
                    # Card styling based on service priority
                    if service_priority == "Specialized":
                        card_class = "priority-high"
                        priority_icon = "🎯"
                    elif service_priority == "Targeted":
                        card_class = "priority-medium"
                        priority_icon = "⚡"
                    else:
                        card_class = "priority-low"
                        priority_icon = "📊"
                    
                    # recommendations_text = "<br>".join([f"• {rec}" for rec in rec_data['recommendations']])
                    # dominant_service = rec_data['dominant_service'].replace('_', ' ').title()
                    recommendations_text = "<br>".join([f"• {rec}" for rec in rec_data['recommendations']])

                    raw_dominant = rec_data['dominant_service']
                    if isinstance(raw_dominant, list):
                        dominant_service = ', '.join(s.replace('_', ' ') for s in raw_dominant)
                    else:
                        dominant_service = str(raw_dominant).replace('_', ' ')
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h3>{priority_icon} {branch}</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                            <span><strong>Service Priority:</strong> {service_priority}</span>
                            <span><strong>Final Score:</strong> {rec_data['final_score']}</span>
                            <span><strong>Rank:</strong> #{rec_data['rank']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 15px; font-size: 0.9em;">
                            <span><strong>Dominant Service:</strong> {dominant_service}</span>
                            <span><strong>Cost Optimization:</strong> {cost_potential}</span>
                        </div>
                        <hr style="margin: 15px 0; border-color: rgba(255,255,255,0.3);">
                        <div>{recommendations_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Service consolidation analysis
                st.markdown("#### 🔄 Branch Consolidation Analysis")
                
                general_branches = [b for b, r in recommendations.items() if r['service_priority'] == 'General']
                high_cost_branches = [b for b, r in recommendations.items() if r['cost_optimization_potential'] == 'High']
                
                if general_branches:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>⚠️ Consolidation Candidates</h4>
                        <p><strong>General Service Branches:</strong> {', '.join(general_branches)}</p>
                        <p>These branches lack clear service specialization and may benefit from consolidation or service reduction to optimize costs.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if high_cost_branches:
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4>💰 High Cost Optimization Potential</h4>
                        <p><strong>Branches:</strong> {', '.join(high_cost_branches)}</p>
                        <p>These branches show significant potential for cost reduction through service optimization and digital transformation.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.warning("⚠️ Complete the service specification analysis first.")

# Run the main dashboard
if __name__ == "__main__":
    main_dashboard()

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); border-radius: 15px; color: white;">
    <h3>🏦 Branch Service Specification Analysis Dashboard</h3>
    <p>Service Optimization • Cost Reduction • Digital Transformation • Strategic POI Analysis</p>
    <p style="opacity: 0.8;">🎯 Service Specification • 💰 Cost Efficiency • 📱 Digital Ready • 🔄 Branch Optimization</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar with Service Specification info
st.sidebar.markdown("""
## 🎯 Service Specification Analysis Features

### 📁 Data Requirements:
- **POI Data CSV**: name, enhanced_category, branch, lat, lon, distance_to_branch_m
- **Traffic Data CSV**: nama_jalan, jenis_jalan, kategori_traffic, lat, lon, branch

### 🏦 Service Categories:
- **Industrial/Business** - KUR, Cash Management, Trade Finance
- **Residential/Consumer** - KPR, Tabungan Keluarga, Consumer Loans  
- **Youth/Digital** - Digital Banking, E-wallet, Mobile Services
- **Corporate/Office** - Corporate Banking, Payroll, Business Loans
- **Wealthy/Premium** - Wealth Management, Private Banking
- **Education/Student** - Student Accounts, Education Financing
- **Healthcare/Insurance** - Health Insurance, Medical Financing
- **Competitor Analysis** - Market positioning and defensive strategies
- **Accessibility** - Extended hours, drive-thru, mobile banking

### 🚦 Enhanced Traffic Analysis:
- **Traffic Categories**: Padat (Heavy), Sedang (Medium), Lancar (Light)
- **Weighted Scoring**: Road type based scoring (0-100 scale)
- **Density Bonus**: Additional points for road network density
- **Service Impact**: Traffic affects accessibility and service hours

### 📊 New Features:
- **Interactive Maps** - POI visualization with branch filtering
- **EDA Dashboard** - Comprehensive exploratory data analysis
- **Branch Breakdown** - Detailed individual branch analysis
- **Category Bar Charts** - Visual POI category distribution
- **Service Correlation** - Inter-category relationship analysis

### 🎯 Analysis Benefits:
- **Cost Optimization** - Identify services to reduce/remove
- **Service Specialization** - Focus on dominant service types
- **Digital Transformation** - Youth-oriented branches for digital-first
- **Branch Consolidation** - Identify merger/closure candidates
- **Market Positioning** - Competitive analysis and strategy
""")

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
    <h4>🎯 Current Service Weights</h4>
    <p><small>Industrial: {SERVICE_WEIGHTS['industrial_business']:.2f} | Residential: {SERVICE_WEIGHTS['residential_consumer']:.2f}</small></p>
    <p><small>Youth/Digital: {SERVICE_WEIGHTS['youth_digital']:.2f} | Corporate: {SERVICE_WEIGHTS['corporate_office']:.2f}</small></p>
    <p><small>Premium: {SERVICE_WEIGHTS['wealthy_premium']:.2f} | Education: {SERVICE_WEIGHTS['education_student']:.2f}</small></p>
    <p><small>Healthcare: {SERVICE_WEIGHTS['healthcare_insurance']:.2f} | Competition: {SERVICE_WEIGHTS['competitor_threat']:.2f}</small></p>
    <p><strong>Total: {sum(SERVICE_WEIGHTS.values()):.2f}</strong></p>
</div>
""", unsafe_allow_html=True)