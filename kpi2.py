# Advanced Cross-Sell Analytics Dashboard (v5.0)
# =======================================
# Author: Claude • June 2025
# Description: Comprehensive cross-selling and customer behavior analytics
# ---------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from collections import Counter
import warnings
from datetime import datetime, timedelta
import itertools

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
#                    AUTOMATIC COLUMN MAPPING
# --------------------------------------------------------------
COLUMN_MAPPING = {
    'customer': 'Client Name',
    'category': 'Business Line', 
    'year': 'Year',
    'month': 'Month',
    'sales': 'Value'
}

ADDITIONAL_COLUMNS = ['Industry', 'Sector', 'Type']

# --------------------------------------------------------------
#                    DEPENDENCY CHECKING
# --------------------------------------------------------------
def check_dependencies():
    """Check and install required dependencies."""
    missing_deps = []
    
    try:
        import openpyxl
    except ImportError:
        missing_deps.append("openpyxl")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        missing_deps.append("plotly")
    
    try:
        import networkx as nx
    except ImportError:
        missing_deps.append("networkx")
    
    if missing_deps:
        st.error(f"""
        **Missing Dependencies Detected!**
        
        Please install the following packages:
        ```bash
        pip3 install {' '.join(missing_deps)}
        ```
        
        After installation, please restart your Streamlit app.
        """)
        st.stop()

check_dependencies()

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# --------------------------------------------------------------
#                       STREAMLIT CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Cross-Sell Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# --------------------------------------------------------------
#                    ADVANCED ANALYTICS FUNCTIONS
# --------------------------------------------------------------

@st.cache_data(show_spinner="Calculating cross-sell metrics...")
def calculate_cross_sell_metrics(_df: pd.DataFrame):
    """Calculate comprehensive cross-selling metrics."""
    try:
        metrics = {}
        
        # Customer journey analysis
        customer_journeys = _df.sort_values(['Account', 'Date']).groupby('Account').agg({
            'BusinessLine': lambda x: list(x),
            'Date': lambda x: list(x),
            'Revenue': lambda x: list(x)
        }).reset_index()
        
        # Cross-sell success metrics
        single_product_customers = customer_journeys[customer_journeys['BusinessLine'].apply(lambda x: len(set(x)) == 1)]
        multi_product_customers = customer_journeys[customer_journeys['BusinessLine'].apply(lambda x: len(set(x)) > 1)]
        
        metrics['cross_sell_success_rate'] = len(multi_product_customers) / len(customer_journeys) if len(customer_journeys) > 0 else 0
        metrics['total_customers'] = len(customer_journeys)
        metrics['cross_sell_customers'] = len(multi_product_customers)
        
        # Time to cross-sell analysis
        time_to_cross_sell = []
        for _, row in multi_product_customers.iterrows():
            business_lines = row['BusinessLine']
            dates = row['Date']
            
            # Find first occurrence of each unique business line
            first_occurrences = {}
            for bl, date in zip(business_lines, dates):
                if bl not in first_occurrences:
                    first_occurrences[bl] = date
            
            if len(first_occurrences) > 1:
                sorted_dates = sorted(first_occurrences.values())
                time_diff = (sorted_dates[1] - sorted_dates[0]).days
                time_to_cross_sell.append(time_diff)
        
        metrics['avg_time_to_cross_sell'] = np.mean(time_to_cross_sell) if time_to_cross_sell else None
        metrics['median_time_to_cross_sell'] = np.median(time_to_cross_sell) if time_to_cross_sell else None
        
        return metrics, customer_journeys
        
    except Exception as e:
        st.error(f"Error calculating cross-sell metrics: {str(e)}")
        return {}, pd.DataFrame()

@st.cache_data(show_spinner="Analyzing business line performance...")
def analyze_business_line_roles(_df: pd.DataFrame):
    """Analyze each business line's role in cross-selling (donor vs beneficiary)."""
    try:
        # Get customer journeys
        customer_journeys = _df.sort_values(['Account', 'Date']).groupby('Account').agg({
            'BusinessLine': lambda x: list(x),
            'Date': lambda x: list(x)
        }).reset_index()
        
        business_lines = _df['BusinessLine'].unique()
        bl_analysis = {}
        
        for bl in business_lines:
            bl_metrics = {
                'total_customers': 0,
                'as_first_product': 0,
                'leads_to_cross_sell': 0,
                'follows_other_products': 0,
                'donor_score': 0,
                'beneficiary_score': 0,
                'retention_customers': 0
            }
            
            for _, row in customer_journeys.iterrows():
                business_sequence = row['BusinessLine']
                unique_products = list(dict.fromkeys(business_sequence))  # Preserve order, remove duplicates
                
                if bl in business_sequence:
                    bl_metrics['total_customers'] += 1
                    
                    # Check if it's the first product
                    if unique_products[0] == bl:
                        bl_metrics['as_first_product'] += 1
                        
                        # Check if it leads to cross-sell
                        if len(set(business_sequence)) > 1:
                            bl_metrics['leads_to_cross_sell'] += 1
                    
                    # Check if it follows other products
                    if bl in unique_products[1:]:
                        bl_metrics['follows_other_products'] += 1
                    
                    # Check retention (customer comes back to this product)
                    if business_sequence.count(bl) > 1:
                        bl_metrics['retention_customers'] += 1
            
            # Calculate scores
            if bl_metrics['as_first_product'] > 0:
                bl_metrics['donor_score'] = bl_metrics['leads_to_cross_sell'] / bl_metrics['as_first_product']
            
            if bl_metrics['total_customers'] > 0:
                bl_metrics['beneficiary_score'] = bl_metrics['follows_other_products'] / bl_metrics['total_customers']
                bl_metrics['retention_rate'] = bl_metrics['retention_customers'] / bl_metrics['total_customers']
            
            bl_analysis[bl] = bl_metrics
        
        return bl_analysis
        
    except Exception as e:
        st.error(f"Error analyzing business line roles: {str(e)}")
        return {}

@st.cache_data(show_spinner="Creating cohort analysis...")
def create_cohort_analysis(_df: pd.DataFrame):
    """Create customer cohort analysis based on first purchase month."""
    try:
        # Get first purchase for each customer
        first_purchases = _df.sort_values('Date').groupby('Account').agg({
            'Date': 'first',
            'BusinessLine': 'first'
        }).reset_index()
        
        first_purchases['CohortMonth'] = first_purchases['Date'].dt.to_period('M')
        
        # Track customer behavior over time
        cohort_data = []
        
        for cohort_month in first_purchases['CohortMonth'].unique():
            cohort_customers = first_purchases[first_purchases['CohortMonth'] == cohort_month]['Account'].tolist()
            
            for period in range(12):  # Track for 12 months
                period_start = pd.to_datetime(str(cohort_month)) + pd.DateOffset(months=period)
                period_end = period_start + pd.DateOffset(months=1)
                
                period_data = _df[
                    (_df['Account'].isin(cohort_customers)) & 
                    (_df['Date'] >= period_start) & 
                    (_df['Date'] < period_end)
                ]
                
                if not period_data.empty:
                    active_customers = period_data['Account'].nunique()
                    unique_products = period_data.groupby('Account')['BusinessLine'].nunique().mean()
                    avg_revenue = period_data.groupby('Account')['Revenue'].sum().mean()
                else:
                    active_customers = 0
                    unique_products = 0
                    avg_revenue = 0
                
                cohort_data.append({
                    'CohortMonth': str(cohort_month),
                    'Period': period,
                    'ActiveCustomers': active_customers,
                    'CohortSize': len(cohort_customers),
                    'RetentionRate': active_customers / len(cohort_customers) if len(cohort_customers) > 0 else 0,
                    'AvgProductsPerCustomer': unique_products,
                    'AvgRevenuePerCustomer': avg_revenue
                })
        
        return pd.DataFrame(cohort_data)
        
    except Exception as e:
        st.error(f"Error creating cohort analysis: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Calculating market saturation...")
def calculate_market_saturation(_df: pd.DataFrame):
    """Calculate market saturation metrics by business line and segment."""
    try:
        saturation_data = []
        
        # Overall saturation
        total_customers = _df['Account'].nunique()
        
        for bl in _df['BusinessLine'].unique():
            bl_customers = _df[_df['BusinessLine'] == bl]['Account'].nunique()
            penetration = bl_customers / total_customers
            
            # Calculate growth trajectory
            monthly_data = _df[_df['BusinessLine'] == bl].groupby('YM')['Account'].nunique().reset_index()
            monthly_data = monthly_data.sort_values('YM')
            
            if len(monthly_data) > 1:
                recent_growth = (monthly_data['Account'].iloc[-1] - monthly_data['Account'].iloc[-2]) / monthly_data['Account'].iloc[-2]
            else:
                recent_growth = 0
            
            saturation_data.append({
                'BusinessLine': bl,
                'Customers': bl_customers,
                'MarketPenetration': penetration,
                'RecentGrowthRate': recent_growth,
                'SaturationLevel': 'High' if penetration > 0.7 else 'Medium' if penetration > 0.3 else 'Low'
            })
        
        # Segment-based saturation
        segment_saturation = []
        for segment_col in ['Industry', 'Sector', 'Type']:
            if segment_col in _df.columns:
                for segment in _df[segment_col].unique():
                    segment_customers = _df[_df[segment_col] == segment]['Account'].nunique()
                    
                    for bl in _df['BusinessLine'].unique():
                        bl_segment_customers = _df[
                            (_df[segment_col] == segment) & 
                            (_df['BusinessLine'] == bl)
                        ]['Account'].nunique()
                        
                        penetration = bl_segment_customers / segment_customers if segment_customers > 0 else 0
                        
                        segment_saturation.append({
                            'Segment': f"{segment_col}: {segment}",
                            'BusinessLine': bl,
                            'Penetration': penetration,
                            'Customers': bl_segment_customers,
                            'TotalSegmentCustomers': segment_customers
                        })
        
        return pd.DataFrame(saturation_data), pd.DataFrame(segment_saturation)
        
    except Exception as e:
        st.error(f"Error calculating market saturation: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# --------------------------------------------------------------
#                       MAIN DATA FUNCTIONS
# --------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_df(upload):
    """Load data from uploaded file with comprehensive error handling."""
    try:
        if upload.name.lower().endswith('.csv'):
            try:
                df = pd.read_csv(upload, encoding='utf-8')
            except UnicodeDecodeError:
                upload.seek(0)
                df = pd.read_csv(upload, encoding='latin-1')
        else:
            df = pd.read_excel(upload, sheet_name=0, engine='openpyxl')
        
        return df
        
    except Exception as e:
        st.error(f"""
        **Error loading file:** {str(e)}
        
        **Troubleshooting tips:**
        - Ensure the file is not corrupted
        - Check that the file format matches the extension
        - Try saving as a different format (CSV vs Excel)
        """)
        return None

def validate_auto_mapping(df: pd.DataFrame) -> tuple[bool, list]:
    """Validate that expected columns exist in the data."""
    issues = []
    
    required_columns = list(COLUMN_MAPPING.values())
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        issues.append(f"Available columns: {', '.join(df.columns)}")
        return False, issues
    
    numeric_cols = [COLUMN_MAPPING['year'], COLUMN_MAPPING['month'], COLUMN_MAPPING['sales']]
    for col in numeric_cols:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            if non_numeric > len(df) * 0.1:
                issues.append(f"Column '{col}' has {non_numeric} non-numeric values")
    
    return len(issues) == 0, issues

@st.cache_data(show_spinner="Preparing data...")
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with automatic column mapping."""
    try:
        processed_df = df.copy()
        
        rename_mapping = {
            COLUMN_MAPPING['customer']: 'Account',
            COLUMN_MAPPING['category']: 'BusinessLine',
            COLUMN_MAPPING['year']: 'Year',
            COLUMN_MAPPING['month']: 'Month',
            COLUMN_MAPPING['sales']: 'Revenue'
        }
        
        processed_df = processed_df.rename(columns=rename_mapping)
        
        for col in ADDITIONAL_COLUMNS:
            if col in df.columns:
                processed_df[col] = df[col]
        
        processed_df['Revenue'] = pd.to_numeric(processed_df['Revenue'], errors='coerce')
        processed_df['Year'] = pd.to_numeric(processed_df['Year'], errors='coerce')
        processed_df['Month'] = pd.to_numeric(processed_df['Month'], errors='coerce')
        
        processed_df.dropna(subset=['Revenue', 'Account', 'BusinessLine', 'Year', 'Month'], inplace=True)
        
        processed_df["Date"] = pd.to_datetime(
            processed_df['Year'].astype(str) + '-' + processed_df['Month'].astype(str) + '-01',
            errors='coerce'
        )
        
        processed_df["YM"] = processed_df["Date"].dt.to_period("M").astype(str)
        processed_df["Q"] = processed_df["Date"].dt.to_period("Q").astype(str)
        processed_df["H"] = processed_df["Date"].dt.year.astype(str) + "-H" + ((processed_df["Date"].dt.quarter + 1) // 2).astype(str)
        processed_df["Y"] = processed_df["Date"].dt.year.astype(str)
        
        return processed_df
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------------------
#                    VISUALIZATION FUNCTIONS
# --------------------------------------------------------------

def create_cross_sell_funnel(df: pd.DataFrame):
    """Create cross-sell funnel visualization."""
    customer_product_counts = df.groupby('Account')['BusinessLine'].nunique().value_counts().sort_index()
    
    fig = go.Figure()
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
    
    values = []
    labels = []
    colors_list = []
    
    for i, (product_count, customer_count) in enumerate(customer_product_counts.items()):
        values.append(customer_count)
        labels.append(f"{product_count} Product{'s' if product_count > 1 else ''}")
        colors_list.append(colors[i % len(colors)])
    
    fig.add_trace(go.Funnel(
        y=labels,
        x=values,
        textinfo="value+percent initial",
        marker=dict(color=colors_list),
        connector=dict(
            line=dict(color="royalblue", dash="solid", width=3)
        ),
        textfont=dict(size=14, color="white")
    ))
    
    fig.update_layout(
        title=dict(
            text="Cross-Sell Funnel: Customer Distribution by Product Count",
            font=dict(size=16)
        ),
        height=500,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

def create_business_line_role_chart(bl_analysis: dict):
    """Create donor vs beneficiary chart for business lines."""
    business_lines = list(bl_analysis.keys())
    donor_scores = [bl_analysis[bl]['donor_score'] for bl in business_lines]
    beneficiary_scores = [bl_analysis[bl]['beneficiary_score'] for bl in business_lines]
    sizes = [bl_analysis[bl]['total_customers'] for bl in business_lines]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=donor_scores,
        y=beneficiary_scores,
        mode='markers+text',
        marker=dict(
            size=[s/10 for s in sizes],  # Scale bubble size
            color=donor_scores,
            colorscale='RdYlGn',
            colorbar=dict(
                title=dict(
                    text="Donor Score",
                    font=dict(size=12)
                ),
                thickness=15,
                len=0.7,
                x=1.02,
                xanchor="left",
                tickfont=dict(size=10),
                tickformat=".2f"
            ),
            line=dict(width=2, color='darkblue')
        ),
        text=business_lines,
        textposition="middle center",
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Donor Score: %{x:.2f}<br>' +
            'Beneficiary Score: %{y:.2f}<br>' +
            'Total Customers: %{marker.size}<br>' +
            '<extra></extra>'
        )
    ))
    
    # Add quadrant lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Avg Beneficiary")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Avg Donor")
    
    # Add quadrant labels
    fig.add_annotation(x=0.8, y=0.8, text="High Donor<br>High Beneficiary", showarrow=False, 
                      bgcolor="rgba(0,255,0,0.1)", bordercolor="green")
    fig.add_annotation(x=0.2, y=0.8, text="Low Donor<br>High Beneficiary", showarrow=False,
                      bgcolor="rgba(255,255,0,0.1)", bordercolor="orange")
    fig.add_annotation(x=0.8, y=0.2, text="High Donor<br>Low Beneficiary", showarrow=False,
                      bgcolor="rgba(0,0,255,0.1)", bordercolor="blue")
    fig.add_annotation(x=0.2, y=0.2, text="Low Donor<br>Low Beneficiary", showarrow=False,
                      bgcolor="rgba(255,0,0,0.1)", bordercolor="red")
    
    fig.update_layout(
        title="Business Line Cross-Sell Roles: Donor vs Beneficiary Analysis",
        xaxis_title="Donor Score (Leads to Cross-Sell)",
        yaxis_title="Beneficiary Score (Benefits from Cross-Sell)",
        height=600,
        width=800
    )
    
    return fig

def create_customer_journey_heatmap(df: pd.DataFrame):
    """Create customer journey transition heatmap."""
    # Get transitions between business lines
    customer_journeys = df.sort_values(['Account', 'Date']).groupby('Account')['BusinessLine'].apply(list).reset_index()
    
    business_lines = sorted(df['BusinessLine'].unique())
    transition_matrix = pd.DataFrame(0, index=business_lines, columns=business_lines)
    
    for _, row in customer_journeys.iterrows():
        journey = row['BusinessLine']
        for i in range(len(journey) - 1):
            from_bl = journey[i]
            to_bl = journey[i + 1]
            if from_bl != to_bl:  # Only count actual transitions
                transition_matrix.loc[from_bl, to_bl] += 1
    
    # Calculate transition probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_probs = transition_matrix.div(row_sums, axis=0).fillna(0)
    
    fig = px.imshow(
        transition_probs.values,
        labels=dict(x="To Business Line", y="From Business Line", color="Transition Probability"),
        x=transition_probs.columns,
        y=transition_probs.index,
        color_continuous_scale="Blues",
        title="Customer Journey Transition Probabilities",
        text_auto='.2f'
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(
            title=dict(
                text="Probability",
                font=dict(size=12)
            ),
            tickformat=".2f",
            thickness=15
        )
    )
    
    return fig

def create_cohort_retention_chart(cohort_df: pd.DataFrame):
    """Create cohort retention analysis chart."""
    if cohort_df.empty:
        return go.Figure().add_annotation(
            text="No cohort data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
    
    # Pivot data for heatmap
    cohort_pivot = cohort_df.pivot(index='CohortMonth', columns='Period', values='RetentionRate')
    
    fig = px.imshow(
        cohort_pivot.values,
        labels=dict(x="Period (Months)", y="Cohort Month", color="Retention Rate"),
        x=[f"Month {i}" for i in cohort_pivot.columns],
        y=cohort_pivot.index,
        color_continuous_scale="YlOrRd",
        title="Customer Retention by Cohort",
        text_auto='.1%'
    )
    
    fig.update_layout(
        height=400,
        coloraxis_colorbar=dict(
            title=dict(
                text="Retention Rate",
                font=dict(size=12)
            ),
            tickformat=".1%",
            thickness=15
        )
    )
    
    return fig

# --------------------------------------------------------------
#                            MAIN APP
# --------------------------------------------------------------
def main():
    st.title("🚀 Advanced Cross-Sell Analytics Dashboard")
    st.markdown("*Comprehensive customer behavior and cross-selling intelligence*")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    file = st.sidebar.file_uploader(
        "Upload your sale_data.xlsx file", 
        type=["xlsx", "csv", "xls"],
        help="Upload your sales data file - columns will be mapped automatically"
    )
    
    with st.sidebar.expander("📋 Auto-Mapped Columns"):
        st.info(f"""
        **Columns automatically mapped:**
        - `{COLUMN_MAPPING['customer']}` → Customer/Account
        - `{COLUMN_MAPPING['category']}` → Business Line  
        - `{COLUMN_MAPPING['year']}` → Year
        - `{COLUMN_MAPPING['month']}` → Month
        - `{COLUMN_MAPPING['sales']}` → Revenue/Sales
        
        **Additional columns available:**
        {', '.join(ADDITIONAL_COLUMNS)}
        """)
    
    if not file:
        st.info("👋 **Welcome!** Please upload your sale_data.xlsx file to begin advanced cross-sell analysis.")
        
        st.markdown("""
        ## 🎯 What You'll Get:
        
        ### **Cross-Sell Intelligence**
        - Customer journey mapping and transition analysis
        - Cross-sell success rates and time-to-convert metrics
        - Business line role analysis (donor vs beneficiary)
        
        ### **Customer Behavior Analytics**
        - Cohort retention analysis
        - Market saturation by segments
        - Product adoption patterns
        
        ### **Actionable Insights**
        - Which products drive cross-selling
        - Optimal timing for cross-sell campaigns
        - Underperforming segments and opportunities
        """)
        
        return
    
    # Load and process data
    raw = load_df(file)
    if raw is None:
        return
    
    st.sidebar.success(f"✅ File loaded: {len(raw):,} rows")
    
    is_valid, issues = validate_auto_mapping(raw)
    if not is_valid:
        st.error("**Auto-mapping validation failed:**")
        for issue in issues:
            st.error(f"• {issue}")
        return
    
    df = prepare_data(raw)
    if df.empty:
        st.error("No valid data remaining after processing.")
        return
    
    # Filters
    with st.sidebar.expander("🔧 Analysis Filters"):
        if 'Industry' in df.columns:
            industries = st.multiselect(
                "Filter by Industry:",
                options=sorted(df['Industry'].unique()),
                default=sorted(df['Industry'].unique())
            )
            if industries:
                df = df[df['Industry'].isin(industries)]
        
        if 'Sector' in df.columns:
            sectors = st.multiselect(
                "Filter by Sector:",
                options=sorted(df['Sector'].unique()),
                default=sorted(df['Sector'].unique())
            )
            if sectors:
                df = df[df['Sector'].isin(sectors)]
    
    # Calculate metrics
    cross_sell_metrics, customer_journeys = calculate_cross_sell_metrics(df)
    bl_analysis = analyze_business_line_roles(df)
    cohort_data = create_cohort_analysis(df)
    saturation_data, segment_saturation = calculate_market_saturation(df)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Cross-Sell Performance", 
        "🔄 Business Line Analysis",
        "👥 Customer Journey Intelligence", 
        "📊 Cohort & Retention",
        "📈 Market Saturation",
        "💡 Actionable Insights"
    ])
    
    with tab1:
        st.header("🎯 Cross-Sell Performance Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Cross-Sell Success Rate",
                f"{cross_sell_metrics.get('cross_sell_success_rate', 0):.1%}",
                help="Percentage of customers who purchased multiple products"
            )
        
        with col2:
            st.metric(
                "Multi-Product Customers",
                f"{cross_sell_metrics.get('cross_sell_customers', 0):,}",
                f"of {cross_sell_metrics.get('total_customers', 0):,} total"
            )
        
        with col3:
            avg_time = cross_sell_metrics.get('avg_time_to_cross_sell')
            if avg_time:
                st.metric(
                    "Avg Time to Cross-Sell",
                    f"{avg_time:.0f} days",
                    help="Average days from first to second product purchase"
                )
            else:
                st.metric("Avg Time to Cross-Sell", "N/A")
        
        with col4:
            median_time = cross_sell_metrics.get('median_time_to_cross_sell')
            if median_time:
                st.metric(
                    "Median Time to Cross-Sell",
                    f"{median_time:.0f} days",
                    help="Median days from first to second product purchase"
                )
            else:
                st.metric("Median Time to Cross-Sell", "N/A")
        
        # Cross-sell funnel
        st.subheader("📊 Cross-Sell Funnel Analysis")
        
        with st.expander("ℹ️ What does this chart show?", expanded=False):
            st.markdown("""
            **Cross-Sell Funnel** shows how many customers purchase different numbers of products.
            
            **How it's calculated:**
            - Count unique products per customer
            - Group customers by number of products purchased
            - Display as funnel showing customer distribution
            
            **Business meaning:**
            - **1 Product:** Single-product customers (cross-sell opportunity)
            - **2+ Products:** Successfully cross-sold customers
            - **Wider sections:** More customers at that level
            - **Drop-offs:** Lost cross-sell opportunities
            
            **Key insight:** The bigger the drop from 1 to 2 products, the bigger your cross-sell opportunity.
            """)
        
        funnel_fig = create_cross_sell_funnel(df)
        st.plotly_chart(funnel_fig, use_container_width=True, key="cross_sell_funnel")
        
        # Customer journey heatmap
        st.subheader("🗺️ Customer Journey Transition Heatmap")
        
        with st.expander("ℹ️ What does this chart show?", expanded=False):
            st.markdown("""
            **Customer Journey Heatmap** shows the probability of customers moving from one product to another.
            
            **How it's calculated:**
            - Track each customer's product purchase sequence over time
            - Count transitions from Product A to Product B
            - Calculate probability: (A→B transitions) ÷ (total transitions from A)
            - Display as color-coded grid
            
            **How to read it:**
            - **Rows (Y-axis):** Starting product ("From")
            - **Columns (X-axis):** Next product purchased ("To")
            - **Color intensity:** Higher probability = darker blue
            - **Numbers:** Exact transition probability (0.0 to 1.0)
            
            **Business meaning:**
            - **Hot spots (dark blue):** Strong product combinations for bundling
            - **Cold spots (light blue):** Weak transitions needing attention
            - **Diagonal patterns:** Products that lead to repeat purchases
            """)
        
        heatmap_fig = create_customer_journey_heatmap(df)
        st.plotly_chart(heatmap_fig, use_container_width=True, key="journey_heatmap")
        
        # Insights with improved color coding and visibility
        st.subheader("💡 Key Performance Insights")
        
        success_rate = cross_sell_metrics.get('cross_sell_success_rate', 0)
        
        if success_rate > 0.5:
            st.success(f"""
            🎉 **EXCELLENT CROSS-SELL PERFORMANCE**  
            {success_rate:.1%} of customers purchase multiple products - well above industry average!
            """)
        elif success_rate > 0.3:
            st.warning(f"""
            ⚠️ **MODERATE CROSS-SELL SUCCESS**  
            {success_rate:.1%} cross-sell rate shows room for improvement. Target: 50%+
            """)
        else:
            st.error(f"""
            🚨 **LOW CROSS-SELL PERFORMANCE - URGENT ACTION NEEDED**  
            Only {success_rate:.1%} of customers buy multiple products. Major revenue opportunity!
            """)
        
        avg_time = cross_sell_metrics.get('avg_time_to_cross_sell')
        if avg_time and avg_time < 90:
            st.info(f"""
            ⚡ **FAST CROSS-SELL CONVERSION**  
            Average time of {avg_time:.0f} days suggests strong product synergy and customer satisfaction.
            """)
        elif avg_time and avg_time > 180:
            st.warning(f"""
            🐌 **SLOW CROSS-SELL CONVERSION**  
            {avg_time:.0f} days average suggests need for more proactive cross-sell campaigns and better timing.
            """)
    
    with tab2:
        st.header("🔄 Business Line Cross-Sell Analysis")
        
        if bl_analysis:
            # Business line role analysis
            st.subheader("🎭 Business Line Roles: Donor vs Beneficiary")
            
            with st.expander("ℹ️ What does this chart show?", expanded=False):
                st.markdown("""
                **Business Line Roles Chart** shows whether each product drives cross-selling or benefits from it.
                
                **How it's calculated:**
                - **Donor Score:** (Customers who bought other products after this one) ÷ (Customers who bought this first)
                - **Beneficiary Score:** (Times this product was bought after others) ÷ (Total customers for this product)
                - **Bubble Size:** Total number of customers for this product
                
                **The Four Quadrants:**
                - **Top Right (Green):** Cross-sell CHAMPIONS - both drive and benefit from cross-selling
                - **Bottom Right (Blue):** Cross-sell DRIVERS - great at leading to other purchases
                - **Top Left (Yellow):** Cross-sell TARGETS - often bought as add-ons
                - **Bottom Left (Red):** STANDALONE products - limited cross-sell activity
                
                **Business Actions:**
                - **Champions:** Promote heavily, use in bundles
                - **Drivers:** Lead with these in sales pitches
                - **Targets:** Perfect for upsell campaigns
                - **Standalone:** Investigate why they don't cross-sell
                """)
            
            role_fig = create_business_line_role_chart(bl_analysis)
            st.plotly_chart(role_fig, use_container_width=True, key="bl_roles")
            
            st.info("""
            **📖 Quick Reference Guide:**
            - **X-axis (Donor Score):** How well this product leads to cross-selling other products
            - **Y-axis (Beneficiary Score):** How often this product is purchased after other products  
            - **Bubble Size:** Total customer base for this product
            - **Green Zone (Top Right):** Products that both drive and benefit from cross-selling
            - **Blue Zone (Bottom Right):** Products that drive cross-selling but aren't bought as add-ons
            - **Yellow Zone (Top Left):** Products bought as add-ons but don't drive further sales
            - **Red Zone (Bottom Left):** Products with limited cross-sell activity
            """)
            
            # Detailed business line metrics
            st.subheader("📋 Detailed Business Line Performance")
            
            bl_df = pd.DataFrame(bl_analysis).T.reset_index()
            bl_df.columns = ['Business Line'] + [col for col in bl_df.columns if col != 'Business Line']
            
            # Format the dataframe
            formatted_bl_df = bl_df.style.format({
                'donor_score': '{:.2%}',
                'beneficiary_score': '{:.2%}',
                'retention_rate': '{:.2%}',
                'total_customers': '{:,}',
                'as_first_product': '{:,}',
                'leads_to_cross_sell': '{:,}',
                'follows_other_products': '{:,}',
                'retention_customers': '{:,}'
            })
            
            st.dataframe(formatted_bl_df, use_container_width=True)
            
            # Individual business line deep dive
            st.subheader("🔍 Individual Business Line Deep Dive")
            
            selected_bl = st.selectbox(
                "Select Business Line for Detailed Analysis:",
                options=list(bl_analysis.keys())
            )
            
            if selected_bl:
                bl_data = bl_analysis[selected_bl]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Key Metrics")
                    st.metric("Total Customers", f"{bl_data['total_customers']:,}")
                    st.metric("Donor Score", f"{bl_data['donor_score']:.1%}")
                    st.metric("Beneficiary Score", f"{bl_data['beneficiary_score']:.1%}")
                    st.metric("Retention Rate", f"{bl_data['retention_rate']:.1%}")
                
                with col2:
                    st.markdown("### 🎯 Strategic Recommendations")
                    
                    donor_score = bl_data['donor_score']
                    beneficiary_score = bl_data['beneficiary_score']
                    retention_rate = bl_data['retention_rate']
                    
                    if donor_score > 0.6:
                        st.success("🚀 **Strong Cross-Sell Driver** - Use as entry product in campaigns")
                    elif donor_score < 0.3:
                        st.warning("⚠️ **Weak Cross-Sell Driver** - Focus on improving product bundling")
                    
                    if beneficiary_score > 0.4:
                        st.info("🎁 **Strong Add-On Product** - Perfect for upsell campaigns")
                    
                    if retention_rate > 0.5:
                        st.success("🔄 **High Customer Loyalty** - Leverage for reference sales")
                    elif retention_rate < 0.2:
                        st.error("😞 **Low Retention** - Investigate customer satisfaction issues")
    
    with tab3:
        st.header("👥 Customer Journey Intelligence")
        
        # Customer segmentation by journey complexity
        st.subheader("🛤️ Customer Journey Complexity")
        
        with st.expander("ℹ️ What does this chart show?", expanded=False):
            st.markdown("""
            **Customer Journey Complexity** shows how many different products each customer has purchased.
            
            **How it's calculated:**
            - Count unique products per customer across all time
            - Group customers by this count
            - Display as bar chart
            
            **Business meaning:**
            - **1 Product:** Single-product customers (untapped potential)
            - **2 Products:** Basic cross-sell success
            - **3+ Products:** High-value, loyal customers
            - **Height of bars:** Number of customers in each category
            
            **Key insight:** Customers with more products typically have higher lifetime value.
            """)
        
        journey_complexity = customer_journeys['BusinessLine'].apply(lambda x: len(set(x))).value_counts().sort_index()
        
        fig_complexity = px.bar(
            x=journey_complexity.index,
            y=journey_complexity.values,
            labels={'x': 'Number of Different Products', 'y': 'Number of Customers'},
            title="Customer Distribution by Journey Complexity",
            color=journey_complexity.values,
            color_continuous_scale="Blues"
        )
        fig_complexity.update_layout(showlegend=False)
        st.plotly_chart(fig_complexity, use_container_width=True, key="journey_complexity")
        
        # Average revenue by journey complexity
        st.subheader("💰 Revenue Impact of Journey Complexity")
        
        with st.expander("ℹ️ What does this chart show?", expanded=False):
            st.markdown("""
            **Revenue Impact Analysis** shows how customer spending increases with product diversity.
            
            **How it's calculated:**
            - Calculate total revenue per customer across all purchases
            - Group customers by number of different products purchased
            - Calculate average revenue for each group
            - Display as bar chart
            
            **Business meaning:**
            - **X-axis:** Number of different products customer has bought
            - **Y-axis:** Average total revenue per customer in that group
            - **Higher bars:** More valuable customer segments
            
            **Key insight:** This shows the ROI of cross-selling - typically customers with more products spend significantly more.
            """)
        
        customer_revenue_complexity = []
        for _, row in customer_journeys.iterrows():
            account = row['Account']
            complexity = len(set(row['BusinessLine']))
            total_revenue = df[df['Account'] == account]['Revenue'].sum()
            customer_revenue_complexity.append({'Account': account, 'Complexity': complexity, 'Revenue': total_revenue})
        
        revenue_complexity_df = pd.DataFrame(customer_revenue_complexity)
        avg_revenue_by_complexity = revenue_complexity_df.groupby('Complexity')['Revenue'].mean().reset_index()
        
        fig_revenue_complexity = px.bar(
            avg_revenue_by_complexity,
            x='Complexity',
            y='Revenue',
            labels={'Complexity': 'Number of Different Products', 'Revenue': 'Average Revenue ($)'},
            title="Average Customer Revenue by Journey Complexity",
            color='Revenue',
            color_continuous_scale="Greens",
            text='Revenue'
        )
        fig_revenue_complexity.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_revenue_complexity.update_layout(showlegend=False)
        st.plotly_chart(fig_revenue_complexity, use_container_width=True, key="revenue_complexity")
        
        # Top customer journeys
        st.subheader("🔝 Most Common Customer Journeys")
        
        with st.expander("ℹ️ What does this table show?", expanded=False):
            st.markdown("""
            **Most Common Customer Journeys** shows the typical paths customers take through your products.
            
            **How it's calculated:**
            - Track each customer's product purchase sequence in chronological order
            - Remove consecutive duplicates (if customer buys same product multiple times)
            - Count how many customers follow each unique journey pattern
            - Show top 10 most common patterns
            
            **How to read it:**
            - **Journey Pattern:** Product A → Product B → Product C (in order of first purchase)
            - **Customer Count:** Number of customers who followed this exact path
            - **Percentage:** What % of all customers follow this journey
            
            **Business value:**
            - **Most common single product:** Your biggest opportunity for cross-selling
            - **Popular sequences:** Natural product progressions to encourage
            - **Sequential patterns:** Optimal timing for introducing next products
            """)
        
        journey_patterns = customer_journeys['BusinessLine'].apply(
            lambda x: ' → '.join(list(dict.fromkeys(x)))  # Remove consecutive duplicates
        ).value_counts().head(10)
        
        journey_df = pd.DataFrame({
            'Journey Pattern': journey_patterns.index,
            'Customer Count': journey_patterns.values,
            'Percentage': (journey_patterns.values / len(customer_journeys) * 100).round(1)
        })
        
        # Style the dataframe
        styled_journey_df = journey_df.style.format({
            'Customer Count': '{:,}',
            'Percentage': '{:.1f}%'
        }).background_gradient(subset=['Customer Count'], cmap='Blues')
        
        st.dataframe(styled_journey_df, use_container_width=True)
    
    with tab4:
        st.header("📊 Cohort & Retention Analysis")
        
        if not cohort_data.empty:
            # Cohort retention heatmap
            st.subheader("🔥 Customer Retention Cohort Analysis")
            
            with st.expander("ℹ️ What does this chart show?", expanded=False):
                st.markdown("""
                **Cohort Retention Analysis** tracks how well you retain customers over time, grouped by when they first purchased.
                
                **How it's calculated:**
                - Group customers by their first purchase month (cohort)
                - Track what % of each cohort remains active in subsequent months
                - Retention = (Active customers in month X) ÷ (Total customers in cohort)
                - Display as color-coded heatmap
                
                **How to read it:**
                - **Rows (Y-axis):** Customer cohorts by first purchase month
                - **Columns (X-axis):** Months after first purchase (0, 1, 2, etc.)
                - **Color intensity:** Darker red = higher retention rate
                - **Numbers:** Exact retention percentage
                
                **Business meaning:**
                - **Month 0:** Always 100% (all customers active in first month)
                - **Horizontal patterns:** How retention changes over time
                - **Vertical patterns:** Seasonal or market effects across cohorts
                - **Dark red areas:** Strong customer loyalty periods
                """)
            
            cohort_fig = create_cohort_retention_chart(cohort_data)
            st.plotly_chart(cohort_fig, use_container_width=True, key="cohort_retention")
            
            # Product expansion over time
            st.subheader("📈 Product Portfolio Expansion by Cohort")
            
            with st.expander("ℹ️ What does this chart show?", expanded=False):
                st.markdown("""
                **Product Portfolio Expansion** shows how customers gradually add more products over time.
                
                **How it's calculated:**
                - Track average number of different products per customer in each cohort
                - Calculate for each month after their first purchase
                - Display as color-coded heatmap
                
                **How to read it:**
                - **Rows:** Customer cohorts by first purchase month
                - **Columns:** Months after first purchase
                - **Green intensity:** Darker = more products per customer
                - **Numbers:** Average products per active customer
                
                **Business insights:**
                - **Upward trends:** Successful cross-selling over time
                - **Flat patterns:** Limited product expansion (opportunity!)
                - **Cohort differences:** Which customer groups expand product usage most
                """)
            
            cohort_pivot_products = cohort_data.pivot(index='CohortMonth', columns='Period', values='AvgProductsPerCustomer')
            
            fig_products = px.imshow(
                cohort_pivot_products.values,
                labels=dict(x="Period (Months)", y="Cohort Month", color="Avg Products per Customer"),
                x=[f"Month {i}" for i in cohort_pivot_products.columns],
                y=cohort_pivot_products.index,
                color_continuous_scale="Greens",
                title="Average Products per Customer by Cohort",
                text_auto=True
            )
            fig_products.update_layout(height=400)
            st.plotly_chart(fig_products, use_container_width=True, key="cohort_products")
            
            # Revenue progression
            st.subheader("💰 Revenue Progression by Cohort")
            
            with st.expander("ℹ️ What does this chart show?", expanded=False):
                st.markdown("""
                **Revenue Progression** tracks how customer spending evolves over time for different cohorts.
                
                **How it's calculated:**
                - Calculate average revenue per active customer in each month
                - Group by cohort (first purchase month)
                - Track progression over months since first purchase
                - Display as color-coded heatmap
                
                **How to read it:**
                - **Rows:** Customer cohorts by first purchase month
                - **Columns:** Months after first purchase
                - **Blue intensity:** Darker = higher average revenue per customer
                - **Numbers:** Average revenue per active customer that month
                
                **Business insights:**
                - **Growing intensity:** Customer value increasing over time
                - **Declining patterns:** Customer value erosion (requires attention)
                - **Cohort variations:** Which customer groups are most valuable long-term
                - **Seasonal patterns:** Revenue cycles across different time periods
                """)
            
            cohort_pivot_revenue = cohort_data.pivot(index='CohortMonth', columns='Period', values='AvgRevenuePerCustomer')
            
            fig_revenue = px.imshow(
                cohort_pivot_revenue.values,
                labels=dict(x="Period (Months)", y="Cohort Month", color="Avg Revenue per Customer ($)"),
                x=[f"Month {i}" for i in cohort_pivot_revenue.columns],
                y=cohort_pivot_revenue.index,
                color_continuous_scale="Blues",
                title="Average Revenue per Customer by Cohort",
                text_auto=True
            )
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True, key="cohort_revenue")
        else:
            st.warning("No cohort data available for analysis.")
    
    with tab5:
        st.header("📈 Market Saturation Analysis")
        
        if not saturation_data.empty:
            # Overall market saturation
            st.subheader("🌐 Overall Market Penetration by Business Line")
            
            with st.expander("ℹ️ What does this chart show?", expanded=False):
                st.markdown("""
                **Market Penetration Analysis** shows what percentage of your total customer base uses each product.
                
                **How it's calculated:**
                - Count unique customers who have purchased each business line
                - Calculate: (Customers for Product X) ÷ (Total unique customers) × 100%
                - Color-code by saturation level: Red (High 70%+), Orange (Medium 30-70%), Green (Low <30%)
                
                **How to read it:**
                - **X-axis:** Each business line/product
                - **Y-axis:** Market penetration percentage
                - **Bar colors:** 
                  - 🟢 **Green (Low):** Big growth opportunity
                  - 🟠 **Orange (Medium):** Moderate opportunity  
                  - 🔴 **Red (High):** Saturated market
                
                **Business actions:**
                - **Green bars:** Focus marketing spend here for growth
                - **Orange bars:** Optimize conversion and retention
                - **Red bars:** Focus on customer satisfaction and retention
                """)
            
            fig_saturation = px.bar(
                saturation_data,
                x='BusinessLine',
                y='MarketPenetration',
                color='SaturationLevel',
                labels={'MarketPenetration': 'Market Penetration (%)', 'BusinessLine': 'Business Line'},
                title="Market Penetration by Business Line",
                color_discrete_map={'High': '#dc3545', 'Medium': '#fd7e14', 'Low': '#28a745'},
                text='MarketPenetration'
            )
            fig_saturation.update_layout(yaxis_tickformat='.1%')
            fig_saturation.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            st.plotly_chart(fig_saturation, use_container_width=True, key="market_saturation")
            
            # Growth vs Penetration matrix
            st.subheader("📊 Growth vs Penetration Matrix")
            
            with st.expander("ℹ️ What does this chart show?", expanded=False):
                st.markdown("""
                **Growth vs Penetration Matrix** helps prioritize which products need attention based on market position and momentum.
                
                **How it's calculated:**
                - **X-axis (Penetration):** % of total customers using this product
                - **Y-axis (Growth Rate):** Recent month-over-month customer growth
                - **Bubble Size:** Total number of customers (bigger = more customers)
                
                **The Four Strategic Quadrants:**
                - **🟢 Top Left:** High Growth + Low Penetration = **RISING STARS** (invest heavily)
                - **🔵 Top Right:** High Growth + High Penetration = **MARKET LEADERS** (maintain dominance)  
                - **🟡 Bottom Left:** Low Growth + Low Penetration = **QUESTION MARKS** (fix or divest)
                - **🔴 Bottom Right:** Low Growth + High Penetration = **CASH COWS** (milk for profit)
                
                **Strategic Actions:**
                - **Rising Stars:** Increase marketing investment
                - **Market Leaders:** Defend market position  
                - **Question Marks:** Investigate problems or consider discontinuing
                - **Cash Cows:** Focus on efficiency and profitability
                """)
            
            fig_matrix = px.scatter(
                saturation_data,
                x='MarketPenetration',
                y='RecentGrowthRate',
                size='Customers',
                text='BusinessLine',
                labels={
                    'MarketPenetration': 'Market Penetration (%)',
                    'RecentGrowthRate': 'Recent Growth Rate (%)',
                    'Customers': 'Customer Count'
                },
                title="Strategic Position: Growth vs Penetration Matrix",
                hover_data={'Customers': ':,'}
            )
            
            # Add quadrant lines and labels
            fig_matrix.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig_matrix.add_vline(x=0.5, line_dash="dash", line_color="gray", line_width=1)
            
            # Add quadrant background colors
            fig_matrix.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=1, fillcolor="yellow", opacity=0.1, layer="below")  # Question Marks
            fig_matrix.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=1, fillcolor="red", opacity=0.1, layer="below")  # Cash Cows
            fig_matrix.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=1, fillcolor="green", opacity=0.1, layer="below")  # Rising Stars
            fig_matrix.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=1, fillcolor="blue", opacity=0.1, layer="below")  # Market Leaders
            
            fig_matrix.update_layout(height=500, xaxis_tickformat='.1%', yaxis_tickformat='.1%')
            st.plotly_chart(fig_matrix, use_container_width=True, key="growth_penetration_matrix")
            
            # Segment-based saturation
            if not segment_saturation.empty:
                st.subheader("🎯 Segment-Based Penetration Analysis")
                
                selected_segment = st.selectbox(
                    "Select Segment Type:",
                    options=segment_saturation['Segment'].str.split(':').str[0].unique()
                )
                
                filtered_segments = segment_saturation[
                    segment_saturation['Segment'].str.startswith(selected_segment)
                ]
                
                if not filtered_segments.empty:
                    pivot_segments = filtered_segments.pivot(
                        index='Segment', 
                        columns='BusinessLine', 
                        values='Penetration'
                    ).fillna(0)
                    
                    fig_segment_heatmap = px.imshow(
                        pivot_segments.values,
                        labels=dict(x="Business Line", y="Segment", color="Penetration Rate (%)"),
                        x=pivot_segments.columns,
                        y=[seg.split(': ')[1] for seg in pivot_segments.index],
                        color_continuous_scale="RdYlGn",
                        title=f"Market Penetration by {selected_segment}",
                        text_auto=True
                    )
                    
                    # Add explanation
                    st.markdown(f"""
                    **How to read this heatmap:**
                    - **Rows:** Different {selected_segment.lower()} segments  
                    - **Columns:** Business lines/products
                    - **Colors:** 🔴 Red (Low penetration) → 🟡 Yellow (Medium) → 🟢 Green (High penetration)
                    - **Numbers:** Exact penetration percentage in each segment
                    
                    **Business insights:**
                    - **Red areas:** Untapped opportunities for growth
                    - **Green areas:** Strong market position to defend
                    - **Patterns:** Which segments prefer which products
                    """)
                    
                    fig_segment_heatmap.update_layout(height=400)
                    st.plotly_chart(fig_segment_heatmap, use_container_width=True, key="segment_penetration")
        else:
            st.warning("No saturation data available for analysis.")
    
    with tab6:
        st.header("💡 Actionable Insights & Recommendations")
        
        # Cross-sell opportunities
        st.subheader("🎯 Priority Cross-Sell Opportunities")
        
        if bl_analysis:
            # Find best donor products
            best_donors = sorted(bl_analysis.items(), key=lambda x: x[1]['donor_score'], reverse=True)[:3]
            best_beneficiaries = sorted(bl_analysis.items(), key=lambda x: x[1]['beneficiary_score'], reverse=True)[:3]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🚀 Top Cross-Sell Drivers")
                for i, (bl, data) in enumerate(best_donors, 1):
                    st.write(f"**{i}. {bl}**")
                    st.write(f"   - Donor Score: {data['donor_score']:.1%}")
                    st.write(f"   - {data['leads_to_cross_sell']:,} successful cross-sells")
                    st.write("")
            
            with col2:
                st.markdown("### 🎁 Top Cross-Sell Targets")
                for i, (bl, data) in enumerate(best_beneficiaries, 1):
                    st.write(f"**{i}. {bl}**")
                    st.write(f"   - Beneficiary Score: {data['beneficiary_score']:.1%}")
                    st.write(f"   - {data['follows_other_products']:,} cross-sell purchases")
                    st.write("")
        
        # Strategic recommendations
        st.subheader("📋 Strategic Recommendations")
        
        success_rate = cross_sell_metrics.get('cross_sell_success_rate', 0)
        avg_time = cross_sell_metrics.get('avg_time_to_cross_sell', 0)
        
        recommendations = []
        
        if success_rate < 0.3:
            recommendations.append({
                'Priority': 'HIGH',
                'Area': 'Cross-Sell Strategy',
                'Issue': f'Low cross-sell success rate ({success_rate:.1%})',
                'Action': 'Implement aggressive cross-sell campaigns, review product bundling, train sales team on cross-sell techniques'
            })
        
        if avg_time and avg_time > 180:
            recommendations.append({
                'Priority': 'MEDIUM',
                'Area': 'Sales Velocity',
                'Issue': f'Slow cross-sell conversion ({avg_time:.0f} days)',
                'Action': 'Create time-limited offers, implement automated follow-up sequences, identify friction points in customer journey'
            })
        
        # Market saturation recommendations
        if not saturation_data.empty:
            low_penetration = saturation_data[saturation_data['SaturationLevel'] == 'Low']
            if not low_penetration.empty:
                for _, row in low_penetration.iterrows():
                    recommendations.append({
                        'Priority': 'MEDIUM',
                        'Area': 'Market Expansion',
                        'Issue': f'{row["BusinessLine"]} has low market penetration ({row["MarketPenetration"]:.1%})',
                        'Action': f'Increase marketing spend for {row["BusinessLine"]}, investigate market barriers, consider pricing adjustments'
                    })
        
        # Business line specific recommendations
        if bl_analysis:
            for bl, data in bl_analysis.items():
                if data['retention_rate'] < 0.2:
                    recommendations.append({
                        'Priority': 'HIGH',
                        'Area': 'Customer Retention',
                        'Issue': f'{bl} has low retention rate ({data["retention_rate"]:.1%})',
                        'Action': f'Investigate customer satisfaction issues with {bl}, improve product quality/service, implement retention programs'
                    })
        
        # Display recommendations with better formatting
        if recommendations:
            recommendations_df = pd.DataFrame(recommendations)
            
            # Color code by priority with better contrast
            def color_priority(val):
                if val == 'HIGH':
                    return 'background-color: #dc3545; color: white; font-weight: bold'
                elif val == 'MEDIUM':
                    return 'background-color: #ffc107; color: black; font-weight: bold'
                else:
                    return 'background-color: #28a745; color: white; font-weight: bold'
            
            styled_recommendations = recommendations_df.style.applymap(
                color_priority, subset=['Priority']
            ).set_properties(**{
                'font-size': '14px',
                'border': '1px solid #ddd'
            })
            
            st.markdown("### 🎯 Priority Action Items")
            st.dataframe(styled_recommendations, use_container_width=True)
            
            # Add legend
            st.markdown("""
            **Priority Legend:**
            - 🔴 **HIGH:** Immediate action required (revenue impact)
            - 🟡 **MEDIUM:** Address within 1-3 months  
            - 🟢 **LOW:** Monitor and optimize over time
            """)
        else:
            st.success("🎉 **EXCELLENT PERFORMANCE ACROSS ALL METRICS!** No critical issues identified. Your cross-sell strategy is performing optimally.")
        
        # Next steps with better formatting
        st.subheader("🎯 Strategic Action Plan")
        
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
        
        ### 📅 **WEEK 1-2: IMMEDIATE QUICK WINS**
        - **🎯 Launch targeted campaigns** for top donor products to their most likely cross-sell targets
        - **📦 Create product bundles** combining high-donor and high-beneficiary products  
        - **🤖 Implement automated follow-up** for customers 30-60 days after first purchase
        - **📊 Set up tracking** for cross-sell conversion rates and time-to-convert
        
        ### 📅 **MONTH 1-3: STRATEGIC INITIATIVES** 
        - **🎯 Develop segment-specific** cross-sell strategies based on penetration analysis
        - **⚠️ Address retention issues** for low-performing business lines
        - **🧪 A/B test timing** of cross-sell offers to optimize conversion rates
        - **💰 Analyze pricing** for bundled vs individual products
        
        ### 📅 **QUARTER 1-2: LONG-TERM TRANSFORMATION**
        - **🌱 Expand into underserved segments** identified in saturation analysis  
        - **🔬 Develop new product combinations** based on journey pattern insights
        - **🤖 Build predictive models** to identify high-potential cross-sell prospects
        - **📈 Establish benchmarks** and regular review cycles for cross-sell performance
        
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()