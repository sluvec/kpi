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
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
#                    AUTO FILE LOADING
# --------------------------------------------------------------
def get_default_file_path():
    """Get the path to sale_data.xlsx in the same directory as this script."""
    try:
        # Try to get the current script directory
        if __name__ == "__main__":
            current_dir = Path.cwd()
        else:
            # When running as module
            current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    except:
        # Fallback to current working directory
        current_dir = Path.cwd()
    
    file_path = current_dir / "sale_data.xlsx"
    return file_path

def load_default_data():
    """Try to load sale_data.xlsx from the script directory."""
    try:
        file_path = get_default_file_path()
        if file_path.exists():
            df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
            return df, str(file_path)
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading default file: {str(e)}")
        return None, None

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
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
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
#                    DATA VALIDATION & PREPARATION
# --------------------------------------------------------------

def validate_auto_mapping(df) -> tuple[bool, list]:
    """Validate that required columns exist for auto-mapping."""
    issues = []
    required_columns = ['Client Name', 'Business Line', 'Year', 'Month', 'Value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    if 'Value' in df.columns and not pd.api.types.is_numeric_dtype(df['Value']):
        issues.append("'Value' column must contain numeric data")
    if 'Year' in df.columns and not pd.api.types.is_numeric_dtype(df['Year']):
        issues.append("'Year' column must contain numeric data")
    if 'Month' in df.columns and not pd.api.types.is_numeric_dtype(df['Month']):
        issues.append("'Month' column must contain numeric data")
    return len(issues) == 0, issues

@st.cache_data(show_spinner="Preparing data...")
def prepare_data(df) -> pd.DataFrame:
    try:
        df_clean = df.copy()
        df_clean['Account'] = df_clean['Client Name']
        df_clean['Date'] = pd.to_datetime(df_clean[['Year', 'Month']].assign(day=1))
        df_clean['Revenue'] = pd.to_numeric(df_clean['Value'], errors='coerce')
        df_clean['BusinessLine'] = df_clean['Business Line']  # Map the column name
        df_clean = df_clean.dropna(subset=['Account', 'BusinessLine', 'Date', 'Revenue'])
        df_clean = df_clean[df_clean['Revenue'] > 0]
        return df_clean
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return pd.DataFrame()

# --------------------------------------------------------------
#                    ANALYTICS FUNCTIONS (CACHED)
# --------------------------------------------------------------

@st.cache_data(show_spinner="Calculating cross-sell metrics...")
def calculate_cross_sell_metrics(_df):
    try:
        metrics = {}
        customer_journeys = _df.sort_values(['Account', 'Date']).groupby('Account').agg({
            'BusinessLine': lambda x: list(x),
            'Date': lambda x: list(x),
            'Revenue': lambda x: list(x)
        }).reset_index()
        single_product_customers = customer_journeys[customer_journeys['BusinessLine'].apply(lambda x: len(set(x)) == 1)]
        multi_product_customers = customer_journeys[customer_journeys['BusinessLine'].apply(lambda x: len(set(x)) > 1)]
        metrics['cross_sell_success_rate'] = len(multi_product_customers) / len(customer_journeys) if len(customer_journeys) > 0 else 0
        metrics['total_customers'] = len(customer_journeys)
        metrics['cross_sell_customers'] = len(multi_product_customers)
        time_to_cross_sell = []
        for _, row in multi_product_customers.iterrows():
            business_lines = row['BusinessLine']
            dates = row['Date']
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
def analyze_business_line_roles(_df):
    try:
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
                business_line_list = row['BusinessLine']
                date_list = row['Date']
                first_occurrence_idx = None
                for i, bl_name in enumerate(business_line_list):
                    if bl_name == bl:
                        first_occurrence_idx = i
                        break
                if first_occurrence_idx is not None:
                    bl_metrics['total_customers'] += 1
                    if first_occurrence_idx == 0:
                        bl_metrics['as_first_product'] += 1
                        unique_products_after = set(business_line_list[first_occurrence_idx + 1:])
                        if len(unique_products_after) > 0:
                            bl_metrics['leads_to_cross_sell'] += 1
                    unique_products_before = set(business_line_list[:first_occurrence_idx])
                    if len(unique_products_before) > 0:
                        bl_metrics['follows_other_products'] += 1
                    product_count = business_line_list.count(bl)
                    if product_count > 1:
                        bl_metrics['retention_customers'] += 1
            if bl_metrics['as_first_product'] > 0:
                bl_metrics['donor_score'] = bl_metrics['leads_to_cross_sell'] / bl_metrics['as_first_product']
            if bl_metrics['total_customers'] > 0:
                bl_metrics['beneficiary_score'] = bl_metrics['follows_other_products'] / bl_metrics['total_customers']
                bl_metrics['retention_rate'] = bl_metrics['retention_customers'] / bl_metrics['total_customers']
            else:
                bl_metrics['retention_rate'] = 0
            bl_analysis[bl] = bl_metrics
        return bl_analysis
    except Exception as e:
        st.error(f"Error analyzing business line roles: {str(e)}")
        return {}

@st.cache_data(show_spinner="Creating cohort analysis...")
def create_cohort_analysis(_df):
    try:
        cohort_data = []
        customer_first_purchase = _df.groupby('Account')['Date'].min().reset_index()
        customer_first_purchase.columns = ['Account', 'FirstPurchaseDate']
        df_with_cohort = _df.merge(customer_first_purchase, on='Account')
        df_with_cohort['Period'] = ((df_with_cohort['Date'].dt.year - df_with_cohort['FirstPurchaseDate'].dt.year) * 12 + 
                                   (df_with_cohort['Date'].dt.month - df_with_cohort['FirstPurchaseDate'].dt.month))
        df_with_cohort['CohortMonth'] = df_with_cohort['FirstPurchaseDate'].dt.to_period('M')
        
        # Convert Period objects to strings for JSON serialization
        df_with_cohort['CohortMonth'] = df_with_cohort['CohortMonth'].astype(str)
        
        for cohort_month in df_with_cohort['CohortMonth'].unique():
            cohort_customers = df_with_cohort[df_with_cohort['CohortMonth'] == cohort_month]['Account'].unique()
            total_cohort_customers = len(cohort_customers)
            for period in range(12):
                period_customers = df_with_cohort[
                    (df_with_cohort['CohortMonth'] == cohort_month) & 
                    (df_with_cohort['Period'] == period)
                ]['Account'].unique()
                active_customers = len(period_customers)
                retention_rate = active_customers / total_cohort_customers if total_cohort_customers > 0 else 0
                period_data = df_with_cohort[
                    (df_with_cohort['CohortMonth'] == cohort_month) & 
                    (df_with_cohort['Period'] == period)
                ]
                if not period_data.empty:
                    avg_products = period_data.groupby('Account')['BusinessLine'].nunique().mean()
                else:
                    avg_products = 0
                cohort_data.append({
                    'CohortMonth': cohort_month,
                    'Period': period,
                    'TotalCustomers': total_cohort_customers,
                    'ActiveCustomers': active_customers,
                    'RetentionRate': retention_rate,
                    'AvgProductsPerCustomer': avg_products
                })
        return pd.DataFrame(cohort_data)
    except Exception as e:
        st.error(f"Error creating cohort analysis: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Analyzing top customer performance...")
def analyze_top_customers(_df):
    try:
        customer_metrics = _df.groupby('Account').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'BusinessLine': lambda x: len(set(x)),
            'Date': ['min', 'max']
        }).reset_index()
        customer_metrics.columns = ['Account', 'Total_Revenue', 'Avg_Transaction_Value', 'Transaction_Count', 
                                  'Unique_Products', 'First_Purchase', 'Last_Purchase']
        customer_metrics['Customer_Lifetime_Days'] = (customer_metrics['Last_Purchase'] - customer_metrics['First_Purchase']).dt.days
        customer_metrics['Revenue_Per_Day'] = customer_metrics['Total_Revenue'] / customer_metrics['Customer_Lifetime_Days'].replace(0, 1)
        customer_metrics = customer_metrics.sort_values('Total_Revenue', ascending=False)
        return customer_metrics
    except Exception as e:
        st.error(f"Error analyzing top customers: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Creating customer cross-sell journey analysis...")
def analyze_customer_cross_sell_journeys(_df, top_customers):
    try:
        if not top_customers:
            return pd.DataFrame()
        df_top = _df[_df['Account'].isin(top_customers)]
        journey_data = []
        for customer in top_customers:
            customer_data = df_top[df_top['Account'] == customer].sort_values('Date')
            if len(customer_data) > 1:
                for i in range(len(customer_data) - 1):
                    current_product = customer_data.iloc[i]['BusinessLine']
                    next_product = customer_data.iloc[i + 1]['BusinessLine']
                    days_between = (customer_data.iloc[i + 1]['Date'] - customer_data.iloc[i]['Date']).days
                    journey_data.append({
                        'Account': customer,
                        'From_Product': current_product,
                        'To_Product': next_product,
                        'Days_Between': days_between,
                        'Revenue_From': customer_data.iloc[i]['Revenue'],
                        'Revenue_To': customer_data.iloc[i + 1]['Revenue']
                    })
        return pd.DataFrame(journey_data)
    except Exception as e:
        st.error(f"Error analyzing customer cross-sell journeys: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Calculating customer business line penetration...")
def calculate_customer_bl_penetration(_df):
    try:
        customer_bl_matrix = _df.groupby(['Account', 'BusinessLine'])['Revenue'].sum().unstack(fill_value=0)
        penetration_data = []
        for customer in customer_bl_matrix.index:
            customer_revenue = customer_bl_matrix.loc[customer]
            total_revenue = customer_revenue.sum()
            if total_revenue > 0:
                for bl in customer_bl_matrix.columns:
                    bl_revenue = customer_revenue[bl]
                    penetration_pct = (bl_revenue / total_revenue) * 100
                    penetration_data.append({
                        'Account': customer,
                        'BusinessLine': bl,
                        'Revenue': bl_revenue,
                        'Total_Revenue': total_revenue,
                        'Penetration_Pct': penetration_pct
                    })
        return pd.DataFrame(penetration_data)
    except Exception as e:
        st.error(f"Error calculating customer business line penetration: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Analyzing customer growth patterns...")
def analyze_customer_growth_patterns(_df, top_customers):
    try:
        if not top_customers:
            return pd.DataFrame()
        df_top = _df[_df['Account'].isin(top_customers)]
        monthly_growth = df_top.groupby(['Account', df_top['Date'].dt.to_period('M')]).agg({
            'Revenue': 'sum',
            'BusinessLine': lambda x: len(set(x))
        }).reset_index()
        monthly_growth.columns = ['Account', 'Month', 'Monthly_Revenue', 'Monthly_Products']
        growth_data = []
        for customer in top_customers:
            customer_data = monthly_growth[monthly_growth['Account'] == customer].sort_values('Month')
            if len(customer_data) > 1:
                for i in range(1, len(customer_data)):
                    current_month = customer_data.iloc[i]
                    prev_month = customer_data.iloc[i - 1]
                    revenue_growth = ((current_month['Monthly_Revenue'] - prev_month['Monthly_Revenue']) / 
                                    prev_month['Monthly_Revenue'] * 100) if prev_month['Monthly_Revenue'] > 0 else 0
                    product_growth = current_month['Monthly_Products'] - prev_month['Monthly_Products']
                    growth_data.append({
                        'Account': customer,
                        'Month': current_month['Month'],
                        'Revenue_Growth_Pct': revenue_growth,
                        'Product_Growth': product_growth,
                        'Monthly_Revenue': current_month['Monthly_Revenue'],
                        'Monthly_Products': current_month['Monthly_Products']
                    })
        return pd.DataFrame(growth_data)
    except Exception as e:
        st.error(f"Error analyzing customer growth patterns: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Calculating market saturation...")
def calculate_market_saturation(_df):
    try:
        bl_saturation = _df.groupby('BusinessLine').agg({
            'Account': 'nunique',
            'Revenue': 'sum'
        }).reset_index()
        bl_saturation.columns = ['BusinessLine', 'Unique_Customers', 'Total_Revenue']
        segment_saturation = pd.DataFrame()
        if 'Industry' in _df.columns:
            segment_saturation = _df.groupby(['Industry', 'BusinessLine']).agg({
                'Account': 'nunique',
                'Revenue': 'sum'
            }).reset_index()
            segment_saturation.columns = ['Industry', 'BusinessLine', 'Unique_Customers', 'Total_Revenue']
        return bl_saturation, segment_saturation
    except Exception as e:
        st.error(f"Error calculating market saturation: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(show_spinner="Creating temporal analysis...")
def create_temporal_analysis(_df, granularity="Monthly"):
    try:
        if granularity == "Monthly":
            _df['TimeGroup'] = _df['Date'].dt.to_period('M')
        elif granularity == "Quarterly":
            _df['TimeGroup'] = _df['Date'].dt.to_period('Q')
        else:
            _df['TimeGroup'] = _df['Date'].dt.to_period('Y')
        
        # Convert Period objects to strings for JSON serialization
        _df['TimeGroup'] = _df['TimeGroup'].astype(str)
        
        temporal_metrics = _df.groupby('TimeGroup').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Account': 'nunique',
            'BusinessLine': lambda x: len(set(x))
        }).reset_index()
        temporal_metrics.columns = ['TimeGroup', 'Total_Revenue', 'Avg_Transaction_Value', 'Transaction_Count', 'Unique_Customers', 'Active_Business_Lines']
        temporal_metrics = temporal_metrics.reset_index()
        cross_sell_data = []
        for period in temporal_metrics['TimeGroup']:
            period_data = _df[_df['TimeGroup'] == period]
            if not period_data.empty:
                period_customers = period_data.groupby('Account')['BusinessLine'].nunique()
                cross_sell_customers = (period_customers > 1).sum()
                total_customers = len(period_customers)
                cross_sell_rate = cross_sell_customers / total_customers if total_customers > 0 else 0
                cross_sell_data.append({
                    'TimeGroup': period,
                    'Cross_Sell_Rate': cross_sell_rate
                })
        cross_sell_df = pd.DataFrame(cross_sell_data)
        temporal_metrics = temporal_metrics.merge(cross_sell_df, on='TimeGroup', how='left')
        temporal_metrics = temporal_metrics.sort_values('TimeGroup')
        temporal_metrics['Revenue_Growth'] = temporal_metrics['Total_Revenue'].pct_change() * 100
        temporal_metrics['Customer_Growth'] = temporal_metrics['Unique_Customers'].pct_change() * 100
        return temporal_metrics
    except Exception as e:
        st.error(f"Error creating temporal analysis: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Analyzing seasonal patterns...")
def analyze_seasonal_patterns(_df):
    try:
        monthly_patterns = _df.groupby(_df['Date'].dt.month).agg({
            'Revenue': 'sum',
            'Account': 'nunique',
            'BusinessLine': lambda x: len(set(x))
        }).reset_index()
        monthly_patterns.columns = ['Month', 'Total_Revenue', 'Unique_Customers', 'Active_Business_Lines']
        quarterly_patterns = _df.groupby(_df['Date'].dt.quarter).agg({
            'Revenue': 'sum',
            'Account': 'nunique',
            'BusinessLine': lambda x: len(set(x))
        }).reset_index()
        quarterly_patterns.columns = ['Quarter', 'Total_Revenue', 'Unique_Customers', 'Active_Business_Lines']
        yearly_comparison = _df.groupby(_df['Date'].dt.year).agg({
            'Revenue': 'sum',
            'Account': 'nunique',
            'BusinessLine': lambda x: len(set(x))
        }).reset_index()
        yearly_comparison.columns = ['Year', 'Total_Revenue', 'Unique_Customers', 'Active_Business_Lines']
        return monthly_patterns, quarterly_patterns, yearly_comparison
    except Exception as e:
        st.error(f"Error analyzing seasonal patterns: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(show_spinner="Creating business line temporal analysis...")
def create_bl_temporal_analysis(_df, granularity="Monthly"):
    try:
        if granularity == "Monthly":
            _df['TimeGroup'] = _df['Date'].dt.to_period('M')
        elif granularity == "Quarterly":
            _df['TimeGroup'] = _df['Date'].dt.to_period('Q')
        else:
            _df['TimeGroup'] = _df['Date'].dt.to_period('Y')
        
        # Convert Period objects to strings for JSON serialization
        _df['TimeGroup'] = _df['TimeGroup'].astype(str)
        
        bl_temporal = _df.groupby(['TimeGroup', 'BusinessLine']).agg({
            'Revenue': 'sum',
            'Account': 'nunique'
        }).reset_index()
        bl_temporal.columns = ['TimeGroup', 'BusinessLine', 'Revenue', 'Unique_Customers']
        bl_revenue_pivot = bl_temporal.pivot(index='TimeGroup', columns='BusinessLine', values='Revenue').fillna(0)
        bl_customers_pivot = bl_temporal.pivot(index='TimeGroup', columns='BusinessLine', values='Unique_Customers').fillna(0)
        bl_revenue_pivot_pct = bl_revenue_pivot.div(bl_revenue_pivot.sum(axis=1), axis=0) * 100
        return bl_temporal, bl_revenue_pivot, bl_customers_pivot, bl_revenue_pivot_pct
    except Exception as e:
        st.error(f"Error creating business line temporal analysis: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --------------------------------------------------------------
#                    CHART CREATION FUNCTIONS
# --------------------------------------------------------------

def create_cross_sell_funnel(df):
    """Create cross-sell funnel chart."""
    try:
        customer_product_counts = df.groupby('Account')['BusinessLine'].nunique()
        funnel_data = customer_product_counts.value_counts().sort_index()
        
        fig = go.Figure(go.Funnel(
            y=[f"{i} Product{'s' if i > 1 else ''}" for i in funnel_data.index],
            x=funnel_data.values,
            textinfo="value+percent initial",
            textposition="inside",
            marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]},
            connector={"line": {"color": "royalblue", "width": 3}}
        ))
        
        fig.update_layout(
            title="Cross-Sell Funnel: Customer Distribution by Product Count",
            height=500,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating cross-sell funnel: {str(e)}")
        return go.Figure()

def create_business_line_role_chart(bl_analysis):
    """Create business line role analysis chart."""
    try:
        bl_data = []
        for bl, metrics in bl_analysis.items():
            bl_data.append({
                'BusinessLine': bl,
                'DonorScore': metrics['donor_score'],
                'BeneficiaryScore': metrics['beneficiary_score'],
                'TotalCustomers': metrics['total_customers'],
                'RetentionRate': metrics['retention_rate']
            })
        
        df_bl = pd.DataFrame(bl_data)
        
        fig = px.scatter(
            df_bl,
            x='DonorScore',
            y='BeneficiaryScore',
            size='TotalCustomers',
            color='RetentionRate',
            hover_data=['BusinessLine', 'TotalCustomers'],
            title="Business Line Roles: Donor vs Beneficiary Analysis",
            labels={
                'DonorScore': 'Donor Score (Drives Cross-Selling)',
                'BeneficiaryScore': 'Beneficiary Score (Benefits from Cross-Selling)',
                'TotalCustomers': 'Total Customers',
                'RetentionRate': 'Retention Rate'
            },
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(
            height=600,
            xaxis_title="Donor Score (How well this product leads to cross-selling)",
            yaxis_title="Beneficiary Score (How often this product is bought as add-on)"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating business line role chart: {str(e)}")
        return go.Figure()

def create_customer_journey_heatmap(df):
    """Create customer journey transition heatmap."""
    try:
        customer_journeys = df.sort_values(['Account', 'Date']).groupby('Account')['BusinessLine'].apply(list).reset_index()
        
        transitions = []
        for _, row in customer_journeys.iterrows():
            business_lines = row['BusinessLine']
            for i in range(len(business_lines) - 1):
                transitions.append((business_lines[i], business_lines[i + 1]))
        
        if not transitions:
            return go.Figure()
        
        transition_counts = Counter(transitions)
        business_lines = sorted(list(set([bl for pair in transitions for bl in pair])))
        
        heatmap_data = []
        for from_bl in business_lines:
            row = []
            for to_bl in business_lines:
                count = transition_counts.get((from_bl, to_bl), 0)
                row.append(count)
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=business_lines,
            y=business_lines,
            colorscale='Blues',
            text=heatmap_data,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Customer Journey Transition Heatmap",
            xaxis_title="To Product",
            yaxis_title="From Product",
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating customer journey heatmap: {str(e)}")
        return go.Figure()

def create_cohort_retention_chart(cohort_df):
    """Create cohort retention heatmap."""
    try:
        if cohort_df.empty:
            return go.Figure()
        
        # Convert Period objects to strings for JSON serialization
        cohort_df_copy = cohort_df.copy()
        cohort_df_copy['CohortMonth'] = cohort_df_copy['CohortMonth'].astype(str)
        cohort_df_copy['Period'] = cohort_df_copy['Period'].astype(str)
        
        cohort_pivot = cohort_df_copy.pivot(index='CohortMonth', columns='Period', values='RetentionRate')
        
        fig = px.imshow(
            cohort_pivot,
            title="Customer Retention Cohort Analysis",
            labels=dict(x="Months After First Purchase", y="Cohort Month", color="Retention Rate"),
            color_continuous_scale="Reds",
            aspect="auto"
        )
        
        fig.update_layout(
            height=500,
            xaxis_title="Months After First Purchase",
            yaxis_title="Cohort Month"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating cohort retention chart: {str(e)}")
        return go.Figure()

@st.cache_data(show_spinner="Loading data...")
def load_df(upload):
    """Load data from uploaded file."""
    try:
        if upload.name.endswith('.csv'):
            df = pd.read_csv(upload)
        elif upload.name.endswith('.xlsx') or upload.name.endswith('.xls'):
            df = pd.read_excel(upload, sheet_name=0, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

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
#                    MAIN APPLICATION
# --------------------------------------------------------------

def main():
    st.title("🚀 Advanced Cross-Sell Analytics Dashboard")
    st.markdown("*Comprehensive customer behavior and cross-selling intelligence*")
    
    # Try to load default file first
    default_data, default_file_path = load_default_data()
    
    # Sidebar
    st.sidebar.header("📁 Data Source")
    
    # Show auto-loaded file status
    if default_data is not None:
        st.sidebar.success(f"✅ **Auto-loaded:** sale_data.xlsx")
        st.sidebar.info(f"📍 **Location:** {default_file_path}")
        st.sidebar.markdown("---")
        
        use_default = st.sidebar.button("🔄 Use Auto-Loaded Data", type="primary")
        st.sidebar.markdown("**OR**")
        
        file = st.sidebar.file_uploader(
            "Upload different data file", 
            type=["xlsx", "csv", "xls"],
            help="Upload a different sales data file to override the auto-loaded one"
        )
        
        # Determine which data to use
        if use_default or file is None:
            raw = default_data
            data_source = "Auto-loaded file"
        else:
            raw = load_df(file)
            data_source = "Uploaded file"
            
    else:
        st.sidebar.warning("⚠️ **sale_data.xlsx not found in script directory**")
        st.sidebar.markdown("Please upload your data file manually:")
        
        file = st.sidebar.file_uploader(
            "Upload your sale_data.xlsx file", 
            type=["xlsx", "csv", "xls"],
            help="Upload your sales data file - columns will be mapped automatically"
        )
        
        if file:
            raw = load_df(file)
            data_source = "Uploaded file"
        else:
            raw = None
            data_source = None
    
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
    
    if raw is None:
        st.info("👋 **Welcome!** Please ensure sale_data.xlsx is in the same folder as this script, or upload your data file manually.")
        
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
        
        ## 📁 **File Setup Instructions:**
        1. Place your `sale_data.xlsx` file in the same folder as this Python script
        2. The app will automatically detect and load it
        3. Alternatively, use the file uploader in the sidebar
        """)
        
        return
    
    # Display data source info
    st.sidebar.success(f"✅ Data loaded from: {data_source}")
    st.sidebar.success(f"📊 Rows: {len(raw):,}")
    
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
    
    # Time-based filtering controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏰ Time Period Analysis")
    
    # Get date range from data
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    # Time period selector
    time_analysis_type = st.sidebar.selectbox(
        "Analysis Time Scope:",
        [
            "All Time",
            "Custom Date Range", 
            "Last 12 Months",
            "Last 6 Months",
            "Last 3 Months",
            "Current Year",
            "Previous Year",
            "Year-over-Year Comparison"
        ],
        help="Select time period for analysis"
    )
    
    # Custom date range selector
    if time_analysis_type == "Custom Date Range":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        if start_date > end_date:
            st.sidebar.error("Start date must be before end date")
            start_date = min_date
            end_date = max_date
    else:
        start_date = min_date
        end_date = max_date
    
    # Apply time filtering
    current_date = df['Date'].max()
    
    if time_analysis_type == "Last 12 Months":
        filtered_start = current_date - pd.DateOffset(months=12)
        df_filtered = df[df['Date'] >= filtered_start]
        analysis_period = "Last 12 Months"
    elif time_analysis_type == "Last 6 Months":
        filtered_start = current_date - pd.DateOffset(months=6)
        df_filtered = df[df['Date'] >= filtered_start]
        analysis_period = "Last 6 Months"
    elif time_analysis_type == "Last 3 Months":
        filtered_start = current_date - pd.DateOffset(months=3)
        df_filtered = df[df['Date'] >= filtered_start]
        analysis_period = "Last 3 Months"
    elif time_analysis_type == "Current Year":
        current_year = current_date.year
        df_filtered = df[df['Date'].dt.year == current_year]
        analysis_period = f"Year {current_year}"
    elif time_analysis_type == "Previous Year":
        previous_year = current_date.year - 1
        df_filtered = df[df['Date'].dt.year == previous_year]
        analysis_period = f"Year {previous_year}"
    elif time_analysis_type == "Custom Date Range":
        df_filtered = df[
            (df['Date'].dt.date >= start_date) & 
            (df['Date'].dt.date <= end_date)
        ]
        analysis_period = f"{start_date} to {end_date}"
    else:  # All Time
        df_filtered = df
        analysis_period = "All Time"
    
    # Show current filter status
    st.sidebar.info(f"📊 **Current Analysis Period:** {analysis_period}")
    st.sidebar.info(f"📈 **Records in Period:** {len(df_filtered):,}")
    
    # Temporal granularity selector
    st.sidebar.markdown("---")
    temporal_granularity = st.sidebar.selectbox(
        "Time Granularity for Charts:",
        ["Monthly", "Quarterly", "Yearly"],
        index=0,
        help="How to group time-based analyses"
    )
    
    # Additional filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Additional Filters")
    
    if 'Industry' in df.columns:
        industries = st.sidebar.multiselect(
            "Filter by Industry:",
            options=sorted(df['Industry'].unique()),
            default=sorted(df['Industry'].unique())
        )
        if industries:
            df_filtered = df_filtered[df_filtered['Industry'].isin(industries)]
    
    if 'Sector' in df.columns:
        sectors = st.sidebar.multiselect(
            "Filter by Sector:",
            options=sorted(df['Sector'].unique()),
            default=sorted(df['Sector'].unique())
        )
        if sectors:
            df_filtered = df_filtered[df_filtered['Sector'].isin(sectors)]
    
    if df_filtered.empty:
        st.warning(f"No data available for the selected time period: {analysis_period}")
        return
    
    # Show comparison with full dataset if filtered
    if len(df_filtered) < len(df):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Filtered Records", f"{len(df_filtered):,}")
        with col3:
            filter_percentage = len(df_filtered) / len(df) * 100
            st.metric("Data Coverage", f"{filter_percentage:.1f}%")
    
    # Calculate metrics on filtered data
    cross_sell_metrics, customer_journeys = calculate_cross_sell_metrics(df_filtered)
    bl_analysis = analyze_business_line_roles(df_filtered)
    cohort_data = create_cohort_analysis(df_filtered)
    saturation_data, segment_saturation = calculate_market_saturation(df_filtered)
    
    # New: Calculate comprehensive customer analytics on filtered data
    top_customer_analysis = analyze_top_customers(df_filtered)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview", "🔄 Cross-Sell Analysis", "👥 Customer Journey", 
        "📈 Cohort Analysis", "🎯 Top Customers", "🏪 Market Saturation",
        "📅 Temporal Analysis", "🌊 Seasonal Patterns", "💡 Insights"
    ])
    
    with tab1:
        st.header("📊 Cross-Sell Performance Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = cross_sell_metrics.get('cross_sell_success_rate', 0)
            st.metric(
                "Cross-Sell Success Rate",
                f"{success_rate:.1%}",
                help="Percentage of customers who bought multiple products"
            )
        
        with col2:
            total_customers = cross_sell_metrics.get('total_customers', 0)
            cross_sell_customers = cross_sell_metrics.get('cross_sell_customers', 0)
            st.metric(
                "Cross-Sell Customers",
                f"{cross_sell_customers:,} / {total_customers:,}",
                help="Customers who bought multiple products vs total customers"
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
        
        funnel_fig = create_cross_sell_funnel(df_filtered)
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
        
        heatmap_fig = create_customer_journey_heatmap(df_filtered)
        st.plotly_chart(heatmap_fig, use_container_width=True, key="journey_heatmap")
    
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
            bl_df = bl_df.rename(columns={'index': 'Business Line'})
            
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
            
            **How to read it:**
            - **X-axis:** Number of different products customer has bought
            - **Y-axis:** Average total revenue per customer in that group
            - **Higher bars:** More valuable customer segments
            
            **Key insight:** This shows the ROI of cross-selling - typically customers with more products spend significantly more.
            """)
        
        customer_revenue_complexity = []
        for _, row in customer_journeys.iterrows():
            account = row['Account']
            complexity = len(set(row['BusinessLine']))
            total_revenue = df_filtered[df_filtered['Account'] == account]['Revenue'].sum()
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
        
        # Style the dataframe with conditional formatting
        try:
            # Try to use background_gradient if matplotlib is available
            import matplotlib
            styled_journey_df = journey_df.style.format({
                'Customer Count': '{:,}',
                'Percentage': '{:.1f}%'
            }).background_gradient(subset=['Customer Count'], cmap='Blues')
        except ImportError:
            # Fallback to simple styling without matplotlib
            styled_journey_df = journey_df.style.format({
                'Customer Count': '{:,}',
                'Percentage': '{:.1f}%'
            }).set_properties(**{
                'background-color': '#f8f9fa',
                'color': 'black',
                'border': '1px solid #dee2e6'
            })
        
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
                cohort_pivot_products,
                title="Product Portfolio Expansion by Cohort",
                labels=dict(x="Months After First Purchase", y="Cohort Month", color="Avg Products per Customer"),
                color_continuous_scale="Greens",
                aspect="auto"
            )
            
            fig_products.update_layout(
                height=500,
                xaxis_title="Months After First Purchase",
                yaxis_title="Cohort Month"
            )
            
            st.plotly_chart(fig_products, use_container_width=True, key="cohort_products")
        else:
            st.warning("Insufficient data for cohort analysis. Need at least 2 months of data with customer repeat purchases.")
    
    with tab5:
        st.header("🎯 Top Customer Analysis")
        
        if not top_customer_analysis.empty:
            # Top customers by revenue
            st.subheader("💰 Top Customers by Revenue")
            
            top_10_customers = top_customer_analysis.head(10)
            
            fig_top_customers = px.bar(
                top_10_customers,
                x='Account',
                y='Total_Revenue',
                title="Top 10 Customers by Total Revenue",
                labels={'Total_Revenue': 'Total Revenue ($)', 'Account': 'Customer'},
                color='Total_Revenue',
                color_continuous_scale="Viridis"
            )
            
            fig_top_customers.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_top_customers, use_container_width=True, key="top_customers_revenue")
            
            # Customer lifetime value analysis
            st.subheader("⏱️ Customer Lifetime Value Analysis")
            
            fig_lifetime = px.scatter(
                top_customer_analysis,
                x='Customer_Lifetime_Days',
                y='Total_Revenue',
                size='Transaction_Count',
                color='Unique_Products',
                hover_data=['Account'],
                title="Customer Lifetime Value vs Customer Lifetime",
                labels={
                    'Customer_Lifetime_Days': 'Customer Lifetime (Days)',
                    'Total_Revenue': 'Total Revenue ($)',
                    'Transaction_Count': 'Number of Transactions',
                    'Unique_Products': 'Number of Products'
                }
            )
            
            fig_lifetime.update_layout(height=500)
            st.plotly_chart(fig_lifetime, use_container_width=True, key="customer_lifetime")
            
            # Top customers table
            st.subheader("📋 Top Customers Detailed View")
            
            display_columns = ['Account', 'Total_Revenue', 'Transaction_Count', 'Unique_Products', 
                             'Customer_Lifetime_Days', 'Revenue_Per_Day']
            
            formatted_top_customers = top_customer_analysis[display_columns].head(20).style.format({
                'Total_Revenue': '${:,.0f}',
                'Transaction_Count': '{:,.0f}',
                'Unique_Products': '{:,.0f}',
                'Customer_Lifetime_Days': '{:,.0f}',
                'Revenue_Per_Day': '${:,.2f}'
            })
            
            st.dataframe(formatted_top_customers, use_container_width=True)
        else:
            st.warning("No customer data available for analysis.")
    
    with tab6:
        st.header("🏪 Market Saturation Analysis")
        
        if not saturation_data.empty:
            # Business line saturation
            st.subheader("📊 Business Line Market Saturation")
            
            fig_saturation = px.bar(
                saturation_data,
                x='BusinessLine',
                y='Unique_Customers',
                title="Customer Penetration by Business Line",
                labels={'Unique_Customers': 'Number of Customers', 'BusinessLine': 'Business Line'},
                color='Total_Revenue',
                color_continuous_scale="Blues"
            )
            
            fig_saturation.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_saturation, use_container_width=True, key="business_line_saturation")
            
            # Revenue vs customer count scatter
            st.subheader("💰 Revenue vs Customer Penetration")
            
            fig_revenue_customers = px.scatter(
                saturation_data,
                x='Unique_Customers',
                y='Total_Revenue',
                size='Unique_Customers',
                color='BusinessLine',
                hover_data=['BusinessLine'],
                title="Revenue vs Customer Penetration by Business Line",
                labels={
                    'Unique_Customers': 'Number of Customers',
                    'Total_Revenue': 'Total Revenue ($)',
                    'BusinessLine': 'Business Line'
                }
            )
            
            fig_revenue_customers.update_layout(height=500)
            st.plotly_chart(fig_revenue_customers, use_container_width=True, key="revenue_customers")
            
            # Saturation table
            st.subheader("📋 Market Saturation Summary")
            
            formatted_saturation = saturation_data.style.format({
                'Unique_Customers': '{:,}',
                'Total_Revenue': '${:,.0f}'
            })
            
            st.dataframe(formatted_saturation, use_container_width=True)
        else:
            st.warning("No market saturation data available.")
    
    with tab7:
        st.header("📅 Temporal Analysis")
        
        # Calculate temporal metrics
        temporal_metrics = create_temporal_analysis(df_filtered, temporal_granularity)
        
        if not temporal_metrics.empty:
            st.subheader(f"📈 {temporal_granularity} Performance Trends")
            
            with st.expander("ℹ️ What does this temporal analysis show?", expanded=False):
                st.markdown(f"""
                **{temporal_granularity} Performance Trends** show how key metrics evolve over time in your selected period.
                
                **Key Insights:**
                - **Revenue Trends:** Identify growth patterns, seasonality, and anomalies
                - **Customer Acquisition:** Track new vs returning customer patterns
                - **Cross-Sell Evolution:** See how cross-selling improves over time
                - **Period-over-Period Growth:** Understand momentum and velocity
                
                **Strategic Value:**
                - **Forecast planning** based on historical trends
                - **Seasonal optimization** of marketing and sales efforts  
                - **Performance benchmarking** across time periods
                - **Early warning signals** for declining metrics
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue trend over time
                fig_revenue_trend = px.line(
                    temporal_metrics,
                    x='TimeGroup',
                    y='Total_Revenue',
                    title=f"{temporal_granularity} Revenue Trend",
                    labels={'Total_Revenue': 'Revenue ($)', 'TimeGroup': f'{temporal_granularity} Period'}
                )
                fig_revenue_trend.update_traces(line_width=3, line_color='#1f77b4')
                fig_revenue_trend.update_layout(height=400)
                st.plotly_chart(fig_revenue_trend, use_container_width=True, key="temporal_revenue_trend")
            
            with col2:
                # Cross-sell rate trend
                if 'Cross_Sell_Rate' in temporal_metrics.columns:
                    fig_cross_sell_trend = px.line(
                        temporal_metrics,
                        x='TimeGroup',
                        y='Cross_Sell_Rate',
                        title=f"{temporal_granularity} Cross-Sell Rate Trend",
                        labels={'Cross_Sell_Rate': 'Cross-Sell Rate (%)', 'TimeGroup': f'{temporal_granularity} Period'}
                    )
                    fig_cross_sell_trend.update_traces(line_width=3, line_color='#2ca02c')
                    fig_cross_sell_trend.update_layout(height=400, yaxis_tickformat='.1%')
                    st.plotly_chart(fig_cross_sell_trend, use_container_width=True, key="temporal_cross_sell_trend")
            
            # Multi-metric dashboard
            st.subheader(f"📊 {temporal_granularity} Multi-Metric Dashboard")
            
            # Create subplots for multiple metrics
            fig_multi = go.Figure()
            
            # Revenue (primary axis)
            fig_multi.add_trace(go.Scatter(
                x=temporal_metrics['TimeGroup'],
                y=temporal_metrics['Total_Revenue'],
                name='Revenue',
                line=dict(color='#1f77b4', width=3),
                yaxis='y'
            ))
            
            # Customers (secondary axis)
            fig_multi.add_trace(go.Scatter(
                x=temporal_metrics['TimeGroup'],
                y=temporal_metrics['Unique_Customers'],
                name='Unique Customers',
                line=dict(color='#ff7f0e', width=3),
                yaxis='y2'
            ))
            
            # Cross-sell rate (tertiary axis) if available
            if 'Cross_Sell_Rate' in temporal_metrics.columns:
                fig_multi.add_trace(go.Scatter(
                    x=temporal_metrics['TimeGroup'],
                    y=temporal_metrics['Cross_Sell_Rate'] * 100,
                    name='Cross-Sell Rate (%)',
                    line=dict(color='#2ca02c', width=3),
                    yaxis='y3'
                ))
            
            fig_multi.update_layout(
                title=f"{temporal_granularity} Performance Dashboard",
                xaxis_title=f"{temporal_granularity} Period",
                yaxis=dict(title="Revenue ($)", side="left"),
                yaxis2=dict(title="Customers", side="right", overlaying="y"),
                yaxis3=dict(title="Cross-Sell Rate (%)", side="right", overlaying="y", position=0.95),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_multi, use_container_width=True, key="temporal_multi_metric")
        else:
            st.warning("Insufficient data for temporal analysis.")
    
    with tab8:
        st.header("🌊 Seasonal Patterns")
        
        # Calculate seasonal patterns
        monthly_patterns, quarterly_patterns, yearly_comparison = analyze_seasonal_patterns(df_filtered)
        
        if not monthly_patterns.empty:
            st.subheader("📅 Monthly Seasonal Patterns")
            
            fig_monthly = px.line(
                monthly_patterns,
                x='Month',
                y='Total_Revenue',
                title="Monthly Revenue Patterns",
                labels={'Total_Revenue': 'Revenue ($)', 'Month': 'Month'},
                markers=True
            )
            
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_patterns")
            
            # Quarterly patterns
            if not quarterly_patterns.empty:
                st.subheader("📊 Quarterly Patterns")
                
                fig_quarterly = px.bar(
                    quarterly_patterns,
                    x='Quarter',
                    y='Total_Revenue',
                    title="Quarterly Revenue Distribution",
                    labels={'Total_Revenue': 'Revenue ($)', 'Quarter': 'Quarter'},
                    color='Total_Revenue',
                    color_continuous_scale="Blues"
                )
                
                fig_quarterly.update_layout(height=400)
                st.plotly_chart(fig_quarterly, use_container_width=True, key="quarterly_patterns")
            
            # Yearly comparison
            if not yearly_comparison.empty and len(yearly_comparison) > 1:
                st.subheader("📈 Year-over-Year Comparison")
                
                fig_yearly = px.bar(
                    yearly_comparison,
                    x='Year',
                    y='Total_Revenue',
                    title="Year-over-Year Revenue Comparison",
                    labels={'Total_Revenue': 'Revenue ($)', 'Year': 'Year'},
                    color='Total_Revenue',
                    color_continuous_scale="Greens"
                )
                
                fig_yearly.update_layout(height=400)
                st.plotly_chart(fig_yearly, use_container_width=True, key="yearly_comparison")
        else:
            st.warning("Insufficient data for seasonal analysis.")
    
    with tab9:
        st.header("💡 Key Insights & Recommendations")
        
        # Performance insights
        st.subheader("🎯 Performance Insights")
        
        success_rate = cross_sell_metrics.get('cross_sell_success_rate', 0)
        
        if success_rate > 0.5:
            st.success(f"""
            🎉 **EXCELLENT CROSS-SELL PERFORMANCE** {success_rate:.1%} of customers purchase multiple products - well above industry average!
            """)
        elif success_rate > 0.3:
            st.warning(f"""
            ⚠️ **MODERATE CROSS-SELL SUCCESS** {success_rate:.1%} cross-sell rate shows room for improvement. Target: 50%+
            """)
        else:
            st.error(f"""
            🚨 **LOW CROSS-SELL PERFORMANCE - URGENT ACTION NEEDED** Only {success_rate:.1%} of customers buy multiple products. Major revenue opportunity!
            """)
        
        avg_time = cross_sell_metrics.get('avg_time_to_cross_sell')
        if avg_time and avg_time < 90:
            st.info(f"""
            ⚡ **FAST CROSS-SELL CONVERSION** Average time of {avg_time:.0f} days suggests strong product synergy and customer satisfaction.
            """)
        elif avg_time and avg_time > 180:
            st.warning(f"""
            🐌 **SLOW CROSS-SELL CONVERSION** {avg_time:.0f} days average suggests need for more proactive cross-sell campaigns and better timing.
            """)
        
        # Strategic recommendations
        st.subheader("🚀 Strategic Recommendations")
        
        st.markdown("""
        ### **Immediate Actions:**
        
        **🎯 High-Priority Opportunities:**
        - **Target single-product customers** with personalized cross-sell campaigns
        - **Leverage strong donor products** as entry points in sales pitches
        - **Bundle complementary products** based on journey analysis
        
        **📈 Growth Strategies:**
        - **Develop product combinations** from successful customer journeys
        - **Optimize timing** for cross-sell campaigns based on conversion patterns
        - **Create loyalty programs** to encourage multi-product adoption
        
        **🔍 Continuous Improvement:**
        - **Monitor cohort retention** to identify at-risk customers
        - **Track seasonal patterns** for optimal campaign timing
        - **Benchmark performance** against industry standards
        
        ### **Long-term Strategic Initiatives:**
        
        **🔬 Develop new product combinations** based on journey pattern insights
        **🤖 Build predictive models** to identify high-potential cross-sell prospects
        **📈 Establish benchmarks** and regular review cycles for cross-sell performance
        """)

if __name__ == "__main__":
    main() 