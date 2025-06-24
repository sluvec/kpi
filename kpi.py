# Workplace Services KPI Dashboard (v4.2 - Auto-Mapped Columns)
# =======================================
# Author: Claude • June 2025
# Description: Version with automatic column mapping for sale_data.xlsx
# ---------------------------------------

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
#                    AUTOMATIC COLUMN MAPPING
# --------------------------------------------------------------
# Based on the actual structure of sale_data.xlsx
COLUMN_MAPPING = {
    'customer': 'Client Name',
    'category': 'Business Line', 
    'year': 'Year',
    'month': 'Month',
    'sales': 'Value'
}

# Additional columns available for analysis
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

# Run dependency check
check_dependencies()

# Import dependencies after checking
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# --------------------------------------------------------------
#                       STREAMLIT CONFIG
# --------------------------------------------------------------
st.set_page_config(
    page_title="Workplace Services KPI Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --------------------------------------------------------------
#                       HELPER FUNCTIONS
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
    
    # Check if all required columns exist
    required_columns = list(COLUMN_MAPPING.values())
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
        issues.append(f"Available columns: {', '.join(df.columns)}")
        return False, issues
    
    # Check data types and content
    numeric_cols = [COLUMN_MAPPING['year'], COLUMN_MAPPING['month'], COLUMN_MAPPING['sales']]
    for col in numeric_cols:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            if non_numeric > len(df) * 0.1:  # More than 10% non-numeric
                issues.append(f"Column '{col}' has {non_numeric} non-numeric values")
    
    # Check year range
    try:
        year_col = pd.to_numeric(df[COLUMN_MAPPING['year']], errors='coerce')
        if year_col.min() < 2000 or year_col.max() > 2030:
            issues.append(f"Year values seem unusual: {year_col.min()} to {year_col.max()}")
    except:
        pass
    
    # Check month range
    try:
        month_col = pd.to_numeric(df[COLUMN_MAPPING['month']], errors='coerce')
        if month_col.min() < 1 or month_col.max() > 12:
            issues.append(f"Month values outside 1-12 range: {month_col.min()} to {month_col.max()}")
    except:
        pass
    
    return len(issues) == 0, issues

@st.cache_data(show_spinner="Preparing data...")
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data with automatic column mapping."""
    try:
        # Create a copy and rename columns to standard names
        processed_df = df.copy()
        
        # Auto-map columns to standard names
        rename_mapping = {
            COLUMN_MAPPING['customer']: 'Account',
            COLUMN_MAPPING['category']: 'BusinessLine',
            COLUMN_MAPPING['year']: 'Year',
            COLUMN_MAPPING['month']: 'Month',
            COLUMN_MAPPING['sales']: 'Revenue'
        }
        
        processed_df = processed_df.rename(columns=rename_mapping)
        
        # Keep additional columns for extended analysis
        for col in ADDITIONAL_COLUMNS:
            if col in df.columns:
                processed_df[col] = df[col]
        
        # Convert to appropriate data types
        processed_df['Revenue'] = pd.to_numeric(processed_df['Revenue'], errors='coerce')
        processed_df['Year'] = pd.to_numeric(processed_df['Year'], errors='coerce')
        processed_df['Month'] = pd.to_numeric(processed_df['Month'], errors='coerce')
        
        # Remove rows with missing critical data
        processed_df.dropna(subset=['Revenue', 'Account', 'BusinessLine', 'Year', 'Month'], inplace=True)
        
        # Add date columns
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

@st.cache_data(show_spinner="Calculating KPIs...")
def build_kpis(_df: pd.DataFrame):
    """Calculate portfolio KPIs and business-line metrics."""
    try:
        Acct, Line, Val = "Account", "BusinessLine", "Revenue"
        
        breadth = _df.groupby(["YM", Acct]).agg(
            Breadth=(Line, "nunique"), 
            AccRev=(Val, "sum")
        ).reset_index()
        
        monthly = breadth.groupby("YM").agg(
            ActiveAccounts=(Acct, "nunique"),
            TotalRevenue=("AccRev", "sum"),
            AvgBreadth=("Breadth", "mean")
        ).reset_index()
        
        if not monthly.empty:
            monthly["rwABI"] = breadth.groupby("YM").apply(
                lambda g: np.average(g.Breadth, weights=g.AccRev) 
                if not g.empty and g.AccRev.sum() > 0 else 0
            ).values
            
            multi_service_counts = breadth[breadth.Breadth >= 2].groupby("YM")[Acct].nunique()
            monthly = monthly.set_index('YM').join(
                multi_service_counts.rename('MSP_count')
            ).fillna(0).reset_index()
            
            monthly['MSP'] = np.where(
                monthly['ActiveAccounts'] > 0,
                monthly['MSP_count'] / monthly['ActiveAccounts'],
                0
            )
            
            monthly["AMRA"] = np.where(
                monthly['ActiveAccounts'] > 0,
                monthly.TotalRevenue / monthly.ActiveAccounts,
                0
            )
        
        if not _df.empty and 'Date' in _df.columns:
            last12 = _df[_df["Date"] >= _df["Date"].max() - pd.DateOffset(months=11)]
            bl_board = last12.groupby(Line).agg(
                Revenue_total=(Val, "sum"),
                Accounts=(Acct, "nunique")
            ).reset_index()
            
            first_line = _df.sort_values("Date").groupby(Acct)[Line].first().value_counts()
            total_accounts = _df[Acct].nunique()
            bl_board["EntryRatio"] = bl_board[Line].map(first_line).fillna(0) / max(total_accounts, 1)
            
            bl_board.sort_values("EntryRatio", ascending=False, inplace=True, na_position='last')
        else:
            bl_board = pd.DataFrame()
        
        return monthly.sort_values("YM"), bl_board
        
    except Exception as e:
        st.error(f"Error calculating KPIs: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def get_customer_journeys(_df: pd.DataFrame):
    """Extract customer journey patterns."""
    try:
        Acct, Line = "Account", "BusinessLine"
        sorted_df = _df.sort_values(["Date", Acct])
        
        first_purchases = sorted_df.groupby(Acct)[Line].first().value_counts()
        last_purchases = sorted_df.groupby(Acct)[Line].last().value_counts()
        
        sequences = sorted_df.groupby(Acct)[Line].apply(list)
        transitions = Counter()
        
        for seq in sequences:
            if len(seq) > 1:
                transitions.update(zip(seq, seq[1:]))
                
        return first_purchases, last_purchases, transitions
        
    except Exception as e:
        st.error(f"Error analyzing customer journeys: {str(e)}")
        return pd.Series(), pd.Series(), Counter()

def create_network_graph(transitions, node_weights, title):
    """Create a network graph."""
    try:
        if not transitions:
            return go.Figure().add_annotation(
                text="No transition data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        G = nx.DiGraph()
        for (src, tgt), weight in transitions.items():
            G.add_edge(src, tgt, weight=weight)
        
        if not G.nodes:
            return go.Figure()

        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_text, node_size = [], [], [], []
        node_weights_max = node_weights.max() if not node_weights.empty and node_weights.max() > 0 else 1
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            weight = node_weights.get(node, 0)
            node_text.append(f"{node}<br>Count: {weight}")
            node_size.append(15 + 50 * (weight / node_weights_max))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_text,
            text=[f"<b>{n}</b>" for n in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=[node_weights.get(n, 0) for n in G.nodes()],
                colorbar=dict(
                    thickness=25, 
                    title=dict(
                        text="Customer<br>Frequency",
                        font=dict(size=13, color="darkblue")
                    ),
                    xanchor='left',
                    x=1.02,
                    y=0.5,
                    yanchor="middle",
                    len=0.75,
                    tickfont=dict(size=10, color="darkblue"),
                    tickformat="d",
                    ticks="outside",
                    ticklen=5,
                    tickwidth=1,
                    tickcolor="darkblue",
                    outlinecolor="darkblue",
                    outlinewidth=1,
                    bgcolor="rgba(240,248,255,0.9)",
                    bordercolor="darkblue",
                    borderwidth=1
                )
            )
        )

        annotations = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotations.append(dict(
                ax=x0, ay=y0, axref='x', ayref='y',
                x=x1 * 0.8 + x0 * 0.2, y=y1 * 0.8 + y0 * 0.2, 
                xref='x', yref='y',
                showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=1,
                arrowcolor='#888'
            ))

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(size=16)
                ), 
                showlegend=False, 
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=annotations,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig
        
    except Exception as e:
        st.error(f"Error creating network graph: {str(e)}")
        return go.Figure()

# --------------------------------------------------------------
#                            MAIN APP
# --------------------------------------------------------------
def main():
    st.title("📊 Workplace Services KPI Dashboard")
    st.markdown("*Advanced analytics for sales performance and customer journey insights*")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    file = st.sidebar.file_uploader(
        "Upload your sale_data.xlsx file", 
        type=["xlsx", "csv", "xls"],
        help="Upload your sales data file - columns will be mapped automatically"
    )
    
    # Show expected file format
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
        st.info("👋 **Welcome!** Please upload your sale_data.xlsx file using the sidebar to get started.")
        
        with st.expander("📝 Sample Data Structure"):
            sample_data = pd.DataFrame({
                'Year': [2023, 2023, 2024, 2024],
                'Month': [1, 2, 1, 2], 
                'Client Name': ["Company A", "Company B", "Company A", "Company C"],
                'Business Line': ["Removals", "Storage", "Interiors", "Removals"],
                'Industry': ["Retail", "Finance", "Tech", "Healthcare"],
                'Sector': ["Private", "Public", "Private", "Private"],
                'Type': ["Company", "Individual", "Company", "Individual"],
                'Value': [15000, 8500, 12000, 22000]
            })
            st.dataframe(sample_data)
        
        return
    
    # Load data
    raw = load_df(file)
    if raw is None:
        return
    
    st.sidebar.success(f"✅ File loaded: {len(raw):,} rows")
    
    # Validate automatic mapping
    is_valid, issues = validate_auto_mapping(raw)
    
    if not is_valid:
        st.error("**Auto-mapping validation failed:**")
        for issue in issues:
            st.error(f"• {issue}")
        
        with st.expander("🔍 View Raw Data Structure"):
            st.dataframe(raw.head(10))
            st.write("**Available columns:**", list(raw.columns))
        return
    
    # Process data with auto-mapping
    df = prepare_data(raw)
    
    if df.empty:
        st.error("No valid data remaining after processing.")
        return
    
    # Time aggregation selector
    periodicity = st.sidebar.selectbox(
        "Time aggregation", 
        ["Monthly", "Quarterly", "Half-yearly", "Yearly"], 
        index=0
    )
    
    # Data summary
    with st.sidebar.expander("📈 Data Summary"):
        st.write(f"**Valid Records:** {len(df):,}")
        st.write(f"**Accounts:** {df['Account'].nunique():,}")
        st.write(f"**Business Lines:** {df['BusinessLine'].nunique():,}")
        st.write(f"**Date Range:** {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
        st.write(f"**Total Revenue:** ${df['Revenue'].sum():,.2f}")
        
        # Show unique business lines
        st.write("**Business Lines:**")
        for bl in sorted(df['BusinessLine'].unique()):
            st.write(f"• {bl}")
    
    # Additional filters based on extra columns
    if 'Industry' in df.columns:
        industries = st.sidebar.multiselect(
            "Filter by Industry:",
            options=sorted(df['Industry'].unique()),
            default=sorted(df['Industry'].unique()),
            help="Select industries to include in analysis"
        )
        if industries:
            df = df[df['Industry'].isin(industries)]
    
    if 'Sector' in df.columns:
        sectors = st.sidebar.multiselect(
            "Filter by Sector:",
            options=sorted(df['Sector'].unique()),
            default=sorted(df['Sector'].unique()),
            help="Select sectors to include in analysis"
        )
        if sectors:
            df = df[df['Sector'].isin(sectors)]
    
    # Calculate KPIs
    monthly, board = build_kpis(df)
    
    # Create tabs
    tab_port, tab_lines, tab_journeys, tab_breakdown = st.tabs([
        "📊 Portfolio KPIs", 
        "🏢 Business-Line Board", 
        "🛤️ Customer Journeys",
        "📋 Data Breakdown"
    ])
    
    with tab_port:
        st.header("📊 Portfolio KPIs")
        
        if not monthly.empty:
            latest = monthly.iloc[-1]
            
            cols = st.columns(4)
            cols[0].metric(
                "Total Revenue (Latest Month)", 
                f"${latest['TotalRevenue']:,.2f}",
                help="Total revenue for the most recent month"
            )
            cols[1].metric(
                "Active Accounts", 
                f"{latest['ActiveAccounts']:,}",
                help="Number of accounts with purchases in the latest month"
            )
            cols[2].metric(
                "Avg. Basket Index (ABI)", 
                f"{latest['AvgBreadth']:.2f}",
                help="Average number of different services per account"
            )
            cols[3].metric(
                "Multi-Service Penetration", 
                f"{latest['MSP']:.1%}",
                help="Percentage of accounts using 2+ services"
            )
            
            st.subheader("📈 Revenue Trend")
            fig_revenue = px.line(
                monthly, x='YM', y='TotalRevenue',
                title="Monthly Revenue Trend",
                labels={'TotalRevenue': 'Revenue ($)', 'YM': 'Month'}
            )
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True, key="revenue_trend")
            
            col1, col2 = st.columns(2)
            with col1:
                fig_accounts = px.line(
                    monthly, x='YM', y='ActiveAccounts',
                    title="Active Accounts Over Time"
                )
                st.plotly_chart(fig_accounts, use_container_width=True, key="accounts_trend")
            
            with col2:
                fig_msp = px.line(
                    monthly, x='YM', y='MSP',
                    title="Multi-Service Penetration Rate"
                )
                fig_msp.update_traces(line_color='green')
                fig_msp.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_msp, use_container_width=True, key="msp_trend")
        else:
            st.warning("No KPI data available to display.")
    
    with tab_lines:
        st.header("🏢 Business-Line Board")
        st.markdown("*Performance metrics for the last 12 months*")
        
        if not board.empty:
            formatted_board = board.style.format({
                'Revenue_total': '${:,.2f}',
                'EntryRatio': '{:.1%}', 
                'Accounts': '{:,}'
            })
            st.dataframe(formatted_board, use_container_width=True)
            
            st.subheader("🔄 Cross-Sell Lift Matrix")
            st.info("""
            **How to read:** A value of '1.5' in cell (Y, X) means customers who bought product X 
            are 50% more likely to buy product Y than the average customer. Values > 1 indicate synergy.
            """)
            
            try:
                pivot = df.assign(flag=1).pivot_table(
                    index="Account", 
                    columns="BusinessLine", 
                    values="flag", 
                    aggfunc="max", 
                    fill_value=0
                )
                
                if not pivot.empty:
                    prob_b = pivot.sum() / len(pivot)
                    prob_a_and_b = (pivot.T @ pivot) / len(pivot)
                    
                    prob_b_conditional_a = prob_a_and_b.divide(
                        prob_b.replace(0, np.nan), axis='index'
                    )
                    lift = prob_b_conditional_a.divide(
                        prob_b.replace(0, np.nan), axis='columns'
                    )
                    
                    np.fill_diagonal(lift.values, np.nan)
                    
                    fig_lift = px.imshow(
                        lift, 
                        text_auto=".2f", 
                        aspect="auto",
                        color_continuous_scale="RdYlGn", 
                        origin='lower',
                        labels=dict(x="Then Buy...", y="If They Have...", color="Lift"),
                        title="Cross-Sell Lift: Likelihood to Buy B Given A"
                    )
                    
                    # Enhanced colorbar settings with custom labels
                    fig_lift.update_layout(
                        height=600,
                        coloraxis_colorbar=dict(
                            title=dict(
                                text="Cross-Sell<br>Lift Factor",
                                font=dict(size=14, color="black")
                            ),
                            tickfont=dict(size=11, color="black"),
                            thickness=25,
                            len=0.85,
                            x=1.02,
                            xanchor="left",
                            y=0.5,
                            yanchor="middle",
                            tickmode="array",
                            tickvals=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                            ticktext=["0.5x<br>Weak", "1.0x<br>Normal", "1.5x<br>Strong", "2.0x<br>Very Strong", "2.5x<br>Excellent", "3.0x<br>Outstanding"],
                            ticks="outside",
                            tickcolor="darkgray",
                            ticklen=6,
                            tickwidth=2,
                            outlinecolor="black",
                            outlinewidth=1,
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="black",
                            borderwidth=1
                        )
                    )
                    st.plotly_chart(fig_lift, use_container_width=True, key="cross_sell_matrix")
                else:
                    st.warning("Insufficient data for cross-sell analysis.")
                    
            except Exception as e:
                st.error(f"Error creating cross-sell matrix: {str(e)}")
        else:
            st.warning("No business line data available.")
    
    with tab_journeys:
        st.header("🛤️ Customer Purchase Journeys")
        
        try:
            first_purchases, last_purchases, transitions = get_customer_journeys(df)
            
            if transitions:
                st.subheader("🌐 Customer Flow Network")
                st.info("""
                These network diagrams show how customers move between business lines. 
                Circle size indicates how often each line serves as a starting or ending point.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_net_first = create_network_graph(
                        transitions, first_purchases, 
                        "Entry Points (First Purchase Patterns)"
                    )
                    st.plotly_chart(fig_net_first, use_container_width=True, key="network_first")
                
                with col2:
                    fig_net_last = create_network_graph(
                        transitions, last_purchases, 
                        "Exit Points (Last Purchase Patterns)"
                    )
                    st.plotly_chart(fig_net_last, use_container_width=True, key="network_last")
                
                st.subheader("🔄 Top Purchase Transitions")
                st.info("Simplified view showing the 10 most common purchase sequences.")
                
                if transitions:
                    top_10_transitions = transitions.most_common(10)
                    if top_10_transitions:
                        source_nodes, target_nodes, values = [], [], []
                        for (src, tgt), val in top_10_transitions:
                            source_nodes.append(src)
                            target_nodes.append(tgt)
                            values.append(val)
                        
                        all_nodes = sorted(list(set(source_nodes + target_nodes)))
                        node_map = {node: i for i, node in enumerate(all_nodes)}
                        
                        sankey_fig = go.Figure(go.Sankey(
                            node=dict(label=all_nodes, pad=25, thickness=20, color='lightblue'),
                            link=dict(
                                source=[node_map[s] for s in source_nodes],
                                target=[node_map[t] for t in target_nodes],
                                value=values
                            )
                        ))
                        sankey_fig.update_layout(title_text="Top 10 Customer Transitions", height=500)
                        st.plotly_chart(sankey_fig, use_container_width=True, key="sankey_transitions")
            else:
                st.warning("No customer journey transitions found in the data.")
                
        except Exception as e:
            st.error(f"Error analyzing customer journeys: {str(e)}")
    
    with tab_breakdown:
        st.header("📋 Data Breakdown & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue by Business Line")
            revenue_by_line = df.groupby('BusinessLine')['Revenue'].sum().sort_values(ascending=False)
            fig_pie = px.pie(
                values=revenue_by_line.values,
                names=revenue_by_line.index,
                title="Revenue Distribution by Business Line",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="revenue_pie")
        
        with col2:
            if 'Industry' in df.columns:
                st.subheader("🏭 Revenue by Industry")
                revenue_by_industry = df.groupby('Industry')['Revenue'].sum().sort_values(ascending=False)
                fig_bar = px.bar(
                    x=revenue_by_industry.values,
                    y=revenue_by_industry.index,
                    orientation='h',
                    title="Revenue by Industry",
                    labels={'x': 'Revenue ($)', 'y': 'Industry'},
                    color=revenue_by_industry.values,
                    color_continuous_scale="viridis"
                )
                fig_bar.update_layout(
                    coloraxis_colorbar=dict(
                        title=dict(
                            text="Revenue<br>($)",
                            font=dict(size=12, color="black")
                        ),
                        thickness=20,
                        len=0.6,
                        x=1.02,
                        tickformat="$,.0f",
                        tickfont=dict(size=10),
                        ticks="outside",
                        ticklen=4
                    )
                )
                fig_bar.update_traces(
                    hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>'
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="revenue_by_industry")
        
        # Summary table
        st.subheader("📊 Summary Statistics")
        summary_stats = df.groupby('BusinessLine').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Account': 'nunique'
        }).round(2)
        
        summary_stats.columns = ['Total Revenue', 'Avg Revenue', 'Transactions', 'Unique Accounts']
        summary_stats = summary_stats.sort_values('Total Revenue', ascending=False)
        
        formatted_summary = summary_stats.style.format({
            'Total Revenue': '${:,.2f}',
            'Avg Revenue': '${:,.2f}',
            'Transactions': '{:,}',
            'Unique Accounts': '{:,}'
        })
        st.dataframe(formatted_summary, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()