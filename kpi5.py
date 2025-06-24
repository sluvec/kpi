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

        journey_complexity = customer_journeys['BusinessLine'].apply(
            lambda x: len(set(x))).value_counts().sort_index()

        fig_complexity = px.bar(
            x=journey_complexity.index,
            y=journey_complexity.values,
            labels={
    'x': 'Number of Different Products',
     'y': 'Number of Customers'},
            title="Customer Distribution by Journey Complexity",
            color=journey_complexity.values,
            color_continuous_scale="Blues"
        )
        fig_complexity.update_layout(showlegend=False)
        st.plotly_chart(
    fig_complexity,
    use_container_width=True,
     key="journey_complexity")

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
            total_revenue = df_filtered[df_filtered['Account']
                == account]['Revenue'].sum()
            customer_revenue_complexity.append(
                {'Account': account, 'Complexity': complexity, 'Revenue': total_revenue})

        revenue_complexity_df = pd.DataFrame(customer_revenue_complexity)
        avg_revenue_by_complexity = revenue_complexity_df.groupby(
            'Complexity')['Revenue'].mean().reset_index()

        fig_revenue_complexity = px.bar(
            avg_revenue_by_complexity,
            x='Complexity',
            y='Revenue',
            labels={
    'Complexity': 'Number of Different Products',
     'Revenue': 'Average Revenue ($)'},
            title="Average Customer Revenue by Journey Complexity",
            color='Revenue',
            color_continuous_scale="Greens",
            text='Revenue'
        )
        fig_revenue_complexity.update_traces(
    texttemplate='$%{text:,.0f}', textposition='outside')
        fig_revenue_complexity.update_layout(showlegend=False)
        st.plotly_chart(
    fig_revenue_complexity,
    use_container_width=True,
     key="revenue_complexity")

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
            # Remove consecutive duplicates
            lambda x: ' → '.join(list(dict.fromkeys(x)))
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
            fig_matrix.add_shape(type="rect", x0=0, y0=-1, x1=0.5, y1=0, fillcolor="yellow", opacity=0.1, layer="below")  # Question Marks
            fig_matrix.add_shape(type="rect", x0=0.5, y0=-1, x1=1, y1=0, fillcolor="red", opacity=0.1, layer="below")  # Cash Cows  
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
        st.header("📅 Seasonal Intelligence & Temporal Patterns")
        
        st.subheader("🌍 Comprehensive Seasonal Analysis")
        
        with st.expander("ℹ️ What does seasonal analysis reveal?", expanded=False):
            st.markdown("""
            **Seasonal Intelligence** uncovers time-based patterns that drive strategic planning.
            
            **Business Value:**
            - **Budget Planning:** Allocate resources based on seasonal peaks/valleys
            - **Inventory Management:** Predict demand fluctuations  
            - **Marketing Timing:** Launch campaigns during optimal periods
            - **Sales Forecasting:** Accurate revenue predictions
            - **Staff Planning:** Scale team size with seasonal demand
            
            **Pattern Types:**
            - **Monthly Seasonality:** Recurring monthly patterns
            - **Quarterly Cycles:** Business cycle effects
            - **Year-over-Year Growth:** Long-term trend analysis
            - **Cross-Sell Seasonality:** When customers are most receptive
            """)
        
        # Monthly seasonality analysis
        if not monthly_patterns.empty:
            st.subheader("📆 Monthly Performance Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly revenue pattern
                fig_monthly_revenue = px.bar(
                    monthly_patterns,
                    x='Month_Name',
                    y='Total_Revenue',
                    title="Monthly Revenue Seasonality",
                    labels={'Total_Revenue': 'Total Revenue ($)', 'Month_Name': 'Month'},
                    color='Total_Revenue',
                    color_continuous_scale='Blues'
                )
                fig_monthly_revenue.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_monthly_revenue, use_container_width=True, key="monthly_revenue_pattern")
            
            with col2:
                # Monthly customer pattern
                fig_monthly_customers = px.bar(
                    monthly_patterns,
                    x='Month_Name',
                    y='Unique_Customers',
                    title="Monthly Customer Activity",
                    labels={'Unique_Customers': 'Unique Customers', 'Month_Name': 'Month'},
                    color='Unique_Customers',
                    color_continuous_scale='Greens'
                )
                fig_monthly_customers.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_monthly_customers, use_container_width=True, key="monthly_customer_pattern")
            
            # Monthly insights
            peak_revenue_month = monthly_patterns.loc[monthly_patterns['Total_Revenue'].idxmax(), 'Month_Name']
            low_revenue_month = monthly_patterns.loc[monthly_patterns['Total_Revenue'].idxmin(), 'Month_Name']
            revenue_seasonality = (monthly_patterns['Total_Revenue'].max() - monthly_patterns['Total_Revenue'].min()) / monthly_patterns['Total_Revenue'].mean() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Revenue Month", peak_revenue_month)
            with col2:
                st.metric("Lowest Revenue Month", low_revenue_month)
            with col3:
                st.metric("Revenue Seasonality", f"{revenue_seasonality:.1f}%", help="Higher = more seasonal variation")
        
        # Quarterly analysis
        if not quarterly_patterns.empty:
            st.subheader("📊 Quarterly Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quarterly revenue
                fig_quarterly = px.bar(
                    quarterly_patterns,
                    x='Quarter_Name',
                    y='Total_Revenue',
                    title="Quarterly Revenue Distribution",
                    labels={'Total_Revenue': 'Revenue ($)', 'Quarter_Name': 'Quarter'},
                    color='Total_Revenue',
                    color_continuous_scale='Oranges'
                )
                fig_quarterly.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_quarterly, use_container_width=True, key="quarterly_revenue")
            
            with col2:
                # Quarterly efficiency (Revenue per Customer)
                quarterly_patterns['Revenue_Per_Customer'] = quarterly_patterns['Total_Revenue'] / quarterly_patterns['Unique_Customers']
                
                fig_quarterly_efficiency = px.bar(
                    quarterly_patterns,
                    x='Quarter_Name',
                    y='Revenue_Per_Customer',
                    title="Quarterly Revenue per Customer",
                    labels={'Revenue_Per_Customer': 'Revenue per Customer ($)', 'Quarter_Name': 'Quarter'},
                    color='Revenue_Per_Customer',
                    color_continuous_scale='Purples'
                )
                fig_quarterly_efficiency.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_quarterly_efficiency, use_container_width=True, key="quarterly_efficiency")
        
        # Year-over-year analysis
        if not yearly_comparison.empty and len(yearly_comparison) > 1:
            st.subheader("📈 Year-over-Year Growth Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # YoY Revenue Growth
                fig_yoy_revenue = px.bar(
                    yearly_comparison[yearly_comparison['YoY_Revenue_Growth'].notna()],
                    x='Y',
                    y='YoY_Revenue_Growth',
                    title="Year-over-Year Revenue Growth",
                    labels={'YoY_Revenue_Growth': 'Revenue Growth (%)', 'Y': 'Year'},
                    color='YoY_Revenue_Growth',
                    color_continuous_scale='RdYlGn'
                )
                fig_yoy_revenue.add_hline(y=0, line_dash="dash", line_color="black")
                fig_yoy_revenue.update_layout(height=400)
                st.plotly_chart(fig_yoy_revenue, use_container_width=True, key="yoy_revenue_growth")
            
            with col2:
                # YoY Customer Growth
                fig_yoy_customers = px.bar(
                    yearly_comparison[yearly_comparison['YoY_Customer_Growth'].notna()],
                    x='Y',
                    y='YoY_Customer_Growth',
                    title="Year-over-Year Customer Growth",
                    labels={'YoY_Customer_Growth': 'Customer Growth (%)', 'Y': 'Year'},
                    color='YoY_Customer_Growth',
                    color_continuous_scale='RdYlGn'
                )
                fig_yoy_customers.add_hline(y=0, line_dash="dash", line_color="black")
                fig_yoy_customers.update_layout(height=400)
                st.plotly_chart(fig_yoy_customers, use_container_width=True, key="yoy_customer_growth")
            
            # Growth insights
            if 'YoY_Revenue_Growth' in yearly_comparison.columns:
                latest_revenue_growth = yearly_comparison['YoY_Revenue_Growth'].iloc[-1]
                latest_customer_growth = yearly_comparison['YoY_Customer_Growth'].iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latest YoY Revenue Growth", f"{latest_revenue_growth:.1f}%")
                with col2:
                    st.metric("Latest YoY Customer Growth", f"{latest_customer_growth:.1f}%")
                with col3:
                    avg_revenue_growth = yearly_comparison['YoY_Revenue_Growth'].mean()
                    st.metric("Average Revenue Growth", f"{avg_revenue_growth:.1f}%")
        
        # Seasonal actionable insights
        st.subheader("🎯 Seasonal Strategy Recommendations")
        
        if not monthly_patterns.empty:
            # Find best and worst performing months
            best_months = monthly_patterns.nlargest(3, 'Total_Revenue')['Month_Name'].tolist()
            worst_months = monthly_patterns.nsmallest(3, 'Total_Revenue')['Month_Name'].tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🚀 Peak Performance Months")
                for i, month in enumerate(best_months, 1):
                    revenue = monthly_patterns[monthly_patterns['Month_Name'] == month]['Total_Revenue'].iloc[0]
                    st.write(f"**{i}. {month}** - ${revenue:,.0f}")
                
                st.info("**Strategy:** Scale up marketing, inventory, and staff during these months.")
            
            with col2:
                st.markdown("### 📉 Improvement Opportunity Months")
                for i, month in enumerate(worst_months, 1):
                    revenue = monthly_patterns[monthly_patterns['Month_Name'] == month]['Total_Revenue'].iloc[0]
                    st.write(f"**{i}. {month}** - ${revenue:,.0f}")
                
                st.warning("**Strategy:** Focus improvement initiatives, promotions, and customer re-engagement.")
    
    with tab7:
        st.header("🏆 TOP Customer Analytics")
        
        if not top_customer_analysis.empty:
            # Top customers selector
            st.subheader("🎯 Customer Performance Rankings")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                ranking_metric = st.selectbox(
                    "Rank customers by:",
                    [
                        "Total_Revenue",
                        "Revenue_Last_Year", 
                        "Revenue_Last_Quarter",
                        "Revenue_Last_Month",
                        "Avg_Revenue_Per_Transaction",
                        "Cross_Sell_Score",
                        "Elite_Score"
                    ],
                    format_func=lambda x: {
                        "Total_Revenue": "💰 Total Revenue (All Time)",
                        "Revenue_Last_Year": "📅 Revenue (Last Year)",
                        "Revenue_Last_Quarter": "📊 Revenue (Last Quarter)", 
                        "Revenue_Last_Month": "📈 Revenue (Last Month)",
                        "Avg_Revenue_Per_Transaction": "💵 Average Transaction Value",
                        "Cross_Sell_Score": "🔄 Cross-Sell Score",
                        "Elite_Score": "🎖️ Elite Customer Score"
                    }[x]
                )
                
                top_n = st.selectbox(
                    "Show top:",
                    [10, 25, 50, 100, "All"],
                    index=0
                )
            
            with col2:
                # Get top N customers
                if top_n == "All":
                    display_customers = top_customer_analysis.sort_values(ranking_metric, ascending=False)
                else:
                    display_customers = top_customer_analysis.nlargest(top_n, ranking_metric)
                
                # Format for display
                display_columns = [
                    'Account', 'Total_Revenue', 'Revenue_Last_Year', 'Revenue_Last_Quarter', 
                    'Revenue_Last_Month', 'Unique_Business_Lines', 'Transaction_Count',
                    'Cross_Sell_Score', 'Elite_Score'
                ]
                
                available_columns = [col for col in display_columns if col in display_customers.columns]
                display_df = display_customers[available_columns].copy()
                
                # Format the display
                formatted_display = display_df.style.format({
                    'Total_Revenue': '${:,.0f}',
                    'Revenue_Last_Year': '${:,.0f}',
                    'Revenue_Last_Quarter': '${:,.0f}',
                    'Revenue_Last_Month': '${:,.0f}',
                    'Cross_Sell_Score': '{:.2%}',
                    'Elite_Score': '{:.2%}',
                    'Transaction_Count': '{:,}',
                    'Unique_Business_Lines': '{:,}'
                }).background_gradient(subset=[ranking_metric], cmap='Greens')
                
                st.dataframe(formatted_display, use_container_width=True)
            
            # Revenue distribution analysis
            st.subheader("💰 Revenue Distribution Analysis")
            
            with st.expander("ℹ️ What does this analysis show?", expanded=False):
                st.markdown("""
                **Revenue Distribution Analysis** reveals the concentration of revenue among your customer base.
                
                **Key Metrics:**
                - **Pareto Principle (80/20 Rule):** What % of customers generate 80% of revenue
                - **Customer Concentration:** How dependent you are on top customers
                - **Revenue Risk:** Potential impact if top customers leave
                
                **Business Actions:**
                - **High concentration:** Diversify customer base, reduce dependency risk
                - **Low concentration:** Focus on growing key accounts
                - **Identify patterns:** What makes top customers successful
                """)
            
            # Calculate Pareto analysis
            sorted_customers = top_customer_analysis.sort_values('Total_Revenue', ascending=False).reset_index(drop=True)
            sorted_customers['Cumulative_Revenue'] = sorted_customers['Total_Revenue'].cumsum()
            sorted_customers['Cumulative_Revenue_Pct'] = sorted_customers['Cumulative_Revenue'] / sorted_customers['Total_Revenue'].sum()
            sorted_customers['Customer_Rank_Pct'] = (sorted_customers.index + 1) / len(sorted_customers)
            
            # Find 80/20 point
            pareto_80_idx = sorted_customers[sorted_customers['Cumulative_Revenue_Pct'] >= 0.8].index[0] if not sorted_customers.empty else 0
            pareto_80_pct = (pareto_80_idx + 1) / len(sorted_customers) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "80% Revenue Concentration", 
                    f"{pareto_80_pct:.1f}% of customers",
                    help="What percentage of customers generate 80% of revenue"
                )
            with col2:
                top_10_revenue_share = sorted_customers.head(10)['Total_Revenue'].sum() / sorted_customers['Total_Revenue'].sum()
                st.metric(
                    "Top 10 Customer Revenue Share",
                    f"{top_10_revenue_share:.1%}",
                    help="Revenue concentration in top 10 customers"
                )
            with col3:
                avg_customer_revenue = sorted_customers['Total_Revenue'].mean()
                top_customer_revenue = sorted_customers['Total_Revenue'].iloc[0]
                top_vs_avg_ratio = top_customer_revenue / avg_customer_revenue
                st.metric(
                    "Top Customer vs Average",
                    f"{top_vs_avg_ratio:.1f}x",
                    help="How many times larger is the top customer vs average"
                )
            
            # Pareto chart
            fig_pareto = go.Figure()
            
            # Revenue bars
            fig_pareto.add_trace(go.Bar(
                x=sorted_customers.head(50).index + 1,
                y=sorted_customers.head(50)['Total_Revenue'],
                name='Customer Revenue',
                marker_color='lightblue',
                yaxis='y'
            ))
            
            # Cumulative percentage line
            fig_pareto.add_trace(go.Scatter(
                x=sorted_customers.head(50).index + 1,
                y=sorted_customers.head(50)['Cumulative_Revenue_Pct'] * 100,
                name='Cumulative %',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))
            
            # Add 80% line
            fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", 
                                annotation_text="80% Revenue Line", yref="y2")
            
            fig_pareto.update_layout(
                title="Customer Revenue Pareto Analysis (Top 50)",
                xaxis_title="Customer Rank",
                yaxis=dict(title="Revenue ($)", side="left"),
                yaxis2=dict(title="Cumulative Revenue %", side="right", overlaying="y"),
                height=500
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True, key="pareto_analysis")
            
            # Time-based performance comparison
            st.subheader("📊 Multi-Period Performance Comparison")
            
            if 'Revenue_Last_Year' in top_customer_analysis.columns:
                # Create comparison chart
                top_20_comparison = top_customer_analysis.nlargest(20, 'Total_Revenue')[
                    ['Account', 'Total_Revenue', 'Revenue_Last_Year', 'Revenue_Last_Quarter', 'Revenue_Last_Month']
                ].fillna(0)
                
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    name='Last Month',
                    x=top_20_comparison['Account'],
                    y=top_20_comparison['Revenue_Last_Month'],
                    marker_color='lightcoral'
                ))
                
                fig_comparison.add_trace(go.Bar(
                    name='Last Quarter',
                    x=top_20_comparison['Account'],
                    y=top_20_comparison['Revenue_Last_Quarter'],
                    marker_color='lightblue'
                ))
                
                fig_comparison.add_trace(go.Bar(
                    name='Last Year',
                    x=top_20_comparison['Account'],
                    y=top_20_comparison['Revenue_Last_Year'],
                    marker_color='lightgreen'
                ))
                
                fig_comparison.update_layout(
                    title="Revenue Performance: Top 20 Customers Across Time Periods",
                    xaxis_title="Customer",
                    yaxis_title="Revenue ($)",
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True, key="multi_period_comparison")
        else:
            st.warning("No customer data available for analysis.")
    
    with tab8:
        st.header("🎖️ Elite Customer Intelligence")
        
        if not customer_cross_sell_journeys.empty and not customer_bl_penetration.empty:
            # Elite customer cross-sell analysis
            st.subheader("🚀 Cross-Sell Champions Analysis")
            
            with st.expander("ℹ️ What does this analysis show?", expanded=False):
                st.markdown("""
                **Cross-Sell Champions Analysis** identifies customers who excel at adopting multiple business lines.
                
                **Key Insights:**
                - **Cross-Sell Velocity:** How quickly customers adopt new business lines
                - **Business Line Sequence:** Typical progression paths of successful customers
                - **Revenue per Business Line:** Monetization efficiency across services
                - **Time Between Cross-Sells:** Optimal timing for cross-sell campaigns
                
                **Strategic Value:**
                - **Identify patterns** in successful cross-sell journeys
                - **Optimize timing** of cross-sell offers
                - **Create lookalike profiles** for targeting
                - **Develop sequenced** product introduction strategies
                """)
            
            # Top cross-sell performers
            top_cross_sell = customer_cross_sell_journeys.nlargest(20, 'Cross_Sell_Count')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏅 Top Cross-Sell Champions")
                
                champions_display = top_cross_sell[
                    ['Account', 'Total_Revenue', 'Cross_Sell_Count', 'Cross_Sell_Velocity', 'Avg_Time_Between_Cross_Sells']
                ].copy()
                
                champions_styled = champions_display.style.format({
                    'Total_Revenue': '${:,.0f}',
                    'Cross_Sell_Velocity': '{:.2f}',
                    'Avg_Time_Between_Cross_Sells': '{:.0f} days'
                })
                
                st.dataframe(champions_styled, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Cross-Sell Velocity Distribution")
                
                fig_velocity = px.histogram(
                    customer_cross_sell_journeys,
                    x='Cross_Sell_Velocity',
                    nbins=20,
                    title="Cross-Sell Velocity Distribution",
                    labels={'Cross_Sell_Velocity': 'Cross-Sells per Year', 'count': 'Number of Customers'}
                )
                fig_velocity.update_layout(height=400)
                st.plotly_chart(fig_velocity, use_container_width=True, key="velocity_distribution")
            
            # Business line penetration analysis
            st.subheader("🎯 Business Line Penetration Excellence")
            
            if not customer_bl_penetration.empty:
                # Merge with revenue data for comprehensive view
                penetration_with_revenue = customer_bl_penetration.merge(
                    top_customer_analysis[['Account', 'Total_Revenue', 'Elite_Score']], 
                    on='Account', 
                    how='left'
                )
                
                # Elite penetration customers
                top_penetration = penetration_with_revenue.nlargest(20, 'Penetration_Rate')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🌟 Highest Business Line Penetration")
                    
                    # Check which columns are available and create display accordingly
                    available_cols = top_penetration.columns.tolist()
                    
                    display_cols = ['Account', 'Active_Business_Lines', 'Penetration_Rate', 'Diversification_Index']
                    if 'Total_Revenue' in available_cols:
                        display_cols.insert(1, 'Total_Revenue')
                    elif 'Total_Revenue_x' in available_cols:
                        display_cols.insert(1, 'Total_Revenue_x')
                        top_penetration = top_penetration.rename(columns={'Total_Revenue_x': 'Total_Revenue'})
                    elif 'Total_Revenue_y' in available_cols:
                        display_cols.insert(1, 'Total_Revenue_y')
                        top_penetration = top_penetration.rename(columns={'Total_Revenue_y': 'Total_Revenue'})
                    
                    # Filter to only available columns
                    display_cols = [col for col in display_cols if col in top_penetration.columns]
                    
                    if display_cols:
                        penetration_display = top_penetration[display_cols].copy()
                        
                        # Create formatting dict only for available columns
                        format_dict = {}
                        if 'Total_Revenue' in penetration_display.columns:
                            format_dict['Total_Revenue'] = '${:,.0f}'
                        if 'Penetration_Rate' in penetration_display.columns:
                            format_dict['Penetration_Rate'] = '{:.1%}'
                        if 'Diversification_Index' in penetration_display.columns:
                            format_dict['Diversification_Index'] = '{:.2f}'
                        if 'Active_Business_Lines' in penetration_display.columns:
                            format_dict['Active_Business_Lines'] = '{:,}'
                        
                        penetration_styled = penetration_display.style.format(format_dict)
                        st.dataframe(penetration_styled, use_container_width=True)
                    else:
                        st.warning("No suitable columns available for penetration display.")
                
                with col2:
                    st.markdown("### 🎲 Revenue Diversification vs Concentration")
                    
                    # Check if required columns exist for scatter plot
                    if all(col in penetration_with_revenue.columns for col in ['Penetration_Rate', 'Diversification_Index']):
                        # Use available revenue column
                        size_col = 'Total_Revenue'
                        if 'Total_Revenue' not in penetration_with_revenue.columns:
                            if 'Total_Revenue_x' in penetration_with_revenue.columns:
                                size_col = 'Total_Revenue_x'
                            elif 'Total_Revenue_y' in penetration_with_revenue.columns:
                                size_col = 'Total_Revenue_y'
                            else:
                                size_col = None
                        
                        scatter_kwargs = {
                            'data_frame': penetration_with_revenue,
                            'x': 'Penetration_Rate',
                            'y': 'Diversification_Index',
                            'hover_data': ['Account'],
                            'title': "Customer Portfolio Diversification",
                            'labels': {
                                'Penetration_Rate': 'Business Line Penetration Rate',
                                'Diversification_Index': 'Revenue Diversification Index'
                            }
                        }
                        
                        if size_col and size_col in penetration_with_revenue.columns:
                            scatter_kwargs['size'] = size_col
                            scatter_kwargs['labels'][size_col] = 'Total Revenue'
                        
                        fig_diversification = px.scatter(**scatter_kwargs)
                        fig_diversification.update_layout(height=400)
                        st.plotly_chart(fig_diversification, use_container_width=True, key="diversification_scatter")
                    else:
                        st.warning("Required columns not available for diversification scatter plot.")
            else:
                st.warning("No business line penetration data available.")
            
            # Customer growth trajectory analysis
            st.subheader("📈 Growth Trajectory Champions")
            
            if not customer_growth_patterns.empty:
                with st.expander("ℹ️ What does this analysis show?", expanded=False):
                    st.markdown("""
                    **Growth Trajectory Analysis** identifies customers with exceptional growth patterns.
                    
                    **Growth Metrics:**
                    - **Total Growth Rate:** Overall revenue growth from first to last period
                    - **Average MoM Growth:** Consistent month-over-month improvement
                    - **Growth Consistency:** Reliability of positive growth months
                    - **Growth Volatility:** Stability vs erratic growth patterns
                    - **Recent Performance:** Latest trends vs historical performance
                    
                    **Strategic Applications:**
                    - **Identify high-potential accounts** for increased investment
                    - **Spot declining accounts** needing intervention
                    - **Understand growth drivers** to replicate success
                    - **Predict future performance** based on growth patterns
                    """)
                
                # Growth champions analysis
                stable_growers = customer_growth_patterns[
                    (customer_growth_patterns['Avg_MoM_Growth'] > 0) & 
                    (customer_growth_patterns['Growth_Consistency'] > 0.6)
                ].nlargest(15, 'Avg_MoM_Growth')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🚀 Consistent Growth Champions")
                    
                    if not stable_growers.empty:
                        growth_display = stable_growers[
                            ['Account', 'Total_Revenue', 'Avg_MoM_Growth', 'Growth_Consistency', 'Growth_Volatility']
                        ].copy()
                        
                        growth_styled = growth_display.style.format({
                            'Total_Revenue': '${:,.0f}',
                            'Avg_MoM_Growth': '{:.1f}%',
                            'Growth_Consistency': '{:.1%}',
                            'Growth_Volatility': '{:.1f}%'
                        })
                        
                        st.dataframe(growth_styled, use_container_width=True)
                    else:
                        st.info("No consistent growth champions identified in current dataset.")
                
                with col2:
                    st.markdown("### ⚡ Growth vs Consistency Matrix")
                    
                    fig_growth_matrix = px.scatter(
                        customer_growth_patterns,
                        x='Growth_Consistency',
                        y='Avg_MoM_Growth',
                        size='Total_Revenue',
                        hover_data=['Account'],
                        title="Growth Reliability Analysis",
                        labels={
                            'Growth_Consistency': 'Growth Consistency (% positive months)',
                            'Avg_MoM_Growth': 'Average Monthly Growth Rate (%)',
                            'Total_Revenue': 'Total Revenue'
                        }
                    )
                    
                    # Add quadrant lines
                    fig_growth_matrix.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_growth_matrix.add_vline(x=0.5, line_dash="dash", line_color="gray")
                    
                    fig_growth_matrix.update_layout(height=400)
                    st.plotly_chart(fig_growth_matrix, use_container_width=True, key="growth_consistency_matrix")
            
            # Customer journey sequences analysis
            st.subheader("🛤️ Elite Customer Journey Patterns")
            
            if not customer_cross_sell_journeys.empty:
                # Most successful journey patterns
                journey_patterns = customer_cross_sell_journeys['Business_Line_Sequence'].value_counts().head(10)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if not journey_patterns.empty:
                        journey_df = pd.DataFrame({
                            'Journey Pattern': journey_patterns.index,
                            'Customer Count': journey_patterns.values,
                            'Success Rate': (journey_patterns.values / customer_cross_sell_journeys.shape[0] * 100).round(1)
                        })
                        
                        # Calculate average revenue for each pattern
                        pattern_revenues = []
                        for pattern in journey_patterns.index:
                            pattern_customers = customer_cross_sell_journeys[
                                customer_cross_sell_journeys['Business_Line_Sequence'] == pattern
                            ]
                            avg_revenue = pattern_customers['Total_Revenue'].mean()
                            pattern_revenues.append(avg_revenue)
                        
                        journey_df['Avg Revenue per Customer'] = pattern_revenues
                        
                        journey_styled = journey_df.style.format({
                            'Customer Count': '{:,}',
                            'Success Rate': '{:.1f}%',
                            'Avg Revenue per Customer': '${:,.0f}'
                        })
                        
                        st.markdown("### 🏆 Most Successful Cross-Sell Journey Patterns")
                        st.dataframe(journey_styled, use_container_width=True)
                
                with col2:
                    # Journey complexity vs revenue
                    complexity_revenue = customer_cross_sell_journeys.groupby('Cross_Sell_Count').agg({
                        'Total_Revenue': 'mean',
                        'Account': 'count'
                    }).reset_index()
                    complexity_revenue.columns = ['Cross_Sell_Count', 'Avg_Revenue', 'Customer_Count']
                    
                    fig_complexity = px.bar(
                        complexity_revenue,
                        x='Cross_Sell_Count',
                        y='Avg_Revenue',
                        title="Revenue by Cross-Sell Complexity",
                        labels={
                            'Cross_Sell_Count': 'Number of Cross-Sells',
                            'Avg_Revenue': 'Average Revenue ($)'
                        }
                    )
                    fig_complexity.update_layout(height=400)
                    st.plotly_chart(fig_complexity, use_container_width=True, key="complexity_revenue")
        else:
            st.warning("No customer journey data available for elite analysis.")
    
    with tab9:
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
    main()# Advanced Cross-Sell Analytics Dashboard (v5.0)
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

@st.cache_data(show_spinner="Analyzing top customer performance...")
def analyze_top_customers(_df: pd.DataFrame):
    """Comprehensive analysis of top-performing customers."""
    try:
        current_date = _df['Date'].max()
        
        # Define time periods
        last_month_start = current_date - pd.DateOffset(months=1)
        last_quarter_start = current_date - pd.DateOffset(months=3)
        last_year_start = current_date - pd.DateOffset(months=12)
        
        # Calculate customer metrics for different periods
        customer_metrics = {}
        
        # All-time metrics
        all_time = _df.groupby('Account').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'BusinessLine': ['nunique', lambda x: list(x.unique())],
            'Date': ['min', 'max']
        }).round(2)
        
        all_time.columns = ['Total_Revenue', 'Avg_Revenue_Per_Transaction', 'Transaction_Count', 
                            'Unique_Business_Lines', 'Business_Lines_List', 'First_Purchase', 'Last_Purchase']
        
        # Calculate customer lifetime (days)
        all_time['Customer_Lifetime_Days'] = (all_time['Last_Purchase'] - all_time['First_Purchase']).dt.days
        all_time['Revenue_Per_Day'] = all_time['Total_Revenue'] / (all_time['Customer_Lifetime_Days'] + 1)  # +1 to avoid division by zero
        
        # Last year metrics
        last_year_data = _df[_df['Date'] >= last_year_start]
        if not last_year_data.empty:
            last_year = last_year_data.groupby('Account').agg({
                'Revenue': ['sum', 'mean', 'count'],
                'BusinessLine': 'nunique'
            }).round(2)
            last_year.columns = ['Revenue_Last_Year', 'Avg_Revenue_Last_Year', 'Transactions_Last_Year', 'Business_Lines_Last_Year']
        else:
            last_year = pd.DataFrame()
        
        # Last quarter metrics  
        last_quarter_data = _df[_df['Date'] >= last_quarter_start]
        if not last_quarter_data.empty:
            last_quarter = last_quarter_data.groupby('Account').agg({
                'Revenue': ['sum', 'mean', 'count'],
                'BusinessLine': 'nunique'
            }).round(2)
            last_quarter.columns = ['Revenue_Last_Quarter', 'Avg_Revenue_Last_Quarter', 'Transactions_Last_Quarter', 'Business_Lines_Last_Quarter']
        else:
            last_quarter = pd.DataFrame()
        
        # Last month metrics
        last_month_data = _df[_df['Date'] >= last_month_start]
        if not last_month_data.empty:
            last_month = last_month_data.groupby('Account').agg({
                'Revenue': ['sum', 'mean', 'count'],
                'BusinessLine': 'nunique'
            }).round(2)
            last_month.columns = ['Revenue_Last_Month', 'Avg_Revenue_Last_Month', 'Transactions_Last_Month', 'Business_Lines_Last_Month']
        else:
            last_month = pd.DataFrame()
        
        # Combine all metrics
        customer_analysis = all_time.copy()
        
        if not last_year.empty:
            customer_analysis = customer_analysis.join(last_year, how='left')
        if not last_quarter.empty:
            customer_analysis = customer_analysis.join(last_quarter, how='left')
        if not last_month.empty:
            customer_analysis = customer_analysis.join(last_month, how='left')
        
        # Fill NaN values with 0 for revenue metrics
        revenue_columns = [col for col in customer_analysis.columns if 'Revenue' in col or 'Transactions' in col or 'Business_Lines' in col]
        customer_analysis[revenue_columns] = customer_analysis[revenue_columns].fillna(0)
        
        # Calculate cross-sell progression metrics
        customer_analysis['Cross_Sell_Score'] = customer_analysis['Unique_Business_Lines'] / customer_analysis['Unique_Business_Lines'].max()
        customer_analysis['Customer_Value_Score'] = customer_analysis['Total_Revenue'] / customer_analysis['Total_Revenue'].max()
        customer_analysis['Engagement_Score'] = customer_analysis['Transaction_Count'] / customer_analysis['Transaction_Count'].max()
        
        # Combined score for ranking
        customer_analysis['Elite_Score'] = (
            customer_analysis['Cross_Sell_Score'] * 0.3 +
            customer_analysis['Customer_Value_Score'] * 0.5 +
            customer_analysis['Engagement_Score'] * 0.2
        )
        
        # Calculate growth rates
        if not last_year.empty:
            # Year-over-year growth (comparing last year to previous year)
            previous_year_start = last_year_start - pd.DateOffset(months=12)
            previous_year_data = _df[(_df['Date'] >= previous_year_start) & (_df['Date'] < last_year_start)]
            if not previous_year_data.empty:
                previous_year = previous_year_data.groupby('Account')['Revenue'].sum()
                customer_analysis = customer_analysis.join(previous_year.rename('Revenue_Previous_Year'), how='left')
                customer_analysis['Revenue_Previous_Year'] = customer_analysis['Revenue_Previous_Year'].fillna(0)
                customer_analysis['YoY_Growth_Rate'] = (
                    (customer_analysis['Revenue_Last_Year'] - customer_analysis['Revenue_Previous_Year']) / 
                    (customer_analysis['Revenue_Previous_Year'] + 1)  # +1 to avoid division by zero
                ) * 100
            
        return customer_analysis.reset_index()
        
    except Exception as e:
        st.error(f"Error analyzing top customers: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Creating customer cross-sell journey analysis...")
def analyze_customer_cross_sell_journeys(_df: pd.DataFrame, top_customers: list):
    """Analyze cross-sell journeys for top customers."""
    try:
        journey_analysis = []
        
        for customer in top_customers:
            customer_data = _df[_df['Account'] == customer].sort_values('Date')
            
            if customer_data.empty:
                continue
                
            # Get business line progression
            business_lines = customer_data['BusinessLine'].tolist()
            dates = customer_data['Date'].tolist()
            revenues = customer_data['Revenue'].tolist()
            
            # Find unique business line sequence (preserve order)
            unique_sequence = []
            seen = set()
            for bl in business_lines:
                if bl not in seen:
                    unique_sequence.append(bl)
                    seen.add(bl)
            
            # Calculate metrics
            total_revenue = sum(revenues)
            first_business_line = unique_sequence[0] if unique_sequence else None
            cross_sell_count = len(unique_sequence) - 1
            customer_lifetime = (dates[-1] - dates[0]).days if len(dates) > 1 else 0
            
            # Time between cross-sells
            cross_sell_times = []
            if len(unique_sequence) > 1:
                bl_first_dates = {}
                for bl, date in zip(business_lines, dates):
                    if bl not in bl_first_dates:
                        bl_first_dates[bl] = date
                
                sorted_bl_dates = sorted([(bl_first_dates[bl], bl) for bl in unique_sequence])
                for i in range(1, len(sorted_bl_dates)):
                    time_diff = (sorted_bl_dates[i][0] - sorted_bl_dates[i-1][0]).days
                    cross_sell_times.append(time_diff)
            
            avg_cross_sell_time = np.mean(cross_sell_times) if cross_sell_times else 0
            
            journey_analysis.append({
                'Account': customer,
                'Total_Revenue': total_revenue,
                'First_Business_Line': first_business_line,
                'Business_Line_Sequence': ' → '.join(unique_sequence),
                'Cross_Sell_Count': cross_sell_count,
                'Customer_Lifetime_Days': customer_lifetime,
                'Avg_Time_Between_Cross_Sells': avg_cross_sell_time,
                'Cross_Sell_Velocity': cross_sell_count / max(customer_lifetime / 365, 0.1),  # Cross-sells per year
                'Revenue_Per_Business_Line': total_revenue / len(unique_sequence) if unique_sequence else 0
            })
        
        return pd.DataFrame(journey_analysis)
        
    except Exception as e:
        st.error(f"Error analyzing customer cross-sell journeys: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Calculating customer business line penetration...")
def calculate_customer_bl_penetration(_df: pd.DataFrame):
    """Calculate how well each customer penetrates different business lines."""
    try:
        # Get all unique business lines
        all_business_lines = _df['BusinessLine'].unique()
        total_bl_count = len(all_business_lines)
        
        # Create customer-business line matrix
        customer_bl_matrix = _df.pivot_table(
            index='Account', 
            columns='BusinessLine', 
            values='Revenue', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Calculate penetration metrics
        penetration_metrics = []
        
        for customer in customer_bl_matrix.index:
            customer_row = customer_bl_matrix.loc[customer]
            
            # Business lines with revenue > 0
            active_bls = (customer_row > 0).sum()
            penetration_rate = active_bls / total_bl_count
            
            # Revenue distribution across business lines
            total_revenue = customer_row.sum()
            revenue_concentration = (customer_row.max() / total_revenue) if total_revenue > 0 else 0
            
            # Calculate Herfindahl-Hirschman Index for concentration
            revenue_shares = customer_row / total_revenue if total_revenue > 0 else customer_row
            hhi = sum(share**2 for share in revenue_shares)
            
            penetration_metrics.append({
                'Account': customer,
                'Total_Revenue': total_revenue,
                'Active_Business_Lines': active_bls,
                'Penetration_Rate': penetration_rate,
                'Revenue_Concentration': revenue_concentration,
                'Diversification_Index': 1 - hhi,  # Higher = more diversified
                'Primary_Business_Line': customer_row.idxmax(),
                'Primary_BL_Revenue_Share': revenue_concentration
            })
        
        return pd.DataFrame(penetration_metrics)
        
    except Exception as e:
        st.error(f"Error calculating customer business line penetration: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Analyzing customer growth patterns...")
def analyze_customer_growth_patterns(_df: pd.DataFrame, top_customers: list):
    """Analyze growth patterns and trends for top customers."""
    try:
        growth_analysis = []
        
        for customer in top_customers:
            customer_data = _df[_df['Account'] == customer].sort_values('Date')
            
            if len(customer_data) < 2:
                continue
            
            # Monthly revenue aggregation
            monthly_revenue = customer_data.groupby(customer_data['Date'].dt.to_period('M'))['Revenue'].sum()
            
            if len(monthly_revenue) < 2:
                continue
            
            # Calculate growth metrics
            total_months = len(monthly_revenue)
            total_growth = (monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0] * 100 if monthly_revenue.iloc[0] > 0 else 0
            
            # Month-over-month growth rates
            mom_growth_rates = monthly_revenue.pct_change().dropna() * 100
            avg_mom_growth = mom_growth_rates.mean()
            growth_volatility = mom_growth_rates.std()
            
            # Trend analysis (simple linear regression slope)
            months_numeric = range(len(monthly_revenue))
            if len(months_numeric) > 1:
                trend_slope = np.polyfit(months_numeric, monthly_revenue.values, 1)[0]
            else:
                trend_slope = 0
            
            # Growth consistency (% of months with positive growth)
            positive_growth_months = (mom_growth_rates > 0).sum()
            growth_consistency = positive_growth_months / len(mom_growth_rates) if len(mom_growth_rates) > 0 else 0
            
            # Recent performance (last 3 months vs previous 3 months)
            if len(monthly_revenue) >= 6:
                recent_avg = monthly_revenue.iloc[-3:].mean()
                previous_avg = monthly_revenue.iloc[-6:-3].mean()
                recent_vs_previous = (recent_avg - previous_avg) / previous_avg * 100 if previous_avg > 0 else 0
            else:
                recent_vs_previous = 0
            
            growth_analysis.append({
                'Account': customer,
                'Total_Revenue': customer_data['Revenue'].sum(),
                'Total_Growth_Rate': total_growth,
                'Avg_MoM_Growth': avg_mom_growth,
                'Growth_Volatility': growth_volatility,
                'Trend_Slope': trend_slope,
                'Growth_Consistency': growth_consistency,
                'Recent_vs_Previous_Performance': recent_vs_previous,
                'Active_Months': total_months
            })
        
        return pd.DataFrame(growth_analysis)
        
    except Exception as e:
        st.error(f"Error analyzing customer growth patterns: {str(e)}")
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

@st.cache_data(show_spinner="Creating temporal analysis...")
def create_temporal_analysis(_df: pd.DataFrame, granularity: str = "Monthly"):
    """Create temporal analysis of revenue, customers, and cross-sell metrics."""
    try:
        if granularity == "Monthly":
            time_col = 'YM'
            _df['TimeGroup'] = _df['Date'].dt.to_period('M').astype(str)
        elif granularity == "Quarterly":
            time_col = 'Q'
            _df['TimeGroup'] = _df['Date'].dt.to_period('Q').astype(str)
        else:  # Yearly
            time_col = 'Y'
            _df['TimeGroup'] = _df['Date'].dt.year.astype(str)
        
        # Aggregate metrics by time period
        temporal_metrics = _df.groupby('TimeGroup').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Account': 'nunique',
            'BusinessLine': 'nunique'
        }).round(2)
        
        temporal_metrics.columns = ['Total_Revenue', 'Avg_Transaction_Value', 'Transaction_Count', 'Unique_Customers', 'Active_Business_Lines']
        temporal_metrics = temporal_metrics.reset_index()
        
        # Calculate cross-sell metrics per period
        cross_sell_temporal = []
        for period in temporal_metrics['TimeGroup']:
            period_data = _df[_df['TimeGroup'] == period]
            
            if not period_data.empty:
                # Customer journey analysis for this period
                customer_products = period_data.groupby('Account')['BusinessLine'].nunique()
                multi_product_customers = (customer_products > 1).sum()
                total_customers = len(customer_products)
                cross_sell_rate = multi_product_customers / total_customers if total_customers > 0 else 0
                
                cross_sell_temporal.append({
                    'TimeGroup': period,
                    'Cross_Sell_Rate': cross_sell_rate,
                    'Multi_Product_Customers': multi_product_customers,
                    'Total_Customers': total_customers
                })
        
        cross_sell_df = pd.DataFrame(cross_sell_temporal)
        
        # Merge temporal metrics
        if not cross_sell_df.empty:
            temporal_metrics = temporal_metrics.merge(cross_sell_df, on='TimeGroup', how='left')
        
        # Calculate period-over-period growth
        temporal_metrics = temporal_metrics.sort_values('TimeGroup')
        temporal_metrics['Revenue_Growth'] = temporal_metrics['Total_Revenue'].pct_change() * 100
        temporal_metrics['Customer_Growth'] = temporal_metrics['Unique_Customers'].pct_change() * 100
        
        return temporal_metrics
        
    except Exception as e:
        st.error(f"Error creating temporal analysis: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Analyzing seasonal patterns...")
def analyze_seasonal_patterns(_df: pd.DataFrame):
    """Analyze seasonal patterns in revenue and customer behavior."""
    try:
        # Monthly seasonality
        _df['Month_Name'] = _df['Date'].dt.month_name()
        _df['Month_Num'] = _df['Date'].dt.month
        _df['Quarter_Name'] = 'Q' + _df['Date'].dt.quarter.astype(str)
        _df['Weekday'] = _df['Date'].dt.day_name()
        
        # Monthly patterns
        monthly_patterns = _df.groupby(['Month_Num', 'Month_Name']).agg({
            'Revenue': ['sum', 'mean'],
            'Account': 'nunique',
            'BusinessLine': 'nunique'
        }).round(2)
        
        monthly_patterns.columns = ['Total_Revenue', 'Avg_Revenue', 'Unique_Customers', 'Active_Business_Lines']
        monthly_patterns = monthly_patterns.reset_index()
        
        # Quarterly patterns
        quarterly_patterns = _df.groupby('Quarter_Name').agg({
            'Revenue': ['sum', 'mean'],
            'Account': 'nunique',
            'BusinessLine': 'nunique'
        }).round(2)
        
        quarterly_patterns.columns = ['Total_Revenue', 'Avg_Revenue', 'Unique_Customers', 'Active_Business_Lines']
        quarterly_patterns = quarterly_patterns.reset_index()
        
        # Year-over-year comparison
        yearly_comparison = _df.groupby('Y').agg({
            'Revenue': ['sum', 'mean'],
            'Account': 'nunique',
            'BusinessLine': 'nunique'
        }).round(2)
        
        yearly_comparison.columns = ['Total_Revenue', 'Avg_Revenue', 'Unique_Customers', 'Active_Business_Lines']
        yearly_comparison = yearly_comparison.reset_index()
        
        # Calculate year-over-year growth
        yearly_comparison = yearly_comparison.sort_values('Y')
        yearly_comparison['YoY_Revenue_Growth'] = yearly_comparison['Total_Revenue'].pct_change() * 100
        yearly_comparison['YoY_Customer_Growth'] = yearly_comparison['Unique_Customers'].pct_change() * 100
        
        return monthly_patterns, quarterly_patterns, yearly_comparison
        
    except Exception as e:
        st.error(f"Error analyzing seasonal patterns: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(show_spinner="Creating business line temporal analysis...")
def create_bl_temporal_analysis(_df: pd.DataFrame, granularity: str = "Monthly"):
    """Analyze business line performance over time."""
    try:
        if granularity == "Monthly":
            _df['TimeGroup'] = _df['Date'].dt.to_period('M').astype(str)
        elif granularity == "Quarterly":
            _df['TimeGroup'] = _df['Date'].dt.to_period('Q').astype(str)
        else:  # Yearly
            _df['TimeGroup'] = _df['Date'].dt.year.astype(str)
        
        # Business line performance over time
        bl_temporal = _df.groupby(['TimeGroup', 'BusinessLine']).agg({
            'Revenue': 'sum',
            'Account': 'nunique'
        }).reset_index()
        
        # Pivot for easier analysis
        bl_revenue_pivot = bl_temporal.pivot(index='TimeGroup', columns='BusinessLine', values='Revenue').fillna(0)
        bl_customers_pivot = bl_temporal.pivot(index='TimeGroup', columns='BusinessLine', values='Account').fillna(0)
        
        # Calculate market share over time
        bl_revenue_pivot_pct = bl_revenue_pivot.div(bl_revenue_pivot.sum(axis=1), axis=0) * 100
        
        return bl_temporal, bl_revenue_pivot, bl_customers_pivot, bl_revenue_pivot_pct
        
    except Exception as e:
        st.error(f"Error creating business line temporal analysis: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    
    # Get top customers for detailed analysis
    if not top_customer_analysis.empty:
        top_10_customers = top_customer_analysis.nlargest(10, 'Total_Revenue')['Account'].tolist()
        top_50_customers = top_customer_analysis.nlargest(50, 'Total_Revenue')['Account'].tolist()
        top_100_customers = top_customer_analysis.nlargest(100, 'Total_Revenue')['Account'].tolist()
        
        customer_cross_sell_journeys = analyze_customer_cross_sell_journeys(df_filtered, top_50_customers)
        customer_bl_penetration = calculate_customer_bl_penetration(df_filtered)
        customer_growth_patterns = analyze_customer_growth_patterns(df_filtered, top_50_customers)
        
        # Add temporal analysis
        temporal_metrics = create_temporal_analysis(df_filtered, temporal_granularity)
        monthly_patterns, quarterly_patterns, yearly_comparison = analyze_seasonal_patterns(df_filtered)
        bl_temporal, bl_revenue_pivot, bl_customers_pivot, bl_revenue_pivot_pct = create_bl_temporal_analysis(df_filtered, temporal_granularity)
    else:
        top_10_customers = top_50_customers = top_100_customers = []
        customer_cross_sell_journeys = pd.DataFrame()
        customer_bl_penetration = pd.DataFrame()
        customer_growth_patterns = pd.DataFrame()
        temporal_metrics = pd.DataFrame()
        monthly_patterns = pd.DataFrame()
        quarterly_patterns = pd.DataFrame()
        yearly_comparison = pd.DataFrame()
        bl_temporal = pd.DataFrame()
        bl_revenue_pivot = pd.DataFrame()
        bl_customers_pivot = pd.DataFrame()
        bl_revenue_pivot_pct = pd.DataFrame()
    
    
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
            
            # Add Business Line Temporal Performance
            st.subheader(f"📈 Business Line Performance Over Time ({temporal_granularity})")
            
            with st.expander("ℹ️ What does this temporal analysis show?", expanded=False):
                st.markdown(f"""
                **Business Line Temporal Performance** reveals how each product/service performs over time.
                
                **Key Metrics:**
                - **Revenue Evolution:** Which business lines are growing vs declining
                - **Market Share Changes:** How the competitive landscape shifts over time
                - **Customer Adoption Patterns:** Which products gain/lose customer traction
                - **Seasonal Variations:** Time-based performance patterns
                
                **Strategic Applications:**
                - **Investment Decisions:** Which business lines deserve more resources
                - **Product Lifecycle Management:** Identify products needing intervention
                - **Market Timing:** Optimal periods for product launches/promotions
                - **Portfolio Balancing:** Ensure sustainable revenue diversification
                """)
            
            if not bl_revenue_pivot.empty:
                # Business line revenue trends
                fig_bl_trends = go.Figure()
                
                for bl in bl_revenue_pivot.columns:
                    fig_bl_trends.add_trace(go.Scatter(
                        x=bl_revenue_pivot.index,
                        y=bl_revenue_pivot[bl],
                        mode='lines+markers',
                        name=bl,
                        line=dict(width=3)
                    ))
                
                fig_bl_trends.update_layout(
                    title=f"Business Line Revenue Trends ({temporal_granularity})",
                    xaxis_title=f"{temporal_granularity} Period",
                    yaxis_title="Revenue ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_bl_trends, use_container_width=True, key="bl_revenue_trends")
                
                # Market share evolution
                st.subheader("🏪 Market Share Evolution")
                
                fig_market_share = px.area(
                    bl_revenue_pivot_pct.reset_index().melt(id_vars='TimeGroup', var_name='BusinessLine', value_name='Market_Share'),
                    x='TimeGroup',
                    y='Market_Share',
                    color='BusinessLine',
                    title=f"Market Share Evolution by Business Line ({temporal_granularity})",
                    labels={'Market_Share': 'Market Share (%)', 'TimeGroup': f'{temporal_granularity} Period'}
                )
                fig_market_share.update_layout(height=500)
                st.plotly_chart(fig_market_share, use_container_width=True, key="market_share_evolution")
                
                # Business line performance table
                st.subheader("📊 Business Line Performance Summary")
                
                # Calculate performance metrics
                bl_performance_summary = []
                for bl in bl_revenue_pivot.columns:
                    bl_data = bl_revenue_pivot[bl]
                    if len(bl_data) > 1:
                        total_revenue = bl_data.sum()
                        avg_revenue = bl_data.mean()
                        growth_rate = ((bl_data.iloc[-1] - bl_data.iloc[0]) / bl_data.iloc[0] * 100) if bl_data.iloc[0] > 0 else 0
                        volatility = bl_data.std()
                        trend = "📈 Growing" if growth_rate > 5 else "📉 Declining" if growth_rate < -5 else "➡️ Stable"
                        
                        bl_performance_summary.append({
                            'Business Line': bl,
                            'Total Revenue': total_revenue,
                            'Average Revenue': avg_revenue,
                            'Growth Rate (%)': growth_rate,
                            'Volatility': volatility,
                            'Trend': trend
                        })
                
                if bl_performance_summary:
                    bl_summary_df = pd.DataFrame(bl_performance_summary)
                    bl_summary_styled = bl_summary_df.style.format({
                        'Total Revenue': '${:,.0f}',
                        'Average Revenue': '${:,.0f}',
                        'Growth Rate (%)': '{:.1f}%',
                        'Volatility': '{:.0f}'
                    })
                    
                    st.dataframe(bl_summary_styled, use_container_width=True)
    
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
        
        # Add temporal analysis section
        st.subheader(f"📅 {temporal_granularity} Performance Trends")
        
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
        
        if not temporal_metrics.empty:
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
                yaxis=dict(title="Revenue ($)", side="left", titlefont=dict(color="#1f77b4")),
                yaxis2=dict(title="Customers", side="right", overlaying="y", titlefont=dict(color="#ff7f0e")),
                yaxis3=dict(title="Cross-Sell Rate (%)", side="right", overlaying="y", position=0.95, titlefont=dict(color="#2ca02c")),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_multi, use_container_width=True, key="temporal_multi_metric")
        
        # Insights with improved color coding and visibility
        st.subheader("💡 Key Performance Insights")
        
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