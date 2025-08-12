import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional
import json

# Page configuration
st.set_page_config(
    page_title="GSC Cannibalization vs Diversification Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GSCAnalyzer:
    """Google Search Console Cannibalization vs Diversification Analyzer"""
    
    def __init__(self):
        self.stopwords = {'the', 'and', 'a', 'an', 'to', 'for', 'of', 'in'}
        self.thresholds = {
            'leader_changes': 3,
            'sim_rank_share': 30,  # percentage
            'similar_intent_jaccard': 0.6,
            'min_second_share': 15,  # percentage
            'uplift_min': 0.15,
            'top_positions_cutoff': 20,
            'stability_threshold': 30,  # percentage
            'min_window_days': 28
        }
    
    def tokenize_url(self, url: str) -> set:
        """Tokenize URL path for similarity analysis"""
        # Extract path from URL
        if '//' in url:
            path = url.split('//', 1)[1]
        else:
            path = url
        
        # Remove domain
        if '/' in path:
            path = '/'.join(path.split('/')[1:])
        
        # Tokenize on common separators
        tokens = re.split(r'[/_\-\s]+', path.lower())
        
        # Remove stopwords and empty tokens
        tokens = {token for token in tokens if token and token not in self.stopwords}
        
        return tokens
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_query(self, query_data: pd.DataFrame) -> Dict:
        """Analyze a single query for cannibalization vs diversification"""
        query_name = query_data['query'].iloc[0]
        
        # Group by page and calculate metrics
        page_metrics = query_data.groupby('page').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean',
            'date': 'count'
        }).reset_index()
        
        page_metrics['stability_pct'] = (page_metrics['date'] / query_data['date'].nunique()) * 100
        page_metrics = page_metrics.sort_values('clicks', ascending=False)
        
        # Check if we have enough data
        total_dates = query_data['date'].nunique()
        if total_dates < self.thresholds['min_window_days']:
            return {
                'query': query_name,
                'classification': 'INSUFFICIENT_DATA',
                'reason': f'Only {total_dates} days of data (minimum {self.thresholds["min_window_days"]} required)'
            }
        
        # Calculate simultaneous ranking
        simultaneous_dates = 0
        leader_changes = 0
        previous_leader = None
        
        for date in query_data['date'].unique():
            date_data = query_data[query_data['date'] == date]
            ranking_pages = date_data[
                (date_data['position'] <= self.thresholds['top_positions_cutoff']) & 
                (date_data['impressions'] > 0)
            ]
            
            if len(ranking_pages) >= 2:
                simultaneous_dates += 1
            
            # Track leader changes
            if len(ranking_pages) > 0:
                current_leader = ranking_pages.loc[ranking_pages['position'].idxmin(), 'page']
                if previous_leader is not None and current_leader != previous_leader:
                    leader_changes += 1
                previous_leader = current_leader
        
        simultaneous_pct = (simultaneous_dates / total_dates) * 100
        
        # Get top 2 pages by clicks
        if len(page_metrics) < 2:
            return {
                'query': query_name,
                'classification': 'SINGLE_PAGE',
                'reason': 'Only one page ranking for this query'
            }
        
        top_page = page_metrics.iloc[0]
        second_page = page_metrics.iloc[1]
        
        # Calculate URL similarity
        top_tokens = self.tokenize_url(top_page['page'])
        second_tokens = self.tokenize_url(second_page['page'])
        jaccard_sim = self.jaccard_similarity(top_tokens, second_tokens)
        
        # Calculate traffic shares
        total_clicks = page_metrics['clicks'].sum()
        if total_clicks > 0:
            top_share = (top_page['clicks'] / total_clicks) * 100
            second_share = (second_page['clicks'] / total_clicks) * 100
        else:
            top_share = 0
            second_share = 0
        
        # Calculate uplift
        uplift = second_page['clicks'] / top_page['clicks'] if top_page['clicks'] > 0 else 0
        
        # Classification logic
        is_swapping = leader_changes >= self.thresholds['leader_changes']
        is_similar_intent = jaccard_sim >= self.thresholds['similar_intent_jaccard']
        is_rare_simultaneous = simultaneous_pct < self.thresholds['sim_rank_share']
        is_low_second_share = second_share < self.thresholds['min_second_share']
        
        is_frequent_simultaneous = simultaneous_pct >= self.thresholds['sim_rank_share']
        is_different_intent = jaccard_sim < self.thresholds['similar_intent_jaccard']
        is_meaningful_uplift = uplift >= self.thresholds['uplift_min']
        
        # Determine classification
        if (is_swapping and is_similar_intent and (is_rare_simultaneous or is_low_second_share)):
            classification = 'CANNIBALIZATION'
            reason = f'Swapping detected ({leader_changes} changes), similar URLs (Jaccard: {jaccard_sim:.2f}), '
            if is_rare_simultaneous:
                reason += f'rare simultaneous ranking ({simultaneous_pct:.1f}%)'
            else:
                reason += f'low second page share ({second_share:.1f}%)'
        elif (is_frequent_simultaneous and is_different_intent and is_meaningful_uplift):
            classification = 'DIVERSIFICATION'
            reason = f'Frequent simultaneous ranking ({simultaneous_pct:.1f}%), different URLs (Jaccard: {jaccard_sim:.2f}), meaningful uplift ({uplift:.2f})'
        else:
            classification = 'NEUTRAL'
            reason = 'Does not meet criteria for cannibalization or diversification'
        
        return {
            'query': query_name,
            'classification': classification,
            'reason': reason,
            'metrics': {
                'total_clicks': int(total_clicks),
                'total_impressions': int(page_metrics['impressions'].sum()),
                'leader_changes': leader_changes,
                'simultaneous_pct': simultaneous_pct,
                'jaccard_similarity': jaccard_sim,
                'top_page': top_page['page'],
                'second_page': second_page['page'],
                'top_share': top_share,
                'second_share': second_share,
                'uplift': uplift,
                'top_avg_position': top_page['position'],
                'second_avg_position': second_page['position']
            }
        }
    
    def generate_recommendations(self, analysis_results: List[Dict]) -> Dict:
        """Generate actionable recommendations"""
        cannibalization_cases = [r for r in analysis_results if r['classification'] == 'CANNIBALIZATION']
        diversification_cases = [r for r in analysis_results if r['classification'] == 'DIVERSIFICATION']
        
        # Sort by impact (total clicks)
        cannibalization_cases.sort(key=lambda x: x['metrics']['total_clicks'], reverse=True)
        diversification_cases.sort(key=lambda x: x['metrics']['total_clicks'], reverse=True)
        
        recommendations = {
            'cannibalization': [],
            'diversification': []
        }
        
        # Top 5 cannibalization fixes
        for case in cannibalization_cases[:5]:
            metrics = case['metrics']
            rec = {
                'query': case['query'],
                'priority': 'High' if metrics['total_clicks'] > 100 else 'Medium',
                'action': f"Consolidate to {metrics['top_page']} (stronger performer)",
                'details': [
                    f"301 redirect {metrics['second_page']} to {metrics['top_page']}",
                    f"Update internal links to point to {metrics['top_page']}",
                    f"Consider merging content if both pages have unique value"
                ],
                'expected_uplift': f"{metrics['total_clicks'] * 0.15:.0f}-{metrics['total_clicks'] * 0.3:.0f} additional clicks"
            }
            recommendations['cannibalization'].append(rec)
        
        # Top 5 diversification wins
        for case in diversification_cases[:5]:
            metrics = case['metrics']
            rec = {
                'query': case['query'],
                'action': f"Maintain and optimise both pages",
                'details': [
                    f"Strengthen {metrics['top_page']} for primary intent",
                    f"Optimise {metrics['second_page']} for secondary intent",
                    f"Consider adding complementary formats (video, tools, FAQ)"
                ],
                'current_benefit': f"{metrics['total_clicks']:.0f} total clicks from diversification"
            }
            recommendations['diversification'].append(rec)
        
        return recommendations
    
    def create_trend_visualization(self, data: pd.DataFrame, query: str) -> go.Figure:
        """Create trend visualization for a specific query"""
        query_data = data[data['query'] == query].copy()
        
        if query_data.empty:
            return go.Figure()
        
        # Get top pages by total clicks
        top_pages = query_data.groupby('page')['clicks'].sum().nlargest(3).index.tolist()
        query_data = query_data[query_data['page'].isin(top_pages)]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Position Over Time', 'Clicks Over Time'),
            vertical_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, page in enumerate(top_pages):
            page_data = query_data[query_data['page'] == page].sort_values('date')
            
            # Shorten URL for legend
            page_short = page.split('/')[-1] if '/' in page else page
            if len(page_short) > 30:
                page_short = page_short[:27] + '...'
            
            # Position chart
            fig.add_trace(
                go.Scatter(
                    x=page_data['date'],
                    y=page_data['position'],
                    mode='lines+markers',
                    name=f'{page_short} (Position)',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Clicks chart
            fig.add_trace(
                go.Scatter(
                    x=page_data['date'],
                    y=page_data['clicks'],
                    mode='lines+markers',
                    name=f'{page_short} (Clicks)',
                    line=dict(color=colors[i % len(colors)], dash='dash'),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        fig.update_yaxes(title_text="Position", row=1, col=1, autorange="reversed")
        fig.update_yaxes(title_text="Clicks", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=f"Trend Analysis: {query}",
            height=600,
            hovermode='x unified'
        )
        
        return fig

def main():
    st.title("🔍 GSC Cannibalization vs Diversification Analyzer")
    st.markdown("Identify keyword cannibalization issues and healthy diversification opportunities from Google Search Console data.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    analyzer = GSCAnalyzer()
    
    # Threshold adjustments
    st.sidebar.subheader("Analysis Thresholds")
    analyzer.thresholds['leader_changes'] = st.sidebar.slider("Leader Changes (Swapping)", 2, 10, 3)
    analyzer.thresholds['similar_intent_jaccard'] = st.sidebar.slider("Similar Intent Threshold", 0.3, 0.9, 0.6, 0.1)
    analyzer.thresholds['sim_rank_share'] = st.sidebar.slider("Simultaneous Ranking %", 10, 50, 30)
    analyzer.thresholds['min_second_share'] = st.sidebar.slider("Min Second Page Share %", 5, 30, 15)
    analyzer.thresholds['uplift_min'] = st.sidebar.slider("Minimum Uplift", 0.05, 0.5, 0.15, 0.05)
    
    # File upload
    st.header("📁 Upload GSC Data")
    uploaded_file = st.file_uploader(
        "Upload CSV or JSON file with GSC data",
        type=['csv', 'json'],
        help="Required columns: date, query, page, clicks, impressions, ctr, position"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_json(uploaded_file)
            
            # Validate columns
            required_columns = ['date', 'query', 'page', 'clicks', 'impressions', 'ctr', 'position']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            # Convert date column
            data['date'] = pd.to_datetime(data['date'])
            
            # Data overview
            st.header("📊 Data Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", data['query'].nunique())
            with col2:
                st.metric("Total Pages", data['page'].nunique())
            with col3:
                st.metric("Date Range", f"{(data['date'].max() - data['date'].min()).days} days")
            with col4:
                st.metric("Total Clicks", f"{data['clicks'].sum():,}")
            
            # Date range warning
            date_range_days = (data['date'].max() - data['date'].min()).days
            if date_range_days < analyzer.thresholds['min_window_days']:
                st.warning(f"⚠️ Data window is only {date_range_days} days. Minimum {analyzer.thresholds['min_window_days']} days recommended for reliable analysis.")
            
            # Run analysis
            if st.button("🔍 Run Analysis", type="primary"):
                with st.spinner("Analyzing queries..."):
                    # Group by query and analyze
                    analysis_results = []
                    queries = data['query'].unique()
                    
                    progress_bar = st.progress(0)
                    for i, query in enumerate(queries):
                        query_data = data[data['query'] == query]
                        result = analyzer.analyze_query(query_data)
                        analysis_results.append(result)
                        progress_bar.progress((i + 1) / len(queries))
                    
                    progress_bar.empty()
                
                # Generate recommendations
                recommendations = analyzer.generate_recommendations(analysis_results)
                
                # Results summary
                st.header("📈 Analysis Results")
                
                cannibalization_count = len([r for r in analysis_results if r['classification'] == 'CANNIBALIZATION'])
                diversification_count = len([r for r in analysis_results if r['classification'] == 'DIVERSIFICATION'])
                neutral_count = len([r for r in analysis_results if r['classification'] == 'NEUTRAL'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 Cannibalization Issues", cannibalization_count)
                with col2:
                    st.metric("🟢 Diversification Wins", diversification_count)
                with col3:
                    st.metric("⚪ Neutral/Monitor", neutral_count)
                
                # Human-friendly brief
                st.header("📝 Executive Summary")
                
                brief_points = [
                    f"**Analysis Period:** {date_range_days} days ({data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')})",
                    f"**Total Queries Analyzed:** {len(queries):,}",
                    f"**Cannibalization Issues Found:** {cannibalization_count} queries need attention",
                    f"**Diversification Opportunities:** {diversification_count} queries performing well with multiple pages"
                ]
                
                if cannibalization_count > 0:
                    total_cannibal_clicks = sum(r['metrics']['total_clicks'] for r in analysis_results if r['classification'] == 'CANNIBALIZATION')
                    brief_points.append(f"**Potential Recovery:** Up to {total_cannibal_clicks * 0.3:.0f} additional clicks from fixing cannibalization")
                
                if diversification_count > 0:
                    total_diversification_clicks = sum(r['metrics']['total_clicks'] for r in analysis_results if r['classification'] == 'DIVERSIFICATION')
                    brief_points.append(f"**Diversification Value:** {total_diversification_clicks:.0f} clicks from healthy multi-page rankings")
                
                brief_points.extend([
                    "**Next Steps:** Review recommendations below and prioritise high-traffic fixes",
                    "**Owner:** SEO team should implement consolidation and optimisation strategies"
                ])
                
                for point in brief_points:
                    st.markdown(f"• {point}")
                
                # Detailed results tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🔴 Cannibalization", "🟢 Diversification", "📊 Visualizations", "📋 All Results"])
                
                with tab1:
                    st.subheader("Cannibalization Issues (Top 5 Fixes)")
                    if recommendations['cannibalization']:
                        for i, rec in enumerate(recommendations['cannibalization'], 1):
                            with st.expander(f"{i}. {rec['query']} - {rec['priority']} Priority"):
                                st.write(f"**Action:** {rec['action']}")
                                st.write(f"**Expected Uplift:** {rec['expected_uplift']}")
                                st.write("**Recommended Steps:**")
                                for detail in rec['details']:
                                    st.write(f"• {detail}")
                    else:
                        st.info("No cannibalization issues found.")
                
                with tab2:
                    st.subheader("Diversification Wins (Top 5)")
                    if recommendations['diversification']:
                        for i, rec in enumerate(recommendations['diversification'], 1):
                            with st.expander(f"{i}. {rec['query']}"):
                                st.write(f"**Action:** {rec['action']}")
                                st.write(f"**Current Benefit:** {rec['current_benefit']}")
                                st.write("**Optimisation Steps:**")
                                for detail in rec['details']:
                                    st.write(f"• {detail}")
                    else:
                        st.info("No diversification opportunities found.")
                
                with tab3:
                    st.subheader("Trend Visualizations")
                    
                    # Select queries to visualize
                    viz_queries = []
                    for result in analysis_results:
                        if result['classification'] in ['CANNIBALIZATION', 'DIVERSIFICATION']:
                            viz_queries.append(result['query'])
                    
                    if viz_queries:
                        selected_query = st.selectbox("Select Query to Visualize", viz_queries[:10])  # Limit to top 10
                        
                        if selected_query:
                            fig = analyzer.create_trend_visualization(data, selected_query)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No queries available for visualization.")
                
                with tab4:
                    st.subheader("Complete Analysis Results")
                    
                    # Create results dataframe
                    results_df = []
                    for result in analysis_results:
                        if 'metrics' in result:
                            row = {
                                'Query': result['query'],
                                'Classification': result['classification'],
                                'Total Clicks': result['metrics']['total_clicks'],
                                'Leader Changes': result['metrics']['leader_changes'],
                                'Simultaneous %': f"{result['metrics']['simultaneous_pct']:.1f}%",
                                'URL Similarity': f"{result['metrics']['jaccard_similarity']:.2f}",
                                'Top Page': result['metrics']['top_page'],
                                'Second Page': result['metrics']['second_page']
                            }
                        else:
                            row = {
                                'Query': result['query'],
                                'Classification': result['classification'],
                                'Reason': result.get('reason', 'N/A')
                            }
                        results_df.append(row)
                    
                    results_df = pd.DataFrame(results_df)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"gsc_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Sample data format
        st.header("📋 Data Format Requirements")
        st.markdown("""
        Your CSV or JSON file should contain the following columns:
        
        - **date**: Date in YYYY-MM-DD format
        - **query**: Search query/keyword
        - **page**: Full URL of the ranking page
        - **clicks**: Number of clicks
        - **impressions**: Number of impressions
        - **ctr**: Click-through rate (decimal)
        - **position**: Average position in search results
        
        **Minimum Requirements:**
        - At least 28 days of data (longer periods provide better insights)
        - One row per (date, query, page) combination
        - Data exported from Google Search Console
        """)
        
        # Sample data
        st.subheader("Sample Data Format")
        sample_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
            'query': ['seo tools', 'seo tools', 'seo tools', 'seo tools'],
            'page': ['https://example.com/seo-tools', 'https://example.com/best-seo-tools', 'https://example.com/seo-tools', 'https://example.com/best-seo-tools'],
            'clicks': [45, 23, 52, 18],
            'impressions': [1200, 800, 1350, 750],
            'ctr': [0.0375, 0.0288, 0.0385, 0.024],
            'position': [3.2, 5.8, 2.9, 6.1]
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
