# GSC Cannibalization vs Diversification Analyzer

A Streamlit tool that analyzes Google Search Console data to identify keyword cannibalization issues and healthy diversification opportunities, providing actionable recommendations for SEO improvement.

## Features

### 🔍 **Intelligent Analysis**
- **Cannibalization Detection**: Identifies harmful keyword cannibalization where multiple pages compete for the same query
- **Diversification Recognition**: Recognises beneficial multi-page rankings that capture different user intents
- **Trend Analysis**: Uses line charts to visualise ranking patterns and detect swapping behaviour

### 🔗 **Dual Data Sources**
- **File Upload**: Traditional CSV/JSON file upload for offline analysis
- **Google Search Console API**: Direct OAuth integration for real-time GSC data fetching
- **Automatic Authentication**: Secure OAuth flow with credential persistence

### 📊 **Comprehensive Metrics**
- Leader change tracking (page swapping detection)
- Simultaneous ranking analysis
- URL similarity scoring using Jaccard similarity
- Traffic share and uplift calculations
- Stability metrics across time periods

### 🎯 **Actionable Recommendations**
- Prioritised list of cannibalization fixes with expected upside
- Diversification opportunities to maintain and expand
- Specific action items (301 redirects, content consolidation, etc.)
- Client-friendly executive summary

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the application**:
```bash
streamlit run gsc_analyzer.py
```

2. **Prepare your GSC data**:
   - Export data from Google Search Console
   - Ensure data spans at least 28 days (longer periods provide better insights)
   - Format as CSV or JSON with required columns

3. **Choose your data source**:
   - **Option A - File Upload**: Upload CSV/JSON file through the web interface
   - **Option B - Google Search Console**: Use OAuth to connect directly to your GSC account
   
4. **Configure and analyse**:
   - Adjust analysis thresholds in the sidebar if needed
   - For GSC API: Select your website property and date range
   - Click "Run Analysis" to process your data

## Google Search Console API Setup

### Prerequisites
1. **Google Cloud Project**: Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. **Enable Search Console API**: Enable the Google Search Console API for your project
3. **OAuth 2.0 Credentials**: Create OAuth 2.0 client credentials (Web application type)
4. **Download Credentials**: Download the client secret JSON file

### Setup Steps
1. Place your `client_secret_*.json` file in the same directory as `gsc_analyzer.py`
2. The filename should match exactly: `client_secret_483732917438-avvch65f4jrtvklqhqksvvjsu0k4gq4h.apps.googleusercontent.com.json`
3. Run the application and select "🔗 Connect to Google Search Console"
4. Follow the OAuth flow to authorize access to your GSC data

### Security Notes
- Credentials are stored locally in `token.pickle` for reuse
- The `.gitignore` file protects sensitive files from being committed
- Only read-only access to Search Console data is requested

## Data Requirements (File Upload)

Your CSV or JSON file must contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `date` | Date in YYYY-MM-DD format | 2024-01-01 |
| `query` | Search query/keyword | "seo tools" |
| `page` | Full URL of ranking page | https://example.com/seo-tools |
| `clicks` | Number of clicks | 45 |
| `impressions` | Number of impressions | 1200 |
| `ctr` | Click-through rate (decimal) | 0.0375 |
| `position` | Average position in search results | 3.2 |

### Data Format
- One row per (date, query, page) combination
- Minimum 28 days of data recommended
- Longer time periods provide more reliable insights

## Analysis Logic

### Cannibalization Detection
A query is flagged as **CANNIBALIZATION** if ALL conditions are met:
- **Swapping detected**: Top-ranking page changes ≥3 times
- **Similar intent**: URL similarity (Jaccard) ≥ 0.6
- **Poor co-ranking**: Either simultaneous ranking <30% of days OR second page traffic share <15%

### Diversification Recognition
A query is flagged as **DIVERSIFICATION** if ALL conditions are met:
- **Frequent simultaneous ranking**: Occurs on ≥30% of dates
- **Different intents**: URL similarity (Jaccard) < 0.6
- **Meaningful contribution**: Second page provides ≥15% uplift in traffic

### Configurable Thresholds
- Leader changes threshold: 3 (adjustable 2-10)
- Similar intent Jaccard: 0.6 (adjustable 0.3-0.9)
- Simultaneous ranking percentage: 30% (adjustable 10-50%)
- Minimum second page share: 15% (adjustable 5-30%)
- Minimum uplift: 0.15 (adjustable 0.05-0.5)

## Output Features

### 📈 **Visual Analysis**
- Interactive trend charts showing position and click patterns
- Multi-page ranking visualisations
- Time-series analysis for detecting swapping behaviour

### 📝 **Executive Summary**
- Client-friendly bullet-point summary
- Key metrics and findings
- Prioritised action items
- Expected impact estimates

### 📊 **Detailed Results**
- Complete analysis results in tabular format
- Downloadable CSV export
- Classification reasoning for each query
- Comprehensive metrics breakdown

### 🎯 **Recommendations**
- **Cannibalization Fixes**: Top 5 issues with specific consolidation strategies
- **Diversification Wins**: Top 5 opportunities to maintain and expand
- **Action Priority**: High/Medium priority based on traffic impact
- **Expected Uplift**: Quantified improvement estimates

## URL Similarity Logic

The tool uses URL tokenisation for intent similarity:
1. Extract path from URL
2. Split on common separators (/, -, _)
3. Remove stopwords: {the, and, a, an, to, for, of, in}
4. Calculate Jaccard similarity of token sets
5. ≥0.6 similarity = same/similar intent
6. <0.6 similarity = different intent

## Best Practices

### Data Collection
- Export at least 28 days of data (90+ days ideal)
- Include all queries and pages (don't filter)
- Use consistent date ranges for comparison
- Regular analysis (monthly/quarterly) for ongoing optimisation

### Interpretation
- Focus on high-traffic cannibalization issues first
- Preserve diversification that's working well
- Consider seasonal patterns in your analysis
- Validate recommendations with manual SERP analysis

### Implementation
- Implement 301 redirects carefully with proper testing
- Monitor rankings after consolidation changes
- Update internal linking to support chosen primary pages
- Track performance improvements over time

## Technical Notes

- Built with Streamlit for easy web interface
- Uses Plotly for interactive visualisations
- Pandas for efficient data processing
- Configurable thresholds for different business contexts
- Supports both CSV and JSON input formats

## Troubleshooting

**"Missing required columns" error**: Ensure your data file contains all required columns with exact names.

**"Insufficient data" warning**: Analysis requires at least 28 days of data for reliable results.

**No visualisations showing**: Check that queries have been classified as cannibalization or diversification.

**Large file processing slowly**: Consider filtering to top queries before upload, or analysing in smaller date ranges.

## Support

For issues or feature requests, please check that:
1. Your data format matches the requirements exactly
2. You have sufficient data volume (28+ days)
3. Column names match the specification precisely
4. Date format is YYYY-MM-DD

---

**Built with Australian English spelling preferences** 🇦🇺
