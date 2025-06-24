# KPI Analytics Dashboard

A comprehensive cross-selling analytics dashboard built with Streamlit and Python, designed to analyze customer behavior, retention, and business performance across different business lines.

## Features

- **Cross-Selling Analysis**: Track customer purchasing patterns across multiple business lines
- **Cohort Analysis**: Analyze customer retention and behavior over time
- **Temporal Analysis**: Visualize trends and patterns across different time periods
- **Business Line Performance**: Compare performance metrics across different business segments
- **Interactive Visualizations**: Dynamic charts and graphs using Plotly
- **Data Export**: Export analysis results for further processing

## Installation

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd kpi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or use the installation script:
```bash
chmod +x install_deps.sh
./install_deps.sh
```

## Usage

Run the Streamlit application:
```bash
streamlit run kpi11.py
```

The dashboard will be available at `http://localhost:8501`

## Data Format

The application expects an Excel file (`sale_data.xlsx`) with the following columns:
- Customer identification
- Transaction dates
- Business line information
- Sales amounts
- Product categories

## Project Structure

- `kpi11.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `sale_data.xlsx` - Sample data file
- `install_deps.sh` - Installation script
- Other KPI files - Various iterations and versions of the analysis

## Technologies Used

- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computations
- **OpenPyXL** - Excel file handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Commit and push to your branch
5. Create a pull request

## License

This project is open source and available under the MIT License. 