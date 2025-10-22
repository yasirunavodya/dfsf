# 🌞 Solar Power Forecasting Dashboard

A machine learning-powered web application for predicting solar power output using Random Forest algorithms. This dashboard provides an intuitive interface for uploading data, visualizing inputs, running forecasts, and analyzing results.

## 🚀 Features

- **📁 Dataset Upload**: Easy CSV file upload with data preview
- **📊 Data Visualization**: Comprehensive input data analysis and visualization
- **🔮 Forecasting**: 3-hour ahead solar power prediction using Random Forest
- **📈 Results Visualization**: Interactive charts and statistical analysis
- **💾 Export**: Download forecast results as CSV

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## 🛠️ Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Required Dependencies

The application requires the following Python packages:

- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning library
- `joblib` - Model serialization
- `fastapi` - API framework (for backend)
- `uvicorn` - ASGI server
- `pydantic` - Data validation

## 🚀 Running the Application

### Option 1: Web Dashboard (Recommended)

Run the Streamlit web interface:

```bash
streamlit run ui.py
```
