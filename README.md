# Polynomial Regression Sales Predictor

This project predicts product sales using advertising budgets for TV, Radio, and Newspaper.

## Model Used
Polynomial Regression using:
- PolynomialFeatures
- StandardScaler
- LinearRegression

## Dataset
Advertising Dataset with columns:
- TV
- Radio
- Newspaper
- Sales

The Streamlit app can automatically load a public Advertising dataset. You can also upload your own `Advertising.csv`.

## How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit app
```bash
streamlit run app.py
```

## Project Benefits
- Optimize advertising budgets for maximum ROI
- Identify overspending and avoid diminishing returns
- Allocate budgets effectively across TV, Radio, and Newspaper
- Make data-driven marketing decisions quickly
- Understand non-linear relationships in real-world sales data
