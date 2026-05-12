import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Advertising Sales Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Polynomial Regression Sales Predictor")
st.write(
    """
    This web app predicts product sales based on advertising budgets for **TV**, **Radio**, 
    and **Newspaper** using a Polynomial Regression model. 

    The model captures non-linear behavior such as **diminishing returns**, where increasing 
    advertising budget beyond a certain level may produce smaller increases in sales.
    """
)

st.markdown("---")


@st.cache_data
def load_default_dataset():
    """
    Load the Advertising dataset from a public GitHub source.
    If internet is unavailable, the user can upload Advertising.csv manually.
    """
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    df = pd.read_csv(url)

    # Some versions contain an unnecessary index column.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def clean_columns(df):
    """
    Standardize column names and remove unnecessary index columns.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def train_polynomial_model(df, degree):
    """
    Train a Polynomial Regression model using TV, Radio, and Newspaper as input features.
    """
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline(steps=[
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear_regression", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


with st.sidebar:
    st.header("⚙️ Project Settings")

    uploaded_file = st.file_uploader("Upload Advertising.csv", type=["csv"])
    degree = st.slider(
        "Polynomial Degree", 
        min_value=1, 
        max_value=5, 
        value=2,
        help="Degree 2 is usually a good starting point for capturing non-linear patterns."
    )

    st.info(
        "Use degree 2 or 3 for better generalization. Very high degrees may overfit."
    )


try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = clean_columns(df)
    else:
        df = load_default_dataset()
        df = clean_columns(df)
except Exception as e:
    st.error("Dataset could not be loaded automatically. Please upload Advertising.csv manually.")
    st.stop()


required_columns = {"TV", "Radio", "Newspaper", "Sales"}
if not required_columns.issubset(set(df.columns)):
    st.error(
        "The dataset must contain these columns: TV, Radio, Newspaper, Sales"
    )
    st.write("Current columns:", df.columns.tolist())
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset", 
    "🤖 Model Performance", 
    "🔮 Predict Sales", 
    "📉 Diminishing Returns"
])


with tab1:
    st.subheader("Advertising Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape")
        st.write(df.shape)
    with col2:
        st.write("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)


model, X_train, X_test, y_train, y_test, y_pred, metrics = train_polynomial_model(df, degree)


with tab2:
    st.subheader("Model Evaluation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['MAE']:.3f}")
    m2.metric("MSE", f"{metrics['MSE']:.3f}")
    m3.metric("RMSE", f"{metrics['RMSE']:.3f}")
    m4.metric("R² Score", f"{metrics['R2']:.3f}")

    st.write(
        """
        **Interpretation:**  
        A higher R² score means the model explains more variation in sales. 
        Lower MAE and RMSE values mean the prediction error is smaller.
        """
    )

    results_df = pd.DataFrame({
        "Actual Sales": y_test.values,
        "Predicted Sales": y_pred
    })

    st.subheader("Actual vs Predicted Sales")
    st.dataframe(results_df.head(20), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)


with tab3:
    st.subheader("Enter Advertising Budgets")

    col1, col2, col3 = st.columns(3)

    with col1:
        tv_budget = st.number_input(
            "TV Budget", 
            min_value=0.0, 
            value=float(df["TV"].median()), 
            step=1.0
        )

    with col2:
        radio_budget = st.number_input(
            "Radio Budget", 
            min_value=0.0, 
            value=float(df["Radio"].median()), 
            step=1.0
        )

    with col3:
        newspaper_budget = st.number_input(
            "Newspaper Budget", 
            min_value=0.0, 
            value=float(df["Newspaper"].median()), 
            step=1.0
        )

    custom_input = pd.DataFrame({
        "TV": [tv_budget],
        "Radio": [radio_budget],
        "Newspaper": [newspaper_budget]
    })

    predicted_sales = model.predict(custom_input)[0]

    st.success(f"Predicted Sales: {predicted_sales:.2f}")

    total_budget = tv_budget + radio_budget + newspaper_budget
    if total_budget > 0:
        st.write(f"Total Advertising Budget: {total_budget:.2f}")
        st.write(f"Predicted Sales per Budget Unit: {predicted_sales / total_budget:.4f}")

    st.write(
        """
        This prediction helps marketers estimate expected sales before spending money on advertising.
        """
    )


with tab4:
    st.subheader("Diminishing Returns Analysis")

    channel = st.selectbox(
        "Select advertising channel to analyze:",
        ["TV", "Radio", "Newspaper"]
    )

    st.write(
        """
        In this graph, only one selected advertising channel changes while the other two channels 
        remain fixed at their median values. A curve that becomes flatter at high spending levels 
        indicates diminishing returns.
        """
    )

    min_budget = float(df[channel].min())
    max_budget = float(df[channel].max())

    budget_range = np.linspace(min_budget, max_budget, 100)

    scenario = pd.DataFrame({
        "TV": np.full(100, df["TV"].median()),
        "Radio": np.full(100, df["Radio"].median()),
        "Newspaper": np.full(100, df["Newspaper"].median())
    })

    scenario[channel] = budget_range
    sales_curve = model.predict(scenario)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(budget_range, sales_curve)
    ax2.set_xlabel(f"{channel} Advertising Budget")
    ax2.set_ylabel("Predicted Sales")
    ax2.set_title(f"Diminishing Returns Curve for {channel}")
    st.pyplot(fig2)

    marginal_gain = np.diff(sales_curve)
    avg_initial_gain = np.mean(marginal_gain[:20])
    avg_final_gain = np.mean(marginal_gain[-20:])

    st.write("### Marginal Return Explanation")
    st.write(f"Average sales gain at lower {channel} spending: **{avg_initial_gain:.4f}**")
    st.write(f"Average sales gain at higher {channel} spending: **{avg_final_gain:.4f}**")

    if avg_final_gain < avg_initial_gain:
        st.warning(
            f"The model suggests diminishing returns for {channel}: higher spending gives smaller additional sales gains."
        )
    else:
        st.info(
            f"The model does not show strong diminishing returns for {channel} under the current settings."
        )


st.markdown("---")
st.write(
    """
    ### Business Value
    This project helps users optimize advertising budgets, avoid overspending, 
    allocate budget across multiple channels, and make fast data-driven marketing decisions.
    """
)
