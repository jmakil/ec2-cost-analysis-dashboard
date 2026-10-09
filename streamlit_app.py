import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="EC2 Cost Analysis",
    page_icon="☁️",
    layout="wide"
)

st.title("☁️ Amazon EC2 Cost Analysis Dashboard")
st.write(
    "Explore EC2 pricing trends, compare instance families, identify cost outliers, "
    "and estimate On-Demand pricing using a linear regression model."
)
st.divider()

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("ec2dataset.csv")


data = load_data()

# List of cost-related columns
cost_columns = [
    'On Demand',
    'Linux Reserved cost',
    'Linux Spot Minimum cost',
    'Windows On Demand cost',
    'Windows Reserved cost'
]

# Clean the cost columns
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

# Outlier analysis

def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]


outliers_on_demand = detect_outliers("On Demand")

# Lowest-cost comparison
cost_comparison = data[
    ['Name', 'On Demand', 'Linux Reserved cost']
].dropna().sort_values('On Demand')

top10 = cost_comparison.head(10).copy()

# Instance family filters

t2_instances = data[data['Name'].str.startswith('T2')]
t3_instances = data[data['Name'].str.startswith('T3')]

# Regression preparation
model_data = data.copy()
model_data['Instance Memory'] = pd.to_numeric(
    model_data['Instance Memory'].str.replace(' GiB', '')
)
model_data['vCPUs'] = pd.to_numeric(
    model_data['vCPUs'].str.extract(r'(\d+)', expand=False)
)

data_cleaned = model_data.dropna(
    subset=['On Demand', 'Instance Memory', 'vCPUs']
)

X = data_cleaned[['Instance Memory', 'vCPUs']]

# Predict the logarithm of On-Demand cost so transformed predictions stay positive
y = np.log(data_cleaned['On Demand'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict log-costs, then convert them back to dollars
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_test_actual = np.exp(y_test)

mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = mse ** 0.5

# Navigation tabs

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Cost Analysis",
    "Instance Families",
    "Prediction Model"
])

with tab1:
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    st.subheader("Cost Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average On-Demand Cost",
        f"${data['On Demand'].mean():.2f}"
    )
    col2.metric(
        "Lowest On-Demand Cost",
        f"${data['On Demand'].min():.4f}"
    )
    col3.metric(
        "Highest On-Demand Cost",
        f"${data['On Demand'].max():.2f}"
    )

with tab2:
    st.header("Cost Analysis")
    st.write(
        "This section compares EC2 pricing options and highlights unusually expensive instances."
    )

    st.subheader("Cost Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data[cost_columns], palette="Set2", ax=ax)
    ax.set_title("Cost Comparison of Amazon EC2 Instances (Hourly)")
    ax.set_ylabel("Cost (USD)")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Most EC2 prices are concentrated near the lower end, while a smaller number of "
        "high-cost instances appear as outliers."
    )

    st.subheader("On-Demand Cost Outliers")
    st.metric("Number of On-Demand Outliers", len(outliers_on_demand))

    with st.expander("View outlier instances"):
        st.dataframe(
            outliers_on_demand[['Name', 'API Name', 'On Demand']],
            use_container_width=True
        )

    st.subheader("Top 10 Lowest-Cost EC2 Instances")
    st.dataframe(top10, use_container_width=True)
    st.caption(
        "Linux Reserved pricing is lower than On-Demand pricing for the low-cost instances shown."
    )

    st.subheader("On-Demand vs Reserved Cost")
    chart_data = top10.set_index('Name')[['On Demand', 'Linux Reserved cost']]
    st.bar_chart(chart_data, stack=False)

with tab3:
    st.header("T2 vs T3 Instance Families")
    st.write(
        "Compare the cost distributions and summary statistics of T2 and T3 instances."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("T2 Summary")
        st.dataframe(
            t2_instances[cost_columns].describe(),
            use_container_width=True
        )

    with col2:
        st.subheader("T3 Summary")
        st.dataframe(
            t3_instances[cost_columns].describe(),
            use_container_width=True
        )

    st.subheader("T2 Cost Distribution")
    fig_t2, ax_t2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=t2_instances[cost_columns],
        palette="Blues",
        showmeans=True,
        ax=ax_t2
    )
    ax_t2.set_title("Cost Distribution for T2 Instances")
    ax_t2.set_ylabel("Cost (USD)")
    ax_t2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_t2)
    plt.close(fig_t2)

    st.subheader("T3 Cost Distribution")
    fig_t3, ax_t3 = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=t3_instances[cost_columns],
        palette="Greens",
        showmeans=True,
        ax=ax_t3
    )
    ax_t3.set_title("Cost Distribution for T3 Instances")
    ax_t3.set_ylabel("Cost (USD)")
    ax_t3.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_t3)
    plt.close(fig_t3)

with tab4:
    st.header("EC2 On-Demand Cost Prediction Model")
    st.write(
        "A linear regression model predicts On-Demand cost using instance memory and vCPUs."
    )

    info1, info2 = st.columns(2)
    info1.metric("Training Samples", len(X_train))
    info2.metric("Testing Samples", len(X_test))

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"{rmse:.4f}")

    st.caption(
        "Lower MAE, MSE, and RMSE values indicate better prediction performance."
    )

    st.subheader("Actual vs Predicted On-Demand Costs")
    fig_reg, ax_reg = plt.subplots(figsize=(8, 6))
    ax_reg.scatter(y_test_actual, y_pred, alpha=0.7, color='b')
    ax_reg.plot(
        [min(y_test_actual), max(y_test_actual)],
        [min(y_test_actual), max(y_test_actual)],
        color='red',
        linestyle='--'
    )
    ax_reg.set_title("Actual vs Predicted On-Demand Costs")
    ax_reg.set_xlabel("Actual On-Demand Cost")
    ax_reg.set_ylabel("Predicted On-Demand Cost")
    plt.tight_layout()
    st.pyplot(fig_reg)
    plt.close(fig_reg)

    st.subheader("Predict EC2 On-Demand Cost")

    input1, input2 = st.columns(2)

    with input1:
        memory_input = st.number_input(
            "Instance Memory (GiB)",
            min_value=0.5,
            value=4.0,
            step=0.5
        )

    with input2:
        vcpu_input = st.number_input(
            "Number of vCPUs",
            min_value=1,
            value=2,
            step=1
        )

    if st.button("Predict Cost", type="primary"):
        new_instance = pd.DataFrame(
            [[memory_input, vcpu_input]],
            columns=['Instance Memory', 'vCPUs']
        )

        predicted_log_cost = model.predict(new_instance)[0]
        predicted_cost = np.exp(predicted_log_cost)

        st.success(
            f"Predicted On-Demand Cost: ${predicted_cost:.4f} per hour"
        )
