import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Amazon EC2 Cost Analysis Dashboard")

st.write(
    "This dashboard analyzes Amazon EC2 instance costs and performance data."
)

# Load the dataset
data = pd.read_csv("ec2dataset.csv")

# Show basic dataset information
st.subheader("Dataset Overview")

st.write(f"Number of rows: {data.shape[0]}")
st.write(f"Number of columns: {data.shape[1]}")

# Show the first few rows
st.subheader("Dataset Preview")

st.dataframe(data.head())

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

st.subheader("Cost Distribution")

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=data[cost_columns],
    palette="Set2",
    ax=ax
)

ax.set_title("Cost Comparison of Amazon EC2 Instances (Hourly)")
ax.set_ylabel("Cost (USD)")
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

st.pyplot(fig)

st.subheader("On-Demand Cost Outliers")

# Function to identify outliers using the IQR method
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

# Find On-Demand outliers
outliers_on_demand = detect_outliers("On Demand")

st.write(f"Number of On-Demand outliers: {len(outliers_on_demand)}")

st.dataframe(
    outliers_on_demand[
        ['Name', 'API Name', 'On Demand']
    ]
)

st.subheader("Top 10 Lowest-Cost EC2 Instances")

cost_comparison = data[
    ['Name', 'On Demand', 'Linux Reserved cost']
].dropna().sort_values('On Demand')

st.dataframe(
    cost_comparison.head(10)
)

st.subheader("On-Demand vs Reserved Cost")

top10 = cost_comparison.head(10).set_index('Name')

st.bar_chart(
    top10[['On Demand', 'Linux Reserved cost']]
)

st.subheader("T2 vs T3 Instance Family Comparison")

# Filter T2 and T3 instance families
t2_instances = data[data['Name'].str.startswith('T2')]
t3_instances = data[data['Name'].str.startswith('T3')]

# Show summary statistics
st.write("T2 Instance Cost Summary")
st.dataframe(t2_instances[cost_columns].describe())

st.write("T3 Instance Cost Summary")
st.dataframe(t3_instances[cost_columns].describe())

# T2 boxplot
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


# T3 boxplot
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

st.subheader("EC2 On-Demand Cost Prediction Model")

# Convert Instance Memory to numeric
data['Instance Memory'] = pd.to_numeric(
    data['Instance Memory'].str.replace(' GiB', '')
)

# Convert vCPUs to numeric
data['vCPUs'] = pd.to_numeric(
    data['vCPUs'].str.extract(r'(\d+)', expand=False)
)

# Remove rows missing values needed for the regression model
data_cleaned = data.dropna(
    subset=['On Demand', 'Instance Memory', 'vCPUs']
)

# Define predictors and target
X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

st.write(f"Training samples: {len(X_train)}")
st.write(f"Testing samples: {len(X_test)}")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")

st.subheader("Actual vs Predicted On-Demand Costs")

fig_reg, ax_reg = plt.subplots(figsize=(8, 6))

ax_reg.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    color='b'
)

ax_reg.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linestyle='--'
)

ax_reg.set_title("Actual vs Predicted On-Demand Costs")
ax_reg.set_xlabel("Actual On-Demand Cost")
ax_reg.set_ylabel("Predicted On-Demand Cost")

plt.tight_layout()

st.pyplot(fig_reg)

st.subheader("Predict EC2 On-Demand Cost")

memory_input = st.number_input(
    "Instance Memory (GiB)",
    min_value=0.5,
    value=4.0,
    step=0.5
)

vcpu_input = st.number_input(
    "Number of vCPUs",
    min_value=1,
    value=2,
    step=1
)

if st.button("Predict Cost"):
    new_instance = pd.DataFrame(
        [[memory_input, vcpu_input]],
        columns=['Instance Memory', 'vCPUs']
    )

    predicted_cost = model.predict(new_instance)[0]

    st.write(
        f"Predicted On-Demand Cost: ${predicted_cost:.4f} per hour"
    )