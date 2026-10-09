import pandas as pd

# Load the dataset
file_path = 'ec2dataset.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print(data.info())

# Preview the first few rows
print(data.head())

# List of cost-related columns
cost_columns = [
    'On Demand',
    'Linux Reserved cost',
    'Linux Spot Minimum cost',
    'Windows On Demand cost',
    'Windows Reserved cost'
]

# Remove '$', commas, and the word 'hourly', then convert to numbers
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace('[$, hourly]', '', regex=True),
        errors='coerce'
    )

# Check for missing values after conversion
print(data[cost_columns].isnull().sum())

# Generate summary statistics for cost-related columns
cost_summary = data[cost_columns].describe()

print(cost_summary)

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plotting style
sns.set(style="whitegrid")

# Create a boxplot to visualize the distribution of costs
plt.figure(figsize=(12, 6))

sns.boxplot(data=data[cost_columns], palette="Set2")

# Set the plot labels and title
plt.title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

# Display the plot
plt.tight_layout()
plt.savefig('ec2_cost_boxplot.png')
plt.show()

# Function to identify outliers using the IQR method
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Find outliers in the On-Demand cost column
outliers_on_demand = detect_outliers('On Demand')

print(outliers_on_demand)

# Compare Reserved and On-Demand costs
cost_comparison = data[
    ['Name', 'On Demand', 'Linux Reserved cost']
].dropna().sort_values('On Demand')

print(cost_comparison.head(10))

# Filter for specific instance families
def filter_instance_family(family):
    return data[data['Name'].str.startswith(family)]

# Example: Filter for T2 and T3 instance families
t2_instances = filter_instance_family('T2')
t3_instances = filter_instance_family('T3')

# Compare summary statistics for T2 and T3 instances
t2_summary = t2_instances[cost_columns].describe()
t3_summary = t3_instances[cost_columns].describe()

print("T2 Instance Costs Summary:\n", t2_summary)
print("\nT3 Instance Costs Summary:\n", t3_summary)

# Visualize cost comparison for T2 instances
plt.figure(figsize=(12, 6))
sns.boxplot(data=t2_instances[cost_columns], palette="Blues", showmeans=True)

plt.title('Cost Distribution for T2 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.tight_layout()
plt.savefig('t2_cost_distribution.png')
plt.show()

# Visualize cost comparison for T3 instances
plt.figure(figsize=(12, 6))
sns.boxplot(data=t3_instances[cost_columns], palette="Greens", showmeans=True)

plt.title('Cost Distribution for T3 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.tight_layout()
plt.savefig('t3_cost_distribution.png')
plt.show()


# Compare On-Demand and Reserved costs for T2 and T3 families
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]
])

# Sort by On-Demand costs
comparison_sorted = comparison.dropna().sort_values('On Demand')

print(comparison_sorted.head(10))

# Convert Instance Memory from text such as "0.5 GiB" into a number
data['Instance Memory'] = pd.to_numeric(
    data['Instance Memory'].str.replace(' GiB', '')
)

# Extract the number from vCPUs, such as "2 vCPUs" -> 2
data['vCPUs'] = pd.to_numeric(
    data['vCPUs'].str.extract(r'(\d+)', expand=False)
)

# Check the converted values
print(data[['Instance Memory', 'vCPUs']].head())

# Drop rows with missing values in the relevant columns
data_cleaned = data.dropna(
    subset=['On Demand', 'Instance Memory', 'vCPUs']
)

# Verify that the cleaned data has no missing values
print(data_cleaned.isnull().sum())

from sklearn.model_selection import train_test_split

# Define features and target
X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']

# Split the data into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the coefficients of the model
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict the costs for the test data
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7, color='b')

plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linestyle='--'
)

plt.title('Actual vs Predicted On-Demand Costs')
plt.xlabel('Actual On-Demand Cost')
plt.ylabel('Predicted On-Demand Cost')

plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Example: Predict cost for a new instance with 4 GiB memory and 2 vCPUs
new_instance = [[4, 2]]

predicted_cost = model.predict(new_instance)

print(f"Predicted On-Demand Cost for 4 GiB, 2 vCPUs: ${predicted_cost[0]:.4f}")