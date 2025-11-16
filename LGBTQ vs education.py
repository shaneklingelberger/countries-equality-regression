import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm


# Load data
countries = pd.read_csv(r"C:\Users\Shane\Downloads\countries of the world.csv")
equality = pd.read_csv(r"C:\Users\Shane\Downloads\equaldex_equality_index.csv")   # must contain 'country' and 'lgbtq_equality'

# Remove missing values
countries = countries.dropna()
equality = equality.dropna()

# Remove trailing spaces from every row of the Country column in both dataframes
countries['Country'] = countries['Country'].str.rstrip()
equality['Country'] = equality['Country'].str.rstrip()

# Column of Analysis
countries_column_name = "GDP ($ per capita)"

# Replaces commas with periods only if values are strings initially
if countries[countries_column_name].dtype == "object":
    countries[countries_column_name] = countries[countries_column_name].str.replace(",", ".", regex=False).astype(float)

# Merge on the common column "country"
merged = pd.merge(countries, equality, on='Country', how='inner')

# Prepare variables for regression 
X = merged[[countries_column_name]]
Y = merged["EI"]

# Split training/test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Print out results to see which is the best model
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
print("R² score on test set:", model.score(X_test, Y_test))

x_vals = np.linspace(X.min(), X.max(), 100)
x_vals = pd.DataFrame(x_vals, columns=[countries_column_name])
y_vals = model.predict(x_vals)


# Plot
plt.figure(figsize=(8, 6))

# Scatter of actual data
plt.scatter(X, Y, label="Data Points")

# Regression line
plt.plot(x_vals, y_vals, label="Regression Line")

plt.xlabel(countries_column_name)
plt.ylabel("LGBTQ Equality")
plt.title("Linear Regression: " + countries_column_name + " vs LGBTQ Equality")
plt.legend()
plt.grid(True)
plt.show()

X_test_df = pd.DataFrame(X_test, columns=[countries_column_name])
model.predict(X_test_df)


# If you only want the p-value for the slope:
# FIX: sklearn does not compute p-values; must use statsmodels
X_sm = sm.add_constant(X)
model_sm = sm.OLS(Y, X_sm).fit()
p_value_slope = model_sm.pvalues[1]   # index 1 = slope term
print("Slope p-value:", p_value_slope)
