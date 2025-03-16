#!/usr/bin/env python
# coding: utf-8

# # ALI

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'Data Assignment - Ali.csv'
df = pd.read_csv(file_path)

# Fix Date format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# Fill missing values
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
df['Leads'] = imputer_mean.fit_transform(df[['Leads']])
df['Time spent on LG (mins)'] = imputer_mean.fit_transform(df[['Time spent on LG (mins)']])
df['Avg Time Per Lead (mins)'] = imputer_median.fit_transform(df[['Avg Time Per Lead (mins)']])
df['Daily Team Review'].fillna(df['Daily Team Review'].mode()[0], inplace=True)
df['No. of Incomplete Leads'] = imputer_median.fit_transform(df[['No. of Incomplete Leads']])

# Encode categorical column
df['Daily Team Review'] = df['Daily Team Review'].map({'Attended': 1, 'Missed': 0})

# Winsorize to handle outliers (EXCLUDING target variable)
df['Time spent on LG (mins)'] = winsorize(df['Time spent on LG (mins)'], limits=[0.05, 0.05])
df['Avg Time Per Lead (mins)'] = winsorize(df['Avg Time Per Lead (mins)'], limits=[0.05, 0.05])
df['No. of Incomplete Leads'] = winsorize(df['No. of Incomplete Leads'], limits=[0.05, 0.05])

# Save cleaned file
df.to_csv('Data_Assignment_Ali_CLEANED.csv', index=False)

# ==== MODEL TRAINING ====
# Feature selection
X = df[['Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads', 'Daily Team Review']]
y = df['Leads']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_data = pd.DataFrame({
    'Time spent on LG (mins)': np.mean(df['Time spent on LG (mins)']) + np.random.normal(0, 5, 30),
    'Avg Time Per Lead (mins)': np.mean(df['Avg Time Per Lead (mins)']) + np.random.normal(0, 1, 30),
    'No. of Incomplete Leads': np.mean(df['No. of Incomplete Leads']) + np.random.normal(0, 2, 30),
    'Daily Team Review': [1 if i % 2 == 0 else 0 for i in range(30)]
})

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

# Combine predictions with dates
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Leads': future_predictions
})

print(predicted_df)

# Model accuracy
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.4f}')

# ==== VISUALIZATIONS ====
# 1. Histogram - Distribution of Leads
plt.figure(figsize=(8, 5))
sns.histplot(df['Leads'], bins=20, kde=True, color='blue')
plt.title('Distribution of Leads')
plt.xlabel('Leads')
plt.ylabel('Frequency')
plt.show()

# 2. Box Plot - Detecting Outliers in Leads
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Leads'], color='red')
plt.title('Box Plot of Leads')
plt.ylabel('Leads')
plt.show()

# 3. Scatter Plot - Relationship between Time Spent & Leads
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Time spent on LG (mins)'], y=df['Leads'], color='green')
plt.title('Scatter Plot: Time Spent vs. Leads')
plt.xlabel('Time Spent on LG (mins)')
plt.ylabel('Leads')
plt.show()

# 4. Line Graph - Leads over Time
df = df.sort_values(by='Date')
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Leads'], marker='o', linestyle='-', color='purple')
plt.title('Leads Over Time')
plt.xlabel('Date')
plt.ylabel('Leads')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 5. Heatmap - Correlation Matrix
numerical_cols = ['Leads', 'Time spent on LG (mins)', 'Avg Time Per Lead (mins)']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 6. Violin Plot - Leads Distribution
plt.figure(figsize=(8, 5))
sns.violinplot(y=df['Leads'], color='skyblue', inner='quartile')
plt.title('Violin Plot of Leads')
plt.ylabel('Leads')
plt.show()

# 7. Pair Plot - Numerical Columns
sns.pairplot(df[numerical_cols])
plt.suptitle('Pair Plot of Key Variables', y=1.02)
plt.show()

# 8. KDE Plot - Density Distribution of Leads
plt.figure(figsize=(8, 5))
sns.kdeplot(df['Leads'], shade=True, color='orange')
plt.title('KDE Plot of Leads')
plt.xlabel('Leads')
plt.ylabel('Density')
plt.show()

# 9. Bar Plot - Average Leads per Day of the Week
plt.figure(figsize=(8, 5))
sns.barplot(x=df['Date'].dt.day_name(), y=df['Leads'], estimator=np.mean, palette='viridis')
plt.title('Average Leads by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Leads')
plt.xticks(rotation=45)
plt.show()

# 10. Rolling Mean of Leads
df['Rolling Leads'] = df['Leads'].rolling(window=7, min_periods=1).mean()
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Leads'], label='Original Leads', alpha=0.5)
plt.plot(df['Date'], df['Rolling Leads'], label='7-Day Rolling Mean', color='red')
plt.title('Rolling Mean of Leads (7-Day Window)')
plt.xlabel('Date')
plt.ylabel('Leads')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# ✅ All 10 visualizations added!


# In[23]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df["Leads"], model="additive", period=7)
decomposition.plot()
plt.show()
df["Week"] = df["Date"].dt.isocalendar().week
pivot_table = df.pivot_table(values="Leads", index="Week", columns="Day", aggfunc="mean")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Heatmap of Leads Across Weeks and Days")
plt.xlabel("Day of the Week")
plt.ylabel("Week Number")
plt.show()
df["Rolling Leads"] = df["Leads"].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Leads"], label="Original Leads", alpha=0.5)
plt.plot(df["Date"], df["Rolling Leads"], label="7-Day Rolling Mean", color="red")
plt.title("Rolling Mean of Leads (7-Day Window)")
plt.xlabel("Date")
plt.ylabel("Leads")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(10, 5))
autocorrelation_plot(df["Leads"])
plt.title("Autocorrelation Plot for Leads")
plt.show()
plt.figure(figsize=(8, 5))
sns.violinplot(x=df["Day"], y=df["Leads"], inner="quartile", palette="muted")
plt.title("Leads Distribution Across Different Days")
plt.xlabel("Day of the Week")
plt.ylabel("Leads")
plt.show()
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=df["Time spent on LG (mins)"],
    y=df["Leads"],
    z=df["Avg Time Per Lead (mins)"],
    mode="markers",
    marker=dict(size=5, color=df["Leads"], colorscale="Viridis")
)])

fig.update_layout(title="3D Scatter Plot (Leads vs Time Spent)",
                  scene=dict(xaxis_title="Time Spent on LG (mins)",
                             yaxis_title="Leads",
                             zaxis_title="Avg Time Per Lead (mins)"))
fig.show()


# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'Data Assignment - Ali.csv'
df = pd.read_csv(file_path)

# Fix Date format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# Fill missing values
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
df['Leads'] = imputer_mean.fit_transform(df[['Leads']])
df['Time spent on LG (mins)'] = imputer_mean.fit_transform(df[['Time spent on LG (mins)']])
df['Avg Time Per Lead (mins)'] = imputer_median.fit_transform(df[['Avg Time Per Lead (mins)']])
df['Daily Team Review'].fillna(df['Daily Team Review'].mode()[0], inplace=True)
df['No. of Incomplete Leads'] = imputer_median.fit_transform(df[['No. of Incomplete Leads']])

# Encode categorical column
df['Daily Team Review'] = df['Daily Team Review'].map({'Attended': 1, 'Missed': 0})

# Winsorize to handle outliers (EXCLUDING target variable)
df['Time spent on LG (mins)'] = winsorize(df['Time spent on LG (mins)'], limits=[0.05, 0.05])
df['Avg Time Per Lead (mins)'] = winsorize(df['Avg Time Per Lead (mins)'], limits=[0.05, 0.05])
df['No. of Incomplete Leads'] = winsorize(df['No. of Incomplete Leads'], limits=[0.05, 0.05])

# Save cleaned file
df.to_csv('Data_Assignment_Ali_CLEANED.csv', index=False)

# ==== MODEL TRAINING ====
# Feature selection
X = df[['Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads', 'Daily Team Review']]
y = df['Leads']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_data = pd.DataFrame({
    'Time spent on LG (mins)': np.mean(df['Time spent on LG (mins)']) + np.random.normal(0, 5, 30),
    'Avg Time Per Lead (mins)': np.mean(df['Avg Time Per Lead (mins)']) + np.random.normal(0, 1, 30),
    'No. of Incomplete Leads': np.mean(df['No. of Incomplete Leads']) + np.random.normal(0, 2, 30),
    'Daily Team Review': [1 if i % 2 == 0 else 0 for i in range(30)]
})

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

# Combine predictions with dates
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Leads': future_predictions
})

print(predicted_df)

# Model accuracy
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.4f}')

# ==== ANALYSIS ====
# 1. Lead Generation Efficiency
df['Lead Generation Efficiency'] = df['Leads'] / df['Time spent on LG (mins)']
most_efficient = df.loc[df['Lead Generation Efficiency'].idxmax(), 'Lead Generation Efficiency']
print(f'Highest Efficiency Value: {most_efficient:.2f}')

# 2. Daily Performance Variability
variability = df['Leads'].std()
print(f'Highest Variability: {variability:.2f}')

# 3. Time Management Analysis
correlation = df['Avg Time Per Lead (mins)'].corr(df['Leads'])
print(f'Time vs Leads Correlation: {correlation:.2f}')

# 4. Performance with/without Daily Review
avg_with_review = df.loc[df['Daily Team Review'] == 1, 'Leads'].mean()
avg_without_review = df.loc[df['Daily Team Review'] == 0, 'Leads'].mean()
percentage_difference = ((avg_with_review - avg_without_review) / avg_without_review) * 100
print(f'Percentage Difference in Leads: {percentage_difference:.2f}%')

# 5. Incomplete Leads Reduction Over Time
incomplete_trend = LinearRegression()
incomplete_trend.fit(df.index.values.reshape(-1, 1), df['No. of Incomplete Leads'])
trend_slope = incomplete_trend.coef_[0]
print(f'Incomplete Leads Trend: {trend_slope:.2f}')

# 6. Performance Consistency
cv = df['Leads'].std() / df['Leads'].mean()
print(f'Coefficient of Variation (Consistency): {cv:.2f}')

# 7. High-Performance Days
top_10_percent = df['Leads'].quantile(0.9)
high_perf_days = df[df['Leads'] >= top_10_percent]
avg_time_on_high_perf_days = high_perf_days['Time spent on LG (mins)'].mean()
print(f'Avg Time on High-Performance Days: {avg_time_on_high_perf_days:.2f}')

# 8. Impact of Longer Lead Time
threshold = df['Time spent on LG (mins)'].quantile(0.75)
avg_leads_above_threshold = df.loc[df['Time spent on LG (mins)'] > threshold, 'Leads'].mean()
print(f'Average Leads Above Time Threshold: {avg_leads_above_threshold:.2f}')

# 9. Weekday vs Weekend Performance
df['Day Type'] = df['Date'].dt.dayofweek
weekday_avg = df.loc[df['Day Type'] < 5, 'Leads'].mean()
weekend_avg = df.loc[df['Day Type'] >= 5, 'Leads'].mean()
print(f'Weekday Avg: {weekday_avg:.2f}, Weekend Avg: {weekend_avg:.2f}')

# 10. Predictive Analysis
predicted_leads = model.predict(X_scaled)
accuracy = r2_score(y, predicted_leads)
print(f'Predictive Model R²: {accuracy:.4f}')

# ✅ All 10 analytical questions completed!


# In[28]:


from sklearn.model_selection import train_test_split

X = df[['Time spent on LG (mins)']]
y = df['Leads']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression().fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Model accuracy using R² score
accuracy = r2_score(y_test, y_pred)
print("Model R2 Score:", accuracy)


# # RAJ

# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'Data Assignment - Raj.csv'
df = pd.read_csv(file_path)

# Fix Date format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# Fill missing values
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
df['Leads'] = imputer_mean.fit_transform(df[['Leads']])
df['Time spent on LG (mins)'] = imputer_mean.fit_transform(df[['Time spent on LG (mins)']])
df['Avg Time Per Lead (mins)'] = imputer_median.fit_transform(df[['Avg Time Per Lead (mins)']])
df['Daily Team Review'].fillna(df['Daily Team Review'].mode()[0], inplace=True)
df['No. of Incomplete Leads'] = imputer_median.fit_transform(df[['No. of Incomplete Leads']])

# Encode categorical column
df['Daily Team Review'] = df['Daily Team Review'].map({'Attended': 1, 'Missed': 0})

# Winsorize to handle outliers (EXCLUDING target variable)
df['Time spent on LG (mins)'] = winsorize(df['Time spent on LG (mins)'], limits=[0.05, 0.05])
df['Avg Time Per Lead (mins)'] = winsorize(df['Avg Time Per Lead (mins)'], limits=[0.05, 0.05])
df['No. of Incomplete Leads'] = winsorize(df['No. of Incomplete Leads'], limits=[0.05, 0.05])

# Save cleaned file
df.to_csv('Data_Assignment_Raj_CLEANED.csv', index=False)

# ==== MODEL TRAINING ====
# Feature selection
X = df[['Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads', 'Daily Team Review']]
y = df['Leads']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_data = pd.DataFrame({
    'Time spent on LG (mins)': np.mean(df['Time spent on LG (mins)']) + np.random.normal(0, 5, 30),
    'Avg Time Per Lead (mins)': np.mean(df['Avg Time Per Lead (mins)']) + np.random.normal(0, 1, 30),
    'No. of Incomplete Leads': np.mean(df['No. of Incomplete Leads']) + np.random.normal(0, 2, 30),
    'Daily Team Review': [1 if i % 2 == 0 else 0 for i in range(30)]
})

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

# Combine predictions with dates
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Leads': future_predictions
})

print(predicted_df)

# Model accuracy
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.4f}')

# ==== VISUALIZATIONS ====
# 1. Histogram - Distribution of Leads
plt.figure(figsize=(8, 5))
sns.histplot(df['Leads'], bins=20, kde=True, color='blue')
plt.title('Distribution of Leads')
plt.xlabel('Leads')
plt.ylabel('Frequency')
plt.show()

# 2. Box Plot - Detecting Outliers in Leads
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Leads'], color='red')
plt.title('Box Plot of Leads')
plt.ylabel('Leads')
plt.show()

# 3. Scatter Plot - Relationship between Time Spent & Leads
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Time spent on LG (mins)'], y=df['Leads'], color='green')
plt.title('Scatter Plot: Time Spent vs. Leads')
plt.xlabel('Time Spent on LG (mins)')
plt.ylabel('Leads')
plt.show()

# 4. Line Graph - Leads over Time
df = df.sort_values(by='Date')
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Leads'], marker='o', linestyle='-', color='purple')
plt.title('Leads Over Time')
plt.xlabel('Date')
plt.ylabel('Leads')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 5. Heatmap - Correlation Matrix
numerical_cols = ['Leads', 'Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# ✅ All visualizations added!+


# In[31]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df["Leads"], model="additive", period=7)
decomposition.plot()
plt.show()
df["Week"] = df["Date"].dt.isocalendar().week
pivot_table = df.pivot_table(values="Leads", index="Week", columns="Day", aggfunc="mean")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Heatmap of Leads Across Weeks and Days")
plt.xlabel("Day of the Week")
plt.ylabel("Week Number")
plt.show()
df["Rolling Leads"] = df["Leads"].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Leads"], label="Original Leads", alpha=0.5)
plt.plot(df["Date"], df["Rolling Leads"], label="7-Day Rolling Mean", color="red")
plt.title("Rolling Mean of Leads (7-Day Window)")
plt.xlabel("Date")
plt.ylabel("Leads")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(10, 5))
autocorrelation_plot(df["Leads"])
plt.title("Autocorrelation Plot for Leads")
plt.show()
plt.figure(figsize=(8, 5))
sns.violinplot(x=df["Day"], y=df["Leads"], inner="quartile", palette="muted")
plt.title("Leads Distribution Across Different Days")
plt.xlabel("Day of the Week")
plt.ylabel("Leads")
plt.show()
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=df["Time spent on LG (mins)"],
    y=df["Leads"],
    z=df["Avg Time Per Lead (mins)"],
    mode="markers",
    marker=dict(size=5, color=df["Leads"], colorscale="Viridis")
)])

fig.update_layout(title="3D Scatter Plot (Leads vs Time Spent)",
                  scene=dict(xaxis_title="Time Spent on LG (mins)",
                             yaxis_title="Leads",
                             zaxis_title="Avg Time Per Lead (mins)"))
fig.show()


# In[32]:


import pandas as pd
import numpy as np

# 1. Lead Generation Efficiency
# Calculate the lead generation efficiency for each associate

lead_efficiency = df['Leads'] / df['Time spent on LG (mins)']
df['Lead Generation Efficiency'] = lead_efficiency
highest_efficiency = df.loc[df['Lead Generation Efficiency'].idxmax()]
print("Highest Lead Generation Efficiency:", highest_efficiency)

# 2. Daily Performance Variability
std_dev = df.groupby('Date')['Leads'].std()
highest_variability = std_dev.idxmax(), std_dev.max()
print("Highest Variability:", highest_variability)

# 3. Time Management Analysis
correlation = df['Avg Time Per Lead (mins)'].corr(df['Leads'])
print("Correlation between time per lead and total leads:", correlation)

# 4. Compare Average Leads on Attended vs Missed Daily Reviews
attended_avg = df.loc[df['Daily Team Review'] == 1, 'Leads'].mean()
missed_avg = df.loc[df['Daily Team Review'] == 0, 'Leads'].mean()
performance_diff = ((attended_avg - missed_avg) / missed_avg) * 100
print("Performance Difference (%):", performance_diff)

# 5. Incomplete Leads Reduction Over Time
from sklearn.linear_model import LinearRegression
x = np.arange(len(df)).reshape(-1, 1)
y = df['No. of Incomplete Leads']
model = LinearRegression().fit(x, y)
trend = model.coef_[0]
print("Trend in Incomplete Leads:", trend)

# 6. Performance Consistency
cv = df.groupby('Date')['Leads'].std() / df.groupby('Date')['Leads'].mean()
most_consistent = cv.idxmin(), cv.min()
print("Most Consistent Performance:", most_consistent)

# 7. High-Performance Days
threshold = df['Leads'].quantile(0.90)
high_perf_days = df.loc[df['Leads'] >= threshold]
avg_time_high_perf = high_perf_days['Time spent on LG (mins)'].mean()
print("Average Time on High-Performance Days:", avg_time_high_perf)

# 8. Impact of Longer Lead Generation Time
from scipy.stats import pearsonr
thresholds = np.percentile(df['Time spent on LG (mins)'], [25, 50, 75])
for threshold in thresholds:
    above_threshold = df.loc[df['Time spent on LG (mins)'] > threshold, 'Leads']
    correlation = pearsonr(df.loc[df['Time spent on LG (mins)'] > threshold, 'Time spent on LG (mins)'], above_threshold)[0]
    print(f"Threshold {threshold} correlation:", correlation)

# 9. Comparative Day Analysis
weekday_avg = df.loc[df['Date'].dt.weekday < 5, 'Leads'].mean()
weekend_avg = df.loc[df['Date'].dt.weekday >= 5, 'Leads'].mean()
print("Weekday vs Weekend Performance:", weekday_avg, weekend_avg)

# 10. Predictive Analysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X = df[['Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads', 'Daily Team Review']]
y = df['Leads']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("Model R2 Score:", accuracy)


# In[33]:


X = df[['Time spent on LG (mins)']]
y = df['Leads']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression().fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Model accuracy using R² score
accuracy = r2_score(y_test, y_pred)
print("Model R2 Score:", accuracy)


# # ARYA

# In[34]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'Data Assignment - Arya.csv'
df = pd.read_csv(file_path)

# Fix Date format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# Fill missing values
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
df['Leads'] = imputer_mean.fit_transform(df[['Leads']])
df['Time spent on LG (mins)'] = imputer_mean.fit_transform(df[['Time spent on LG (mins)']])
df['Avg Time Per Lead (mins)'] = imputer_median.fit_transform(df[['Avg Time Per Lead (mins)']])
df['Daily Team Review'].fillna(df['Daily Team Review'].mode()[0], inplace=True)
df['No. of Incomplete Leads'] = imputer_median.fit_transform(df[['No. of Incomplete Leads']])

# Encode categorical column
df['Daily Team Review'] = df['Daily Team Review'].map({'Attended': 1, 'Missed': 0})

# Winsorize to handle outliers (EXCLUDING target variable)
df['Time spent on LG (mins)'] = winsorize(df['Time spent on LG (mins)'], limits=[0.05, 0.05])
df['Avg Time Per Lead (mins)'] = winsorize(df['Avg Time Per Lead (mins)'], limits=[0.05, 0.05])
df['No. of Incomplete Leads'] = winsorize(df['No. of Incomplete Leads'], limits=[0.05, 0.05])

# Save cleaned file
df.to_csv('Data_Assignment_Arya_CLEANED.csv', index=False)

# ==== MODEL TRAINING ====
# Feature selection
X = df[['Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads', 'Daily Team Review']]
y = df['Leads']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_data = pd.DataFrame({
    'Time spent on LG (mins)': np.mean(df['Time spent on LG (mins)']) + np.random.normal(0, 5, 30),
    'Avg Time Per Lead (mins)': np.mean(df['Avg Time Per Lead (mins)']) + np.random.normal(0, 1, 30),
    'No. of Incomplete Leads': np.mean(df['No. of Incomplete Leads']) + np.random.normal(0, 2, 30),
    'Daily Team Review': [1 if i % 2 == 0 else 0 for i in range(30)]
})

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

# Combine predictions with dates
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Leads': future_predictions
})

print(predicted_df)

# Model accuracy
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.4f}')

# ==== VISUALIZATIONS ====
# 1. Histogram - Distribution of Leads
plt.figure(figsize=(8, 5))
sns.histplot(df['Leads'], bins=20, kde=True, color='blue')
plt.title('Distribution of Leads')
plt.xlabel('Leads')
plt.ylabel('Frequency')
plt.show()

# 2. Box Plot - Detecting Outliers in Leads
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Leads'], color='red')
plt.title('Box Plot of Leads')
plt.ylabel('Leads')
plt.show()

# 3. Scatter Plot - Relationship between Time Spent & Leads
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Time spent on LG (mins)'], y=df['Leads'], color='green')
plt.title('Scatter Plot: Time Spent vs. Leads')
plt.xlabel('Time Spent on LG (mins)')
plt.ylabel('Leads')
plt.show()

# 4. Line Graph - Leads over Time
df = df.sort_values(by='Date')
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Leads'], marker='o', linestyle='-', color='purple')
plt.title('Leads Over Time')
plt.xlabel('Date')
plt.ylabel('Leads')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 5. Heatmap - Correlation Matrix
numerical_cols = ['Leads', 'Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 6. Violin Plot - Leads Distribution
plt.figure(figsize=(8, 5))
sns.violinplot(y=df['Leads'], color='skyblue', inner='quartile')
plt.title('Violin Plot of Leads')
plt.ylabel('Leads')
plt.show()

# 7. Pair Plot - Numerical Columns
sns.pairplot(df[numerical_cols])
plt.suptitle('Pair Plot of Key Variables', y=1.02)
plt.show()

# 8. KDE Plot - Density Distribution of Leads
plt.figure(figsize=(8, 5))
sns.kdeplot(df['Leads'], shade=True, color='orange')
plt.title('KDE Plot of Leads')
plt.xlabel('Leads')
plt.ylabel('Density')
plt.show()

# 9. Bar Plot - Average Leads per Day of the Week
plt.figure(figsize=(8, 5))
sns.barplot(x=df['Date'].dt.day_name(), y=df['Leads'], estimator=np.mean, palette='viridis')
plt.title('Average Leads by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Leads')
plt.xticks(rotation=45)
plt.show()

# 10. Rolling Mean of Leads
df['Rolling Leads'] = df['Leads'].rolling(window=7, min_periods=1).mean()
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Leads'], label='Original Leads', alpha=0.5)
plt.plot(df['Date'], df['Rolling Leads'], label='7-Day Rolling Mean', color='red')
plt.title('Rolling Mean of Leads (7-Day Window)')
plt.xlabel('Date')
plt.ylabel('Leads')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# ✅ All 10 visualizations added!


# In[35]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df["Leads"], model="additive", period=7)
decomposition.plot()
plt.show()
df["Week"] = df["Date"].dt.isocalendar().week
pivot_table = df.pivot_table(values="Leads", index="Week", columns="Day", aggfunc="mean")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Heatmap of Leads Across Weeks and Days")
plt.xlabel("Day of the Week")
plt.ylabel("Week Number")
plt.show()
df["Rolling Leads"] = df["Leads"].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Leads"], label="Original Leads", alpha=0.5)
plt.plot(df["Date"], df["Rolling Leads"], label="7-Day Rolling Mean", color="red")
plt.title("Rolling Mean of Leads (7-Day Window)")
plt.xlabel("Date")
plt.ylabel("Leads")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(10, 5))
autocorrelation_plot(df["Leads"])
plt.title("Autocorrelation Plot for Leads")
plt.show()
plt.figure(figsize=(8, 5))
sns.violinplot(x=df["Day"], y=df["Leads"], inner="quartile", palette="muted")
plt.title("Leads Distribution Across Different Days")
plt.xlabel("Day of the Week")
plt.ylabel("Leads")
plt.show()
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter3d(
    x=df["Time spent on LG (mins)"],
    y=df["Leads"],
    z=df["Avg Time Per Lead (mins)"],
    mode="markers",
    marker=dict(size=5, color=df["Leads"], colorscale="Viridis")
)])

fig.update_layout(title="3D Scatter Plot (Leads vs Time Spent)",
                  scene=dict(xaxis_title="Time Spent on LG (mins)",
                             yaxis_title="Leads",
                             zaxis_title="Avg Time Per Lead (mins)"))
fig.show()


# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = 'Data Assignment - Arya.csv'
df = pd.read_csv(file_path)

# Fix Date format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# Fill missing values
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
df['Leads'] = imputer_mean.fit_transform(df[['Leads']])
df['Time spent on LG (mins)'] = imputer_mean.fit_transform(df[['Time spent on LG (mins)']])
df['Avg Time Per Lead (mins)'] = imputer_median.fit_transform(df[['Avg Time Per Lead (mins)']])
df['Daily Team Review'].fillna(df['Daily Team Review'].mode()[0], inplace=True)
df['No. of Incomplete Leads'] = imputer_median.fit_transform(df[['No. of Incomplete Leads']])

# Encode categorical column
df['Daily Team Review'] = df['Daily Team Review'].map({'Attended': 1, 'Missed': 0})

# Winsorize to handle outliers (EXCLUDING target variable)
df['Time spent on LG (mins)'] = winsorize(df['Time spent on LG (mins)'], limits=[0.05, 0.05])
df['Avg Time Per Lead (mins)'] = winsorize(df['Avg Time Per Lead (mins)'], limits=[0.05, 0.05])
df['No. of Incomplete Leads'] = winsorize(df['No. of Incomplete Leads'], limits=[0.05, 0.05])

# Save cleaned file
df.to_csv('Data_Assignment_Arya_CLEANED.csv', index=False)

# ==== MODEL TRAINING ====
# Feature selection
X = df[['Time spent on LG (mins)', 'Avg Time Per Lead (mins)', 'No. of Incomplete Leads', 'Daily Team Review']]
y = df['Leads']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train linear regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_data = pd.DataFrame({
    'Time spent on LG (mins)': np.mean(df['Time spent on LG (mins)']) + np.random.normal(0, 5, 30),
    'Avg Time Per Lead (mins)': np.mean(df['Avg Time Per Lead (mins)']) + np.random.normal(0, 1, 30),
    'No. of Incomplete Leads': np.mean(df['No. of Incomplete Leads']) + np.random.normal(0, 2, 30),
    'Daily Team Review': [1 if i % 2 == 0 else 0 for i in range(30)]
})

future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

# Combine predictions with dates
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Leads': future_predictions
})

print(predicted_df)

# Model accuracy
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
print(f'R² Score: {r2:.4f}')

# ==== ANALYSIS ====
# 1. Lead Generation Efficiency
df['Lead Generation Efficiency'] = df['Leads'] / df['Time spent on LG (mins)']
most_efficient = df.loc[df['Lead Generation Efficiency'].idxmax(), 'Lead Generation Efficiency']
print(f'Highest Efficiency Value: {most_efficient:.2f}')

# 2. Daily Performance Variability
variability = df['Leads'].std()
print(f'Highest Variability: {variability:.2f}')

# 3. Time Management Analysis
correlation = df['Avg Time Per Lead (mins)'].corr(df['Leads'])
print(f'Time vs Leads Correlation: {correlation:.2f}')

# 4. Performance with/without Daily Review
avg_with_review = df.loc[df['Daily Team Review'] == 1, 'Leads'].mean()
avg_without_review = df.loc[df['Daily Team Review'] == 0, 'Leads'].mean()
percentage_difference = ((avg_with_review - avg_without_review) / avg_without_review) * 100
print(f'Percentage Difference in Leads: {percentage_difference:.2f}%')

# 5. Incomplete Leads Reduction Over Time
incomplete_trend = LinearRegression()
incomplete_trend.fit(df.index.values.reshape(-1, 1), df['No. of Incomplete Leads'])
trend_slope = incomplete_trend.coef_[0]
print(f'Incomplete Leads Trend: {trend_slope:.2f}')

# 6. Performance Consistency
cv = df['Leads'].std() / df['Leads'].mean()
print(f'Coefficient of Variation (Consistency): {cv:.2f}')

# 7. High-Performance Days
top_10_percent = df['Leads'].quantile(0.9)
high_perf_days = df[df['Leads'] >= top_10_percent]
avg_time_on_high_perf_days = high_perf_days['Time spent on LG (mins)'].mean()
print(f'Avg Time on High-Performance Days: {avg_time_on_high_perf_days:.2f}')

# 8. Impact of Longer Lead Time
threshold = df['Time spent on LG (mins)'].quantile(0.75)
avg_leads_above_threshold = df.loc[df['Time spent on LG (mins)'] > threshold, 'Leads'].mean()
print(f'Average Leads Above Time Threshold: {avg_leads_above_threshold:.2f}')

# 9. Weekday vs Weekend Performance
df['Day Type'] = df['Date'].dt.dayofweek
weekday_avg = df.loc[df['Day Type'] < 5, 'Leads'].mean()
weekend_avg = df.loc[df['Day Type'] >= 5, 'Leads'].mean()
print(f'Weekday Avg: {weekday_avg:.2f}, Weekend Avg: {weekend_avg:.2f}')

# 10. Predictive Analysis
predicted_leads = model.predict(X_scaled)
accuracy = r2_score(y, predicted_leads)
print(f'Predictive Model R²: {accuracy:.4f}')

# ✅ All 10 analytical questions completed!


# In[37]:


# === Predictive Analysis ===
# Only use 'Time spent on LG (mins)' as the feature
X = df[['Time spent on LG (mins)']]
y = df['Leads']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression().fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Model accuracy using R² score
accuracy = r2_score(y_test, y_pred)
print("Model R2 Score:", accuracy)


# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
files = ["Data_Assignment_Arya_CLEANED.csv", "Data_Assignment_Raj_CLEANED.csv", "Data_Assignment_Ali_CLEANED.csv"]
data = []

for file in files:
    df = pd.read_csv(file)
    df["Associate"] = file.split("_")[2].split(".")[0]  # Extract associate name
    data.append(df)

# Combine all data
df = pd.concat(data, ignore_index=True)

# Standardize column names
df = df.rename(columns={
    "Time spent on LG (mins)": "TimeSpent",
    "Avg Time Per Lead (mins)": "AvgTimePerLead",
    "No. of Incomplete Leads": "IncompleteLeads",
    "Daily Team Review": "TeamReview",
    "Day": "Weekday",
    "Date": "Date"
})

# 🔹 Step 1: Convert Date Correctly
df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")

# 🔹 Step 2: Debugging - Check the date column
print(df["Date"].dtype)  # Should print 'datetime64[ns]'
print(df["Date"].head())  # Should print actual date values
print(f"Missing dates: {df['Date'].isna().sum()}")  # Count missing dates

# 🔹 Step 3: Remove invalid dates
df = df.dropna(subset=["Date"])

# 🔹 Step 4: Convert TeamReview to binary (1=Attended, 0=Missed)
df["TeamReview"] = df["TeamReview"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

# Convert numeric columns
num_cols = ["Leads", "TimeSpent", "AvgTimePerLead", "IncompleteLeads"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# Handle missing values
df.fillna(0, inplace=True)

# 🔹 Step 5: Extract month safely
if pd.api.types.is_datetime64_any_dtype(df["Date"]):
    df["Month"] = df["Date"].dt.month
else:
    print("❌ Date column is not in datetime format!")

# ✅ Plot All Graphs

# 1️⃣ **Performance Trend with Attendance**
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="Date", y="Leads", hue="TeamReview", style="Associate", markers=True)
plt.title("Daily Leads vs. Team Review Attendance")
plt.xlabel("Date")
plt.ylabel("Leads Generated")
plt.legend(title="Team Review (1=Attended, 0=Missed)")
plt.xticks(rotation=45)


# 3️⃣ **Monthly Performance Comparison**
plt.figure(figsize=(10,5))
sns.barplot(data=df, x="Month", y="Leads", hue="Associate")
plt.title("Monthly Leads Generated by Associate")
plt.xlabel("Month")
plt.ylabel("Total Leads")

# 4️⃣ **Daily Incomplete Leads Trend**
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="Date", y="IncompleteLeads", hue="Associate", markers=True)
plt.title("Daily Trend of Incomplete Leads")
plt.xlabel("Date")
plt.ylabel("Incomplete Leads")
plt.xticks(rotation=45)

# 5️⃣ **Time Distribution Analysis**
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Associate", y="TimeSpent")
plt.title("Time Spent on Lead Generation (Distribution)")
plt.xlabel("Associate")
plt.ylabel("Time Spent (mins)")

# 🔹 Show all plots at the end
plt.show()


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Ensure "TimeSpent" is numeric
df["TimeSpent"] = pd.to_numeric(df["TimeSpent"], errors="coerce")

# 🔹 Drop missing values for "Leads" and "TimeSpent"
df = df.dropna(subset=["Leads", "TimeSpent"])

# 🔹 Adjust bins to better match the data
bins = [0, 60, 120, 180, 240, np.inf]  # Meaningful time ranges
labels = ["0-1 hr", "1-2 hrs", "2-3 hrs", "3-4 hrs", "4+ hrs"]
df["TimeSpentBins"] = pd.cut(df["TimeSpent"], bins=bins, labels=labels)

# 🔹 Print bin counts for debugging
print(df["TimeSpentBins"].value_counts())  # Check new distribution

# 🔹 Create Pivot Table
pivot_data = df.pivot_table(index="Associate", columns="TimeSpentBins", values="Leads", aggfunc="sum")

# 🔹 Fill NaN values with 0 (important for heatmap)
pivot_data = pivot_data.fillna(0)

# 🔹 Log transform if values are highly skewed
pivot_data = np.log1p(pivot_data)  # log(1 + x) to handle zeros

# 🔹 Plot Enhanced Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_data, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.5, cbar=True)

plt.title("Time Spent vs. Leads Generated (Binned)")
plt.xlabel("Time Spent on Lead Generation")
plt.ylabel("Associate")
plt.xticks(rotation=30)  # Rotate for better visibility
plt.yticks(rotation=0)
plt.show()


# In[17]:


import seaborn as sns

# 🔹 Create Efficiency Column
df["Efficiency"] = df["Leads"] / (df["TimeSpent"] + 1)  # Avoid division by zero

# 🔹 KDE Plot (Efficiency Distribution)
plt.figure(figsize=(8, 5))
sns.kdeplot(df, x="Efficiency", hue="Associate", fill=True, alpha=0.4)

plt.title("Efficiency Distribution Across Associates")
plt.xlabel("Efficiency (Leads per Minute)")
plt.ylabel("Density")
plt.show()


# In[20]:


import plotly.graph_objects as go

# Define funnel stages
stages = ["Leads Contacted", "Leads Interested", "Leads Converted"]

# Actual data from CSV files
arya_counts = [705, 705, 5]
raj_counts = [665, 659, 4]
ali_counts = [726, 726, 5]

fig = go.Figure()

# Add each associate's data to the funnel plot
fig.add_trace(go.Funnel(
    name="Arya",
    y=stages,
    x=arya_counts,
    textinfo="value+percent initial"
))
fig.add_trace(go.Funnel(
    name="Raj",
    y=stages,
    x=raj_counts,
    textinfo="value+percent initial"
))
fig.add_trace(go.Funnel(
    name="Ali",
    y=stages,
    x=ali_counts,
    textinfo="value+percent initial"
))

# Customize layout
fig.update_layout(
    title="Lead Conversion Funnel for Associates",
    funnelmode="group"  # Grouped view
)

# Show the plot
fig.show()

