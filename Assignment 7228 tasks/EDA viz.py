
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set the style for seaborn plots
sns.set_theme(style="whitegrid")

# File path for the dataset
file_path = r'C:\Users\James Corbett\OneDrive\WHR 15.xlsx'

# Load the dataset
data = pd.read_excel(file_path)

# Handling missing values by imputing with mean
data.fillna(data.mean(), inplace=True)

# List of continuous variables for visualization
continuous_columns = ['Life Ladder', 'Log GDP per capita', 'Social support',
                      'Healthy life expectancy at birth', 'Freedom to make life choices',
                      'Generosity', 'Perceptions of corruption', 'Positive affect',
                      'Negative affect']

# Initialize subplots for histograms
fig1, axes1 = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
axes1 = axes
1.flatten() # Flatten for easy iteration

Initialize subplots for boxplots
fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
axes2 = axes2.flatten() # Flatten for easy iteration

Generate histograms and boxplots for each variable
for i, column in enumerate(continuous_columns):
sns.histplot(data[column], kde=True, ax=axes1[i], color='skyblue').set_title(f'Distribution of {column}')
axes1[i].set_ylabel('Frequency')
axes1[i].set_xlabel(column)
sns.boxplot(x=data[column], ax=axes2[i], color='orange').set_title(f'Boxplot of {column}')
axes2[i].set_xlabel(column)

Adjust layout for better readability and display histograms
plt.tight_layout()
plt.show(fig1)

Adjust layout for better readability and display boxplots
plt.tight_layout()
plt.show(fig2)

Generate a heatmap for the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='viridis')
plt.title('Correlation Matrix of Variables')
plt.show()

Prepare the data for modeling
X = data.drop(['Country name', 'year', 'Life Ladder'], axis=1)
y = data['Life Ladder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

Print the performance metrics
print(f'MSE: {mse}')
print(f'R^2: {r2}')

Save the figures if needed
fig1.savefig('path_to_save_histograms.png')
fig2.savefig('path_to_save_boxplots.png')
