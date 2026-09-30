import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_ind

from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")
#load and explorer dataset
df = pd.read_csv('weather_forecast_data.csv')
# Verify shapes.shape)
# Display information for the dataset
print("Dataset Information: \n")
train_info = df.info()
display(train_info)
print('\n')
print("Dataset Statistical Summary: \n")
display(df.describe().T)
# Check for missing values in the dataset
print('--- Missing Values in dataset---\n')
df_missing = df.isnull().sum()
print(df_missing)
print('\n')
# Check for duplicate rows in the dataset
df_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows in the dataset: {df_duplicates}")
# Display the number of unique values in each column of the dataset
print("Unique values in dataset:")
df_unique_counts = df.nunique()
print(df_unique_counts)
# Identifying numerical and non-numerical columns in the dataset
numerical_df = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
non_numerical_df = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

print("\nNumerical columns in the dataset:")
print(numerical_df)

print("\nNon-numerical columns in the dataset:")
print(non_numerical_df)
# Display unique values for each categorical column
print("\nUnique values for each categorical column in the dataset:")
for col in non_numerical_df:
    print(f"\nColumn: {col}")
    print(f"Unique Values: {df[col].unique()}")
    # Define a custom color map
colors = ['#0077b6', '#00b4d8', '#90e0ef','#d8b2ff', '#b266ff', '#4500e2' ]  
cmap = LinearSegmentedColormap.from_list("custom_blue_purple", colors, N=256)

# Set the color palette in seaborn
sns.set_palette(sns.color_palette(colors))
# Create subplots for the 'Rain' feature
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
color_selection = [colors[4], colors[1]] 

# Count plot for the Rain feature
rain_counts = df['Rain'].value_counts()
sns.barplot(x=rain_counts.index, y=rain_counts, ax=axes[0], palette=color_selection)  
axes[0].set_title('Distribution of Rain')
axes[0].set_ylabel('Count')

for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points')
sns.despine(left=True, bottom=True) 

# Pie chart for the percentage distribution of the Rain feature
rain_percentage = df['Rain'].value_counts(normalize=True) * 100
axes[1].pie(rain_percentage, labels=rain_percentage.index, autopct='%1.1f%%',
            colors=color_selection)  
axes[1].set_title('Percentage Distribution of Rain')

plt.tight_layout()
plt.show()
# Function to perform univariate analysis for numeric columns
def univariate_analysis(data, columns):
    plt.figure(figsize=(15, 18))  
    
    for i, column in enumerate(columns, 1):
        plt.subplot(3, 2, i)  
        sns.histplot(data[column], kde=True, bins=30, color=colors[i % len(colors)])  # Use cyclic color indexing
        plt.title(f'{column.replace("_", " ")} Distribution with KDE')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

columns_to_analyze = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

univariate_analysis(df, columns_to_analyze)
# Function to perform univariate analysis for numeric columns
def univariate_analysis(data, column, title):
    plt.figure(figsize=(10, 2))
    
    # Use a custom color from the palette
    sns.boxplot(x=data[column], color=colors[2])  # Selected a color from the palette for consistency
    plt.title(f'{title} Boxplot')
    
    plt.tight_layout()
    plt.show()

    print(f'\nSummary Statistics for {title}:\n', data[column].describe())

columns_to_analyze = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']
for column in columns_to_analyze:
    univariate_analysis(df, column, column.replace('_', ' '))
# Loop through each column and create a displot comparing distributions by 'Rain'
for column in columns_to_analyze:
    plt.figure(figsize=(8, 5))
    color_selection = [colors[0], colors[5]] 
    
    sns.displot(
        data=df,
        x=column,
        hue="Rain",
        kind="kde",
        fill=True,
        palette=color_selection,  
        height=4,
        aspect=1.2
    )
    
    plt.title(f'Distribution of {column} by Rain Status')
    plt.xlabel(column.replace('_', ' '))
    plt.ylabel('Density')
    plt.show()
    plt.figure(figsize=(15, 10))

# Loop through each column and create a box plot for each feature grouped by 'Rain'
for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot(3, 2, i)
    
    color_selection = [colors[1], colors[4]]  
    
    sns.boxplot(x='Rain', y=column, data=df, palette=color_selection)
    plt.title(f'Box Plot of {column} by Rain Status')
    plt.xlabel('Rain Status')
    plt.ylabel(column)

plt.tight_layout()
plt.show()
# Create a figure and set its size
plt.figure(figsize=(18, 12))
color_selection = [colors[2], colors[3]]

# Loop through each column and create a count plot for binned data
for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot(3, 2, i)
    # Create bins for each numerical column
    bins = pd.cut(df[column], bins=10, labels=[f'Bin_{j+1}' for j in range(10)])
    # Count plot for each binned data
    sns.countplot(x=bins, hue=df['Rain'], palette=color_selection)
    plt.title(f'Count of {column} Bins by Rain Status')
    plt.xlabel(f'{column} Range')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  

plt.tight_layout()
plt.show()
# List of columns to analyze plus the target 'Rain'
columns_for_pairplot = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure', 'Rain']

color_selection = [colors[1], colors[3]]  

# Create the pair plot
pairplot = sns.pairplot(df[columns_for_pairplot], hue='Rain', palette=color_selection, 
                        markers=["D", "s"],  
                        plot_kws={'alpha': 0.5})  
pairplot.fig.suptitle('Pair Plot of Weather Features by Rain Status', y=1.02)  
plt.show()
# Calculate the correlation matrix
correlation_matrix = df[columns_to_analyze].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=cmap, cbar=True,
            cbar_kws={"shrink": .82},
            linewidths=.5, square=True)

# Title for the heatmap
plt.title('Correlation Matrix of Weather Features', pad=20)
plt.show()
# Label Encoding for 'Rain' to ensure it's in binary form
df['Rain'] = df['Rain'].map({'no rain': 0, 'rain': 1})

# Splitting data based on 'Rain' values (0 = no rain, 1 = rain)
rain_data = df[df['Rain'] == 1]
no_rain_data = df[df['Rain'] == 0]
# Conducting t-tests for each feature
features_to_test = ['Humidity', 'Cloud_Cover', 'Temperature', 'Pressure', 'Wind_Speed']
t_test_results = {}

for feature in features_to_test:
    t_stat, p_value = ttest_ind(rain_data[feature], no_rain_data[feature], equal_var=False)  
    t_test_results[feature] = {'t_statistic': t_stat, 'p_value': p_value}
    
    # Adding dynamic hypothesis testing output
    if p_value < 0.05:
        print(f"For {feature}:")
        print(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
        print("  Result: Reject the Null Hypothesis (significant difference in means)\n")
    else:
        print(f"For {feature}:")
        print(f"  t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
        print("  Result: Fail to reject the Null Hypothesis (no significant difference in means)\n")

# Displaying results as a dictionary
print("T-Test Results:")
display(pd.DataFrame(t_test_results).T)
# Define the color selection
color_selection = [colors[0], colors[3]]  

# Generate distribution plots
for feature in features_to_test:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=rain_data, x=feature, label='Rain', kde=True, color=color_selection[0], alpha=0.6)
    sns.histplot(data=no_rain_data, x=feature, label='No Rain', kde=True, color=color_selection[1], alpha=0.6)
    plt.title(f'Distribution of {feature} for Rain and No Rain')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    correlation_matrix = df.corr()

# Extract correlation with the target variable 'Rain'
target_correlation = correlation_matrix['Rain'].sort_values(ascending=False)

# Display the correlation values with the target variable
styled_correlation = target_correlation.to_frame().style.background_gradient(cmap=cmap)
styled_correlation
X = df.drop('Rain', axis=1)  # Exclude the target variable from anomaly detection
# Initialize the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)  # 5% contamination by default
iso_forest.fit(X)

# Predict anomalies: -1 indicates an anomaly, 1 indicates normal
outliers = iso_forest.predict(X)

# Add the outlier labels to the DataFrame
df['Anomaly'] = outliers

# Display the counts of normal and anomalous points
print("Anomaly Detection Results:")
print(df['Anomaly'].value_counts())
# Identify anomalies in the dataset
anomalies = df[df['Anomaly'] == -1]  

# Display the number of anomalies and the first few rows of detected anomalies
print(f"Number of anomalies detected: {anomalies.shape[0]}")
anomalies.head()
# Remove outliers from the data
df_cleaned = df[df['Anomaly'] == 1].drop(columns=['Anomaly'])
# Split the cleaned data into features and target
X_cleaned = df_cleaned.drop('Rain', axis=1)
y_cleaned = df_cleaned['Rain']
# Split the dataset into the Training set and Test set
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
# Displaying the shapes of the resulting datasets for verification
print("Training features shape:", X_train.shape)
print("Validation features shape:", X_val.shape)
print("Training target shape:", y_train.shape)
print("Validation target shape:", y_val.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
# Train and evaluate a given model on training and validation data.
def train_and_evaluate_model(name, model, X_train, y_train, X_val, y_val):
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    print(f"\n{name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=cmap)
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    
    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Report": pd.DataFrame(report).T,
        "Confusion Matrix": cm
    }
# Perform cross-validation on a given model.
def cross_validate_model(name, model, X, y, cv):
    print(f"Evaluating {name} using cross-validation...")
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(accuracy_score))
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    return {
        "Mean Accuracy": accuracy_scores.mean(),
        "Mean AUC": auc_scores.mean(),
        "Accuracy Scores": accuracy_scores,
        "AUC Scores": auc_scores
    }
# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(verbosity=-1, random_state=42)
}
# Split the cleaned data into features and target 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate models using cross-validation
cross_val_results = {}
for name, model in models.items():
    cross_val_results[name] = cross_validate_model(name, model, X_cleaned, y_cleaned, cv)

# Display cross-validation results
print("\nCross-Validation Results Summary:\n" + "="*40)
results_df = pd.DataFrame({
    name: {"Mean Accuracy": result["Mean Accuracy"], "Mean AUC": result["Mean AUC"]}
    for name, result in cross_val_results.items()
}).T.sort_values(by="Mean AUC", ascending=False)
print(results_df)
print("="*40)
# Select the best model based on mean AUC
best_model_name = results_df['Mean AUC'].idxmax()
print(f"\nThe best model based on Mean AUC from cross-validation is: {best_model_name} with an AUC of {results_df['Mean AUC'].max():.4f}")
best_model = models[best_model_name]
results = train_and_evaluate_model(best_model_name, best_model, X_train, y_train, X_val, y_val)
# Check if the best model has a feature_importances_ attribute
if hasattr(best_model, "feature_importances_"):
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette=colors)
    plt.title(f'Feature Importance for {best_model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
else:
    print(f"The selected model ({best_model_name}) does not support feature importances.")
    # Predicted probabilities for the positive class from the best model
y_prob = best_model.predict_proba(X_val)[:, 1]
 
plt.figure(figsize=(10, 6))
sns.histplot(y_prob, bins=30, kde=True, color=colors[1])  
plt.title(f'Distribution of Predicted Probabilities - {best_model_name}')
plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.show()
# Predicted binary labels
y_pred = best_model.predict(X_val)

plt.figure(figsize=(10, 6))
sns.countplot(x=y_pred, palette=[colors[3], colors[2]])  
plt.title(f'Distribution of Binary Predictions - {best_model_name}')
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Rain', 'Rain'])  
plt.show()