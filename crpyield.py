import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

# Disable Matplotlib PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Sidebar Title
st.sidebar.title("Crop Yield Prediction")

# Sidebar Description
st.sidebar.markdown("Explore and analyze crop yield data")

st.write("First 5 rows of the dataset:")
st.write(data.head())

# Function to compare crops which require greater or lesser than average for a condition
def compare_crops(condition, comparison_type):
    if comparison_type == "Greater than average":
        st.write(f"Crops which require greater than average {condition}:")
        selected_crops = data[data[condition] > data[condition].mean()]['label'].unique()
    elif comparison_type == "Less than average":
        st.write(f"Crops which require lesser than average {condition}:")
        selected_crops = data[data[condition] < data[condition].mean()]['label'].unique()

    # Create a Pandas DataFrame to hold the comparison results
    comparison_data = pd.DataFrame({
        'Crops': selected_crops
    })

    # Display the comparison as a table
    st.table(comparison_data)

# Function to compare all crops for N, P, K, temperature, humidity, and rainfall
def compare_all_crops():
    condition = st.selectbox("Select Condition", ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'])
    comparison_type = st.selectbox("Select Comparison Type", ['Greater than average', 'Less than average'])
    st.write(f"Comparison for {condition}:")
    compare_crops(condition, comparison_type)

# Show the comparison for all conditions using the dropdown
st.header("Crop Yield Analysis")
compare_all_crops()




# Function to display summary
def summary(crops=None):
    if crops is None:
        crops = list(data['label'].value_counts().index)

    st.write(f"Summary for {crops} - Minimum, Average, and Maximum")
    st.write("--------------------------------------------------")

    # Filter data for selected crops
    x = data[data['label'].isin(crops)]

    if x.empty:
        st.write("No data available for the selected crops.")
    else:
        z = x.drop(['label'], axis=1)

        # Calculate Minimum, Average, and Maximum
        summary_data = pd.DataFrame({
            'Crop': crops
        })

        for i in z:
            summary_data[f'Minimum {i}'] = x[i].min()
            summary_data[f'Average {i}'] = x[i].mean()
            summary_data[f'Maximum {i}'] = x[i].max()

        # Display the summary as a table
        st.table(summary_data)

# Initialize session state with all crops as the default selection
if 'selected_crops' not in st.session_state:
    st.session_state.selected_crops = list(data['label'].value_counts().index)

# Show the selectbox to choose the crops
st.subheader("MIN,MAX,AVG FOR SELECTED CROP")
selected_crops = st.multiselect("Select Crops", data['label'].value_counts().index, st.session_state.selected_crops)

# Update the session state with the selected crops
st.session_state.selected_crops = selected_crops

# Show the summary table for selected crops
summary(selected_crops)




# Function to compare conditions
def compare():
    conditions = st.radio("Select Condition", ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'])
    st.write(f"Average value for {conditions}:")
    grouped = data.groupby('label')[conditions].mean()

    # Create a Pandas DataFrame to hold the comparison results
    comparison_data = pd.DataFrame({
        'Crop': grouped.index,
        'Average Value': grouped.values
    })

    # Display the comparison as a table
    st.table(comparison_data)

# Display the functions in the main content area
st.subheader("AVERAGE CONDITION")
compare()

# Display the bar plots for each feature in the main content area
st.header("Crop Feature Distribution")

# Define the features
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Create subplots for each feature
num_features = len(features)
num_cols = 2
num_rows = (num_features + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows*4))

for i, feature in enumerate(features):
    row = i // num_cols
    col = i % num_cols
    sns.histplot(data[feature], color='blue', ax=axes[row, col])

    # Set x-label and grid for the subplot
    axes[row, col].set_xlabel(f'Ratio of {feature}', fontsize=12)
    axes[row, col].grid()

plt.tight_layout()

# Display the figure using st.pyplot() with the Matplotlib figure as an argument
st.pyplot(fig)

# Display the elbow plot for K-Means clustering in the main content area
st.header("Elbow Method for K-Means")

# Determine Optimum number of clusters by elbow method
x = data.drop(['label'], axis=1).values
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

# Plot the results
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(range(1, 11), wcss)
ax_elbow.set_title('Elbow Method', fontsize=15)
ax_elbow.set_xlabel('No. of clusters')
ax_elbow.set_ylabel('WCSS')
plt.tight_layout()

# Display the figure using st.pyplot() with the Matplotlib figure as an argument
st.pyplot(fig_elbow)

# Prepare the data for model training
y = data['label']
x = data.drop(['label'], axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Interactive Widget for Model Selection
st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.selectbox("Select Model", ['Logistic Regression', 'Random Forest'])
# Train a Logistic Regression model or Random Forest model based on the selected option
# Train a Model based on the selected option
if selected_model == 'Logistic Regression':
    model = LogisticRegression()
else:  # Default to Random Forest
    model = RandomForestClassifier()
# Fit the Model
model.fit(x_train, y_train)

# Feature Importance or Coefficients
st.sidebar.subheader("Model Interpretation")
with st.sidebar.expander("Feature Importance / Coefficients"):
    if selected_model == 'Random Forest':
        feature_importance = pd.Series(model.feature_importances_, index=x.columns)
        st.write(feature_importance)
    elif selected_model == 'Logistic Regression':
        feature_coef = pd.Series(model.coef_[0], index=x.columns)
        st.write(feature_coef)


# Display the confusion matrix and classification report in the main content area
st.header("Model Evaluation")

# Make predictions on test data
y_predict = model.predict(x_test)

# Evaluation metrics
conf_matrix = confusion_matrix(y_test, y_predict)

# Evaluation metrics
evaluation_metric = st.sidebar.selectbox("Select Evaluation Metric", ['Confusion Matrix', 'Classification Report'])

if evaluation_metric == 'Confusion Matrix':
    # Display the confusion matrix in the sidebar
    st.subheader("Confusion Matrix")
    # Plot the confusion matrix
    fig_confusion_matrix, ax_confusion_matrix = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap="Reds", ax=ax_confusion_matrix)
    ax_confusion_matrix.set_title("Confusion Matrix")
    ax_confusion_matrix.set_xlabel("Predicted Labels")
    ax_confusion_matrix.set_ylabel("True Labels")
    plt.tight_layout()

    # Display the figure using st.pyplot() with the Matplotlib figure as an argument
    st.pyplot(fig_confusion_matrix)

elif evaluation_metric == 'Classification Report':
    # Display the classification report in the sidebar
    st.subheader("Classification Report")
    classification_report_str = classification_report(y_test, y_predict)
    classification_report_dict = classification_report(y_test, y_predict, output_dict=True)

    # Convert the classification report dictionary to a Pandas DataFrame
    classification_report_df = pd.DataFrame(classification_report_dict).transpose()
    
    # Display the classification report as a table
    st.table(classification_report_df)



# Prediction for given climatic condition
st.header("Crop Prediction")
st.write("Enter the climatic conditions below and click the 'Predict' button to get the suggested crop.")

# Interactive Widgets for Climatic Conditions
n = st.slider("Nitrogen (N)", min_value=0, max_value=100, value=50)
p = st.slider("Phosphorus (P)", min_value=0, max_value=100, value=50)
k = st.slider("Potassium (K)", min_value=0, max_value=100, value=50)
temperature = st.slider("Temperature", min_value=0, max_value=100, value=25)
humidity = st.slider("Humidity", min_value=0, max_value=100, value=60)
ph = st.slider("pH", min_value=0, max_value=14, value=7)
rainfall = st.slider("Rainfall", min_value=0, max_value=400, value=200)

# Button to predict the crop based on the input climatic conditions
if st.button("Predict"):
    prediction = model.predict([[n, p, k, temperature, humidity, ph, rainfall]])
    st.write("The suggested crop for the given climatic condition is:", prediction[0])
