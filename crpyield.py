import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)


data = pd.read_csv('Crop_recommendation.csv')


st.sidebar.title("Crop Yield Prediction")


st.sidebar.markdown("Explore and analyze crop yield data")





def display_crops_by_condition(condition, threshold, comparison_type):
    if comparison_type == "Greater than":
        st.write(f"Crops which require a very high ratio of {condition} content in soil:")
        selected_crops = data[data[condition] > threshold]['label'].unique()
    elif comparison_type == "Less than":
        st.write(f"Crops which require a very low ratio of {condition} content in soil:")
        selected_crops = data[data[condition] < threshold]['label'].unique()

    st.table(pd.DataFrame({"Crops": selected_crops}))


def compare_crops(condition, comparison_type):
    if condition != 'Select a condition':
        if comparison_type == "Greater than average":
            st.write(f"Crops which require greater than average {condition}:")
            selected_crops = data[data[condition] > data[condition].mean()]['label'].unique()
        elif comparison_type == "Less than average":
            st.write(f"Crops which require lesser than average {condition}:")
            selected_crops = data[data[condition] < data[condition].mean()]['label'].unique()

        
        comparison_data = pd.DataFrame({
            'Crops': selected_crops
        })

        
        st.table(comparison_data)



condition_options = ['Select a condition', 'N', 'P', 'K', 'rainfall', 'temperature', 'humidity', 'ph']
condition = st.sidebar.selectbox("Select Condition", condition_options, index=0)


if condition != 'Select a condition':
    min_threshold = int(data[condition].min())
    max_threshold = int(data[condition].max())
    
    threshold = st.sidebar.slider("Select Threshold Value", min_value=min_threshold, max_value=max_threshold)
else:
    threshold = None


comparison_type = st.sidebar.radio("Select Comparison Type", ['Greater than', 'Less than'])


if condition != 'Select a condition' and threshold and comparison_type:
    result_placeholder = st.empty()
    result_placeholder.text("Loading... Please wait.")
    display_crops_by_condition(condition, threshold, comparison_type)
    result_placeholder.empty()


def compare_all_crops():
    condition_options = ['Select a condition', 'N', 'P', 'K', 'temperature', 'humidity', 'rainfall']
    condition = st.selectbox("Select Condition", condition_options, index=0)
    comparison_type = st.selectbox("Select Comparison Type", ['Greater than average', 'Less than average'])
    st.write(f"Comparison for {condition}:")
    compare_crops(condition, comparison_type)


st.header("Crop Yield Analysis")
compare_all_crops()




def summary(crops=None):
    if crops is None:
        crops = list(data['label'].value_counts().index)

    st.write(f"Summary for {crops} - Minimum, Average, and Maximum")
    st.write("--------------------------------------------------")

    
    x = data[data['label'].isin(crops)]

    if x.empty:
        st.write("No data available for the selected crops.")
    else:
        z = x.drop(['label'], axis=1)

        
        summary_data = pd.DataFrame({
            'Crop': crops
        })

        for i in z:
            summary_data[f'Minimum {i}'] = x[i].min()
            summary_data[f'Average {i}'] = x[i].mean()
            summary_data[f'Maximum {i}'] = x[i].max()

        
        st.table(summary_data)

if 'selected_crops' not in st.session_state:
    st.session_state.selected_crops = list(data['label'].value_counts().index)

st.subheader("MIN,MAX,AVG FOR SELECTED CROP")
selected_crops_placeholder = st.empty()
selected_crops = st.multiselect("Select Crops", data['label'].value_counts().index, st.session_state.selected_crops)


if st.button("Select Crops"):
    st.session_state.selected_crops = selected_crops


selected_crops_placeholder.write(f"Selected Crops: {st.session_state.selected_crops}")
summary(st.session_state.selected_crops)

def compare():
    conditions = st.radio("Select Condition", ['N', 'P', 'K', 'temperature', 'humidity', 'rainfall'])
    st.write(f"Average value for {conditions}:")
    grouped = data.groupby('label')[conditions].mean()

    
    comparison_data = pd.DataFrame({
        'Crop': grouped.index,
        'Average Value': grouped.values
    })

   
    st.table(comparison_data)

st.subheader("AVERAGE CONDITION")
compare()
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']



st.header("Elbow Method for K-Means")


y = data['label']
x = data.drop(['label'], axis=1)


wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)


elbow_data = pd.DataFrame({'Number of Clusters (K)': range(1, 11), 'WCSS': wcss})
st.line_chart(elbow_data.set_index('Number of Clusters (K)'))


st.sidebar.subheader("Model Selection")
selected_model = st.sidebar.selectbox("Select Model", ['Logistic Regression', 'Random Forest'])


if selected_model == 'Logistic Regression':
    model = LogisticRegression()
else:  
    model = RandomForestClassifier()

model.fit(x, y)

st.sidebar.subheader("Model Interpretation")
with st.sidebar.expander("Feature Importance / Coefficients"):
    if selected_model == 'Random Forest':
        feature_importance = pd.Series(model.feature_importances_, index=x.columns)
        st.write(feature_importance)
    elif selected_model == 'Logistic Regression':
        feature_coef = pd.Series(model.coef_[0], index=x.columns)
        st.write(feature_coef)


st.header("Model Evaluation")


y_predict = model.predict(x)

conf_matrix = confusion_matrix(y, y_predict)


evaluation_metric = st.sidebar.selectbox("Select Evaluation Metric", ['Confusion Matrix', 'Classification Report'])

if evaluation_metric == 'Confusion Matrix':
    
    st.subheader("Confusion Matrix")
    
    fig_confusion_matrix, ax_confusion_matrix = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_predict), annot=True, cmap="Reds", ax=ax_confusion_matrix)
    ax_confusion_matrix.set_title("Confusion Matrix")
    ax_confusion_matrix.set_xlabel("Predicted Labels")
    ax_confusion_matrix.set_ylabel("True Labels")
    plt.tight_layout()

    
    st.pyplot(fig_confusion_matrix)

elif evaluation_metric == 'Classification Report':

    st.subheader("Classification Report")
    classification_report_str = classification_report(y, y_predict)
    classification_report_dict = classification_report(y, y_predict, output_dict=True)


    classification_report_df = pd.DataFrame(classification_report_dict).transpose()


    st.table(classification_report_df)


st.header("Crop Prediction")
st.write("Enter the climatic conditions below and click the 'Predict' button to get the suggested crop.")


n = st.slider("Nitrogen (N)", min_value=0, max_value=100, value=50)
p = st.slider("Phosphorus (P)", min_value=0, max_value=100, value=50)
k = st.slider("Potassium (K)", min_value=0, max_value=100, value=50)
temperature = st.slider("Temperature", min_value=0, max_value=100, value=25)
humidity = st.slider("Humidity", min_value=0, max_value=100, value=60)
ph = st.slider("pH", min_value=0, max_value=14, value=7)
rainfall = st.slider("Rainfall", min_value=0, max_value=400, value=200)


if st.button("Predict"):
    prediction = model.predict([[n, p, k, temperature, humidity, ph, rainfall]])
    st.write("The suggested crop for the given climatic condition is:", prediction[0])
