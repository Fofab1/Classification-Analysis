import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Title
st.title("Classification Analysis")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("data/iris.csv", header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    return data

df = load_data()

# Upload button
uploaded_file = st.file_uploader("Upload your dataset (iris)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    # Display the dataset shape
    st.subheader("Exploratory Data Analysis")
    if st.button("Show Dataset Shape"):
        st.write(f"Shape of the dataset: {data.shape}")

    # Display the first few rows of the dataset
    if st.button("Show First Few Rows"):
        st.write(data.head())

    # Check for missing values
    if st.button("Check for Missing Values"):
        missing_values = data.isnull().sum()
        st.write(missing_values)
        st.write(f"Total missing values: {missing_values.sum()}")

    # EDA: Histograms for each feature
    if st.button("Show Histograms"):
        st.write("Histograms for each feature")
        data.hist(figsize=(10, 8), bins=15)
        st.pyplot(plt)
        plt.clf()

    # EDA: Pair plots for visualizing feature relationships
    if st.button("Show Pair Plot"):
        st.write("Pair Plot")
        sns.pairplot(data, hue="species")
        st.pyplot(plt)
        plt.clf()

    # EDA: Correlation heatmap
    if st.button("Show Correlation Heatmap"):
        st.write("Correlation Heatmap")
        corr = data.drop(columns=['species']).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=.5)
        st.pyplot(plt)
        plt.clf()

    # Model Selection
    st.subheader("Model Selection and Training")
    model_options = ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]
    selected_model = st.selectbox("Select a Model", model_options)

    # Hyperparameter Tuning and Scaling Options
    st.subheader("Hyperparameter Tuning and Scaling")
    if selected_model == "K-Nearest Neighbors":
        n_neighbors = st.slider("Number of Neighbors (K)", 1, 15, 3)
    elif selected_model == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 10, 3)

    scaling_option = st.selectbox("Select a Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])

    # Add a button to split the dataset
    if st.button("Split Dataset"):
        X = data.drop(columns=['species'])
        y = data['species']
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100  # Slider to select the test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        st.write(f"Training data shape: {X_train.shape}")
        st.write(f"Testing data shape: {X_test.shape}")

        # Apply scaling if selected
        if scaling_option == "StandardScaler":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif scaling_option == "MinMaxScaler":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Store the split data in session state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['split_data'] = True  # Store the flag in session state

    # Add a train button to train the selected model
    if st.button("Train Model"):
        if st.session_state.get('split_data', False):  # Check if the dataset has been split
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            if selected_model == "Logistic Regression":
                model = LogisticRegression()
            elif selected_model == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=max_depth)
            elif selected_model == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=n_neighbors)

            # Train the model
            model.fit(X_train, y_train)
            st.write("Model trained successfully!")

            # Make predictions
            y_pred = model.predict(X_test)

            # Display metrics
            st.write("### Model Evaluation")
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))

            # Add ROC curve
            if selected_model in ["Logistic Regression", "K-Nearest Neighbors"]:
                y_prob = model.predict_proba(X_test)[:, 1]  # Get probability estimates for the positive class
                fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y_test.unique()[1])
                roc_auc = auc(fpr, tpr)

                st.subheader("ROC Curve")
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                st.pyplot(plt)
                plt.clf()

            # Prepare data for download
            y_pred_df = pd.DataFrame({"Predicted": y_pred})
            csv_data = y_pred_df.to_csv(index=False)

            # Add a download button for predicted results
            st.download_button(
                label="Download Predicted Results as CSV",
                data=csv_data,
                file_name='predicted_results.csv',
                mime='text/csv'
            )

        else:
            st.error("Please split the dataset first!")
