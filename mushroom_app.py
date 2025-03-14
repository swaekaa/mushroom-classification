#[theme]
#primaryColor="#b1846b"
#backgroundColor="#1a1619"
#secondaryBackgroundColor="#5c6296"
#textColor="#eae1e1"
#font="monospace"

# streamlit run mushroom_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, precision_score, recall_score


def main():
    st.title("BINARY CLASSIFICATION WEB APP")
    st.sidebar.title("BINARY CLASSIFICATION WEB APP")
    st.markdown("🍄 Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("🍄 Are your mushrooms edible or poisonous? 🍄")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label_encoder = LabelEncoder()

        for col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df['class']
        x = df.drop(columns=['class'])
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test

    def plot_metrics(metrics_list, y_test, y_pred, y_proba):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            sn.heatmap(cm, annot=True, fmt="d")
            st.pyplot(plt.gcf())

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label="ROC Curve")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            st.pyplot(plt.gcf())

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            plt.figure()
            plt.plot(recall, precision, label="Precision-Recall Curve")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            st.pyplot(plt.gcf())

    data = load_data()
    X_train, X_test, y_train, y_test = split(data)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(data)

    # Model Selection
    st.sidebar.subheader("Model Selection")
    model = st.sidebar.selectbox("Select Model", ["Support Vector Machine", "Logistic Regression", "Random Forest"])

    # Metric Selection (common for all models)
    metrics_list = st.sidebar.multiselect("Select metrics to plot", 
                                          ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])

    classifier = None  # Placeholder for classifier
    y_pred, y_proba = None, None  # Placeholder for predictions

    if model == "Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.slider("C (regularization strength)", 0.01, 1.0, step=0.01, key='C')
        kernel_option = st.sidebar.radio("Kernel", ("Radial Basis Function", "Linear", "Polynomial"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        # Map kernel names to valid values
        kernel_mapping = {"Radial Basis Function": "rbf", "Linear": "linear", "Polynomial": "poly"}
        kernel = kernel_mapping[kernel_option]

        if st.sidebar.button("Classify", key='classify_svm'):
            st.subheader("Support Vector Machine Results")
            classifier = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            classifier.fit(X_train, y_train)
            accuracy = classifier.score(X_test, y_test)
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1] 
            st.write("Accuracy = ", round(accuracy, 2))
            st.write("Precision = ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall = ", round(recall_score(y_test, y_pred), 2))

            

    elif model == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.slider("C (regularization strength)", 0.01, 10.0, step=0.01, key='C_lr')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        if st.sidebar.button("Classify", key='classify_lr'):
            st.subheader("Logistic Regression Results")
            classifier = LogisticRegression(C=C, max_iter=max_iter)
            classifier.fit(X_train, y_train)
            accuracy = classifier.score(X_test, y_test)  # FIXED
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1]
            st.write("Accuracy = ", round(accuracy, 2))
            st.write("Precision = ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall = ", round(recall_score(y_test, y_pred), 2))


    elif model == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of trees in forest", 10, 200, step=10, key='n_estimators')
        max_depth = st.sidebar.slider("Max depth of tree", 1, 20, key='max_depth')

        if st.sidebar.button("Classify", key='classify_rf'):
            st.subheader("Random Forest Results")
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            classifier.fit(X_train, y_train)
            accuracy = classifier.score(X_test, y_test)  # FIXED
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)[:, 1]
            st.write("Accuracy = ", round(accuracy, 2))
            st.write("Precision = ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall = ", round(recall_score(y_test, y_pred), 2))


    # Plot Metrics
    if classifier and metrics_list:
        plot_metrics(metrics_list, y_test, y_pred, y_proba)


if __name__ == "__main__":
    main()
