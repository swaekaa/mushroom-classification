# 🍄 Mushroom Classification Web App

This is a **binary classification web app** built using **Streamlit**. The app allows users to classify whether a mushroom is **edible or poisonous** based on its attributes. The classification models available are **Support Vector Machine (SVM), Logistic Regression, and Random Forest**.

---

## 🚀 Features
- **Interactive UI** built with **Streamlit**.
- **Model Selection**: Choose between **SVM, Logistic Regression, or Random Forest**.
- **Hyperparameter Tuning**: Adjust model parameters via the sidebar.
- **Performance Metrics**: View **Confusion Matrix, ROC Curve, and Precision-Recall Curve**.
- **Data Preview**: Option to view the raw mushroom dataset.

---

## 📂 Dataset
The app uses the **Mushrooms Dataset** from [Kaggle](https://www.kaggle.com/uciml/mushroom-classification), which contains categorical features describing mushrooms.

**Dataset Information:**
- **Classes**: Edible (e) or Poisonous (p)
- **Features**: 22 categorical attributes such as cap-shape, odor, gill-color, etc.

---

## 🛠 Installation & Setup
### **1. Clone the Repository**
```sh
git clone https://github.com/yourusername/mushroom-classification-app.git
cd mushroom-classification-app
```

### **2. Install Dependencies**
Make sure you have Python installed, then install the required libraries:
```sh
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```sh
streamlit run mushroom_app.py
```

---

## 📊 How to Use the App
1. Select a **classification model** from the sidebar.
2. Adjust **hyperparameters** as needed.
3. Click **"Classify"** to train and test the model.
4. View the **accuracy, precision, and recall scores**.
5. (Optional) Select metrics like **Confusion Matrix, ROC Curve, and Precision-Recall Curve** to visualize results.
6. (Optional) Check **"Show raw data"** to inspect the dataset.

---

## 🖥️ Technologies Used
- **Python**
- **Streamlit** (for UI)
- **Pandas, NumPy** (for data processing)
- **Scikit-Learn** (for machine learning models)
- **Matplotlib, Seaborn** (for visualization)

---

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** the repository, create a new branch, and submit a **pull request**.

---

## 📜 License
This project is **open-source** under the **MIT License**.

---

## 📬 Contact
For questions or suggestions, feel free to reach out:
- **GitHub:** [swaekaa](https://github.com/yourusername)
- **Email:** sawariaekaansh@gmail.com
---

🌟 **Star this repo if you find it useful!** 🚀

