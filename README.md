# Customer Segmentation using Unsupervised Machine Learning

# Project Overview
This project performs customer segmentation using **K-Means clustering**, an unsupervised machine learning algorithm. Customers are grouped based on their **annual income** and **spending behavior** to extract meaningful business insights.

# Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

# Dataset
The dataset contains customer information including:
- Age
- Annual Income
- Spending Score

# How It Works
1. Data loading and preprocessing
2. Feature scaling using StandardScaler
3. Optimal cluster selection using Elbow Method
4. Customer segmentation using K-Means
5. Visualization of clusters

# Results
The model successfully identifies distinct customer groups such as:
- High income – high spending customers
- Low income – low spending customers
- Average customers

# How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py
