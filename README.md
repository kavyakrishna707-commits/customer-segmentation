# Customer Segmentation Using K-Means Clustering

## Description
A beginner-friendly Machine Learning project that groups customers based on their annual income and spending score using the K-Means clustering algorithm.

**This project is an example of Unsupervised Learning**, where the algorithm discovers patterns and groups in data without predefined target labels.

The project uses Python, Flask, HTML, and CSS to provide a simple web interface for customer segmentation.

## Technologies Used
- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- HTML and CSS
- CSV Dataset

## Features
- Customer data analysis using a CSV file
- Customer grouping using K-Means clustering
- Visualization of customer segments
- Prediction of the cluster for a new customer
- Interactive web interface

## Project Structure
```text
customer-segmentation/
├── app.py
├── customers.csv
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Machine Learning Algorithm
**K-Means Clustering** is an unsupervised learning algorithm that groups similar data points into K clusters based on their features.

## Type of Machine Learning
**Unsupervised Learning**

The model identifies patterns in customer data without requiring predefined customer group labels.

## Objective
To understand customer behaviour by identifying groups with similar income levels and spending patterns.

## How to Run
1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:

   ```bash
   python app.py
   ```

3. Open your browser and visit:

   `http://127.0.0.1:5001`
