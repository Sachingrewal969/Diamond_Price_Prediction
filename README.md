💎 Diamond Price Prediction

📌 Project Overview
This project aims to predict the price of diamonds based on their physical attributes using machine learning models. The dataset includes features like carat, cut, color, clarity, depth, table, and dimensions. The goal is to build an accurate model to estimate diamond prices based on these characteristics.

Dataset
The dataset used for this project is sourced from https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv(https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv). It contains information about various diamonds, including their attributes and corresponding prices. The dataset is provided in csv format and can be found in the data directory.

🛠️ Tech Stack
Programming Language: Python 3.10
Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
ML Models: Linear Regression, Decision Trees, Random Forest, Gradient Boosting
Deployment: Flask API with a Pickle-based model

📊 Data Processing & Feature Engineering
✔ Data Cleaning (Handling missing values, outliers)
✔ Feature Engineering (Scaling, Encoding categorical features)
✔ Exploratory Data Analysis (EDA) with visualizations
✔ Model Training and Hyperparameter Tuning

🏆 Model Performance
Evaluation Metrics: RMSE, R² Score
Best Model: XGBoost with optimized hyperparameters

🚀 How to Run the Project
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/Diamond_Price_Prediction.git
cd Diamond_Price_Prediction
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Flask app

bash
Copy
Edit
python app.py
4️⃣ Access the web interface at http://127.0.0.1:5000/

📌 Results & Insights
Carat weight has the highest impact on price
Cut, color, and clarity significantly influence the price
Feature scaling and encoding improved model accuracy

📁 Project Structure
bash
Copy
Edit
📂 Diamond_Price_Prediction  
 ├── 📂 data             # Dataset and preprocessing  
 ├── 📂 notebooks        # EDA & model training notebooks  
 ├── 📂 model            # Trained model files  
 ├── 📂 app              # Flask web app  
 ├── requirements.txt    # Dependencies  
 ├── README.md           # Project Documentation  
 └── app.py              # Flask API  
🎯 Future Enhancements
✅ Deploy on a cloud platform (AWS/GCP)
✅ Implement advanced ML models (Neural Networks)
✅ Enhance the UI for better user interaction

# Live link
https://diamond-price-prediction-g2td.onrender.com/
