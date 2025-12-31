intrusion detection system (IDS)
######################################################################
 SOURCE CODE-->https://drive.google.com/file/d/1p-2PYAGZ9HkvYaKP3IuQt5ZqnfDtfNks/view?usp=drive_link
######################################################################
🛡️ IDS Web Tester

Machine Learning–Based Intrusion Detection System with Streamlit UI

This repository contains a complete implementation of a machine-learning-based Intrusion Detection System (IDS) along with a Streamlit web application that allows users to test trained models on network traffic data in CSV format.

The project is designed for academic use and enables instructors or reviewers to run the system end-to-end without retraining models or downloading full datasets.

📌 Project Objectives

Implement an IDS using classical and gradient-boosting ML models

Apply feature selection and proper preprocessing

Provide an interactive UI for testing trained models

Enable reproducible evaluation for university coursework

🚀 Features

Upload CSV files containing network flow features

Support for multiple datasets:

CIC-IDS-2017

CSE-CIC-IDS-2018

CIC-DDoS-2019

Multiple trained models:

LightGBM

Random Forest

XGBoost

CatBoost

Real-time inference (no retraining required)

Visualization of prediction distribution

Optional evaluation (Accuracy, Precision, Recall, F1-score)

Downloadable results (CSV + JSON summary)

📂 Project Structure
IDS_Project/
├── app.py                      # Streamlit application
├── src/                        # Training & preprocessing scripts
├── outputs/
│   └── models/                 # Pre-trained models (.joblib)
├── data/
│   └── fs/                     # Feature-selected column lists  
├── sample_inputs/              # Small CSV files for quick testing
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore

⚙️ Requirements

Python 3.10 or higher (recommended)

Operating System: Windows / Linux / macOS

Install dependencies using:

pip install -r requirements.txt

▶️ Running the Application

From the project root directory, run:

streamlit run app.py


Then open the following URL in your browser:

http://localhost:8501

🧪 Quick Test (Recommended for Instructors)

Sample input files are provided to allow testing without downloading full datasets.

Use files from:

sample_inputs/


Example:

sample_inputs/ids2017_sample.csv

sample_inputs/ids2018_sample.csv

sample_inputs/ddos2019_sample.csv

Steps:

Launch the app

Select the correct dataset

Upload a sample CSV

Click Run Detection

View results and download outputs

🧠 How the System Works (High-Level)

User selects dataset and model

CSV file is uploaded and validated

Required feature columns are checked

Data is preprocessed to match training pipeline

Trained model performs inference

Predictions are visualized and exported

📊 Evaluation (Optional)

If the uploaded CSV file contains a Label column:

The application computes:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Confusion Matrix and Classification Report are displayed

If no Label column exists, the system runs in inference-only mode.

📦 Datasets

Due to size limitations, full datasets are not included in this repository.

Original datasets can be obtained from:

CIC-IDS-2017

CSE-CIC-IDS-2018

CIC-DDoS-2019

Sample subsets are provided for demonstration and testing purposes.

🔁 Reproducibility Notes

Feature selection files ensure consistency between training and inference

No data leakage: resampling techniques are applied only during training

The application performs inference only, ensuring fast and deterministic results

🎓 Academic Use

This project was developed as part of a university coursework / academic project to demonstrate practical application of machine learning techniques in network security and intrusion detection.

📄 License

This project is intended for educational and research purposes.

✉️ Contact


For questions or academic review, please contact the project author.

