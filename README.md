# Crop Yield Prediction and Recommendation Using Machine Learning

## Project Overview

This project recommends suitable crops based on soil nutrients and weather conditions using machine learning. The current notebook focuses on data collection, cleaning, validation, and feature preparation to support model training.

---

## Objective

Prepare a dataset to predict and recommend the most suitable crop for a given area using:

- Soil nutrients: Nitrogen (N), Phosphorus (P), Potassium (K)
- Weather conditions: Temperature, Humidity, pH, Rainfall

## Intended Users

- Farmers
- Agricultural experts
- Researchers
- Policy makers

---

## Datasets

### 1) Crop Recommendation Dataset (Primary)

Contains crop-wise soil and weather requirements.

- Crops include rice, maize, chickpea, kidney beans, pigeon peas, and others
- Features: N, P, K, temperature, humidity, pH, rainfall
- Size: 2,200+ records

Example:
```
Rice:     N=80-90, P=40-50, K=40-43, Temp=20-23C, Humidity=80-82%, pH=6.5, Rainfall=200mm
Chickpea: N=20-40, P=55-80, K=75-85, Temp=17-20C, Humidity=15-20%, pH=7.5, Rainfall=70-90mm
```

### 2) District-Wise Rainfall Dataset

- Monthly rainfall for Indian districts (12 months)
- Used to understand rainfall patterns by region

### 3) Agriculture Crop Production Dataset

- Crop production data from 2001 onwards
- Area, production, and yield across states/districts

---

## Workflow

1) Data collection from the three sources
2) Data cleaning
   - Handle missing values
   - Remove duplicates
   - Validate data types
   - Review outliers (kept where they represent real conditions)
   - Remove invalid records (e.g., zero area or negative production)
3) Feature preparation (define features and target)
4) Feature scaling to normalize different ranges
5) Train-test split (80/20) for evaluation

Planned next steps:

6) Model training
7) Prediction and deployment (future)

---

## Project Structure

```
CSYP-ML/
|-- Datasets/
|   |-- Crop_recommendation.csv
|   |-- DistrictWiseRainfallNormal.csv
|   `-- IndiaAgricultureCropProduction.csv
|-- Models/
|   `-- model.ipynb
`-- README.md
```

---

## Setup and Usage

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Python libraries: pandas, numpy, scikit-learn, matplotlib

### Install Dependencies

Windows:
```bash
pip install pandas numpy scikit-learn jupyter matplotlib
```

macOS/Linux:
```bash
pip3 install pandas numpy scikit-learn jupyter matplotlib
```

### Run the Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `Models/model.ipynb`
3. Run cells from top to bottom

---

## Notebook Sections (High Level)

- Import libraries
- Load datasets
- Data preprocessing and validation
  - Missing values, duplicates, data types, outliers
  - Label encoding, feature correlation, feature scaling
  - Train-test split

---

## Output

Example:
```
Dataset shape: (2200, 8)
Missing values: 0
Unique crops: 22
Training set size: 1760 (80%)
Testing set size: 440 (20%)
```

---

## Key Concepts

- Supervised learning: train using labeled examples
- Feature scaling: normalize features so no single feature dominates due to scale
- Train-test split: evaluate on unseen data
- Label encoding: convert crop names into numeric labels for model training

---

## Quality Checks

The preprocessing validates:

1. Data completeness (no missing values)
2. Data consistency
3. Data validity (e.g., no negative area; realistic ranges)
4. Data balance (reasonable class distribution)
5. Data leakage (train/test separation)

---

## Learning Outcomes

After running the notebook, you will understand:

- Practical data cleaning and validation
- Feature preparation for machine learning
- Why scaling and train-test splitting matter
- A standard end-to-end ML workflow for tabular data

---

## Generated Files

The notebook does not generate new dataset files. It reads existing CSV files, performs analysis/validation, and prepares data in-memory for model training.

---

## Notes

- Current scope: data preprocessing and validation
- Limitation: model training and prediction are not implemented yet

Future work:

1. Train and compare models
2. Evaluate performance
3. Build a simple user interface for recommendations
4. Deploy for practical use

---

## Troubleshooting

- "Module not found: pandas": run `pip install pandas`
- "CSV file not found": confirm the CSV files exist under `Datasets/`
- Notebook cell errors: run cells in order and verify file paths

---

## Resources

- Pandas: https://pandas.pydata.org/docs/
- Scikit-learn: https://scikit-learn.org/stable/
- Kaggle Learn (ML fundamentals): https://www.kaggle.com/learn

---

## Authors

- Vishal Anand
- Aneesh Jain

---

## Support

If you encounter issues, check this README first, then review notebook outputs/errors and confirm dependencies and dataset paths.

---

Last Updated: March 2026
Status: Data preprocessing complete; ready for model training.
