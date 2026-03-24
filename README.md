# Crop Yield Prediction and Recommendation Using Machine Learning

## 📚 Project Overview

This project helps **farmers and agricultural experts recommend the best crops to grow** based on soil and weather conditions. It uses artificial intelligence (machine learning) to analyze data and make intelligent predictions.

**Simple Explanation:** Think of this as a smart advisor that tells you "Based on your soil nutrients, temperature, and rainfall, you should grow rice or wheat" - just like a weather app tells you if you need an umbrella!

---

## 🎯 What Does This Project Do?

### Main Goal
To predict and recommend the **most suitable crop** for a given area based on:
- **Soil nutrients** (Nitrogen, Phosphorus, Potassium)
- **Weather conditions** (Temperature, Humidity, pH level, Rainfall)

### Who Benefits?
- 🌾 **Farmers** - Know which crops will grow best in their fields
- 🎓 **Agricultural experts** - Better planning and crop selection
- 📊 **Researchers** - Understanding crop-environment relationships
- 💼 **Policy makers** - Making informed decisions about agriculture

---

## 📊 The Datasets Used

### 1. **Crop Recommendation Dataset** (Main Dataset)
**What it contains:** 
- Information about different crops (rice, maize, chickpea, kidney beans, pigeon peas)
- Required soil conditions (N, P, K nutrients)
- Required weather conditions (temperature, humidity, pH, rainfall)
- How many records: 2,200+ crop recommendations

**Simple Example:**
```
For Rice:        Needs N=80-90, P=40-50, K=40-43, Temp=20-23°C, Humidity=80-82%, pH=6.5, Rainfall=200mm
For Chickpea:    Needs N=20-40, P=55-80, K=75-85, Temp=17-20°C, Humidity=15-20%, pH=7.5, Rainfall=70-90mm
```

### 2. **District-Wise Rainfall Dataset**
**What it contains:**
- Monthly rainfall data for different Indian districts
- Data for all 12 months of the year
- Helps understand rainfall patterns in different regions

### 3. **Agriculture Crop Production Dataset**
**What it contains:**
- Actual crop production data from 2001 onwards
- Information about area planted, production achieved, and yield
- Data from different states and districts across India

---

## 🔄 How Does The Project Work?

### Step 1️⃣: **Data Collection**
- We gather information from three different sources
- Each dataset contains different but related agricultural information

### Step 2️⃣: **Data Cleaning** (Making data ready for analysis)
The program checks and fixes:
- ❌ **Missing information** - If some values are missing, we fill them intelligently
- ❌ **Duplicate records** - If the same information appears twice, we remove it
- ❌ **Wrong data types** - We make sure numbers are numbers, not text
- ❌ **Outliers** - Very unusual values (we keep them in this case as they represent real farming situations)
- ❌ **Invalid records** - We remove data that doesn't make sense (like zero area or negative production)

### Step 3️⃣: **Feature Preparation**
- We organize the data into a clean format
- **Features** = Input information (soil nutrients, weather)
- **Target** = Output we want to predict (which crop is best)

### Step 4️⃣: **Feature Scaling**
**Why?** Different measurements have different ranges:
- Temperature: 0-50°C
- Rainfall: 0-300mm
- Nitrogen: 0-100

We normalize them so they're all on the same scale (like converting to percentages). This helps the AI learn better.

### Step 5️⃣: **Train-Test Split**
- We divide data into two parts:
  - **80% for training** - AI learns from this
  - **20% for testing** - We check if AI's predictions are correct
- This ensures we test on data the AI has never seen before

### Step 6️⃣: **Model Training** (Next Phase)
- The AI learns patterns from the training data
- It discovers rules like "If nitrogen is high and rainfall is low, then wheat is best"

### Step 7️⃣: **Prediction** (Next Phase)
- A farmer enters their soil/weather data
- AI predicts the best crop using what it learned

---

## 📁 Project Structure

```
CSYP-ML/
│
├── Datasets/                                    # Data files
│   ├── Crop_recommendation.csv                 # Main crop data
│   ├── DistrictWiseRainfallNormal.csv         # Rainfall information
│   └── IndiaAgricultureCropProduction.csv     # Production data
│
├── Models/                                      # Code files
│   └── model.ipynb                            # Main Jupyter Notebook
│
└── README.md                                    # This file
```

---

## 🚀 How to Use This Project

### Prerequisites (What You Need)
1. **Python** (a programming language) - version 3.7 or higher
2. **Jupyter Notebook** (a tool to run Python code in your browser)
3. Some Python libraries:
   - `pandas` - for working with data tables
   - `numpy` - for mathematical calculations
   - `scikit-learn` - for machine learning

### Installation Steps

**For Windows Users:**
```bash
# Open Command Prompt and type:
pip install pandas numpy scikit-learn jupyter matplotlib
```

**For Mac/Linux Users:**
```bash
# Open Terminal and type:
pip3 install pandas numpy scikit-learn jupyter matplotlib
```

### Running The Project

1. **Navigate to the project folder:**
   ```bash
   cd "C:\Users\anees\OneDrive\Desktop\ML PROJECT\CSYP-ML"
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the model file:**
   - Your browser will open automatically
   - Click on `Models/model.ipynb`

4. **Run the cells in order:**
   - Click on the first code block (cell)
   - Press `Shift + Enter` to run it
   - Continue with each cell below it
   - Watch the output appear below each cell

---

## 📊 Understanding The Code Sections

### Section 1: Import Libraries
```
What it does: Loads tools we need (like opening a toolbox)
Why: Each library does specific jobs
```

### Section 2: Load Datasets
```
What it does: Reads the three CSV files
Why: We need the data to work with
```

### Section 3: Data Preprocessing (The Biggest Part)
This section is divided into meaningful chunks:

#### Dataset 1: Crop Recommendation
- **[1.1] Initial Data Inspection** - Look at the data structure
- **[1.2] Missing Values Analysis** - Check if any information is missing
- **[1.3] Duplicate Rows** - Remove repeated records
- **[1.4] Statistical Summary** - Understand the data (averages, ranges)
- **[1.5] Data Type Validation** - Ensure numbers are numbers
- **[1.6] Outlier Detection** - Find unusual values
- **[1.7] Crop Label Analysis** - See which crops are in the data
- **[1.8] Encoding Labels** - Convert crop names to numbers (AI needs numbers)
- **[1.9] Feature Correlation** - See which factors influence each other
- **[1.10] Feature Preparation** - Organize data for AI
- **[1.11] Handle Missing Values** - Fill in any blanks smartly
- **[1.12] Feature Scaling** - Normalize all measurements
- **[1.13] Train-Test Split** - Divide data for training and testing

#### Dataset 2: Rainfall Data
- Similar cleaning and checking process

#### Dataset 3: Production Data
- Remove invalid records and clean the data

### Section 4: Data Validation
```
What it does: Double-checks that everything is correct
Why: Ensures our AI will learn from good quality data
```

---

## 📈 What The Output Tells You

### Example Output:
```
Dataset shape: (2200, 8)
↑ This means: 2200 records, 8 pieces of information per record

Missing values: 0
↑ This means: All data is complete, nothing missing

Unique crops: 22
↑ This means: System can predict 22 different crops

Training set size: 1760 (80%)
Testing set size: 440 (20%)
↑ This means: Properly split for learning and testing
```

---

## 🔬 Key Concepts Explained Simply

### **Supervised Learning**
Imagine teaching a child vegetables by showing pictures with labels ("This is carrot", "This is broccoli"). The AI learns the same way - we show it examples with correct answers.

### **Feature Scaling**
Like converting all measurements to percentages (0-100) so they're comparable. Without this, high rainfall values (100-300) would dominate over pH values (5-8) in AI's thinking.

### **Train-Test Split**
Like studying for a test using past year papers, then taking a new test to check if you really learned. We don't test on data the AI has already seen (that would be cheating!).

### **Label Encoding**
AI understands numbers, not words. So we convert:
- "Rice" → 1
- "Wheat" → 2
- "Maize" → 3
And so on...

---

## ✅ Quality Checks The Program Does

The system validates data by checking:

1. **Data Completeness** - No missing values
2. **Data Consistency** - No contradictions
3. **Data Validity** - Values make sense (no negative area, no temperature below -50°C)
4. **Data Balance** - Each crop has roughly equal representation
5. **Data Leakage** - Training and testing data don't overlap

---

## 🎓 Learning Outcomes

After running this project, you'll understand:
- ✅ How data is cleaned and prepared
- ✅ What preprocessing means and why it's important
- ✅ How machine learning data is organized
- ✅ The importance of data quality
- ✅ Basic ML workflow and best practices

---

## 📝 Files Generated

After running the notebook, the system creates no new data files. Instead, it:
- Analyzes existing data
- Shows statistics and insights
- Prepares data for future model training
- Validates data quality

---

## ⚠️ Important Notes

### For Future Development
Once data preprocessing is complete, the next steps would be:
1. **Train models** - Feed cleaned data to AI algorithms
2. **Test performance** - Check accuracy of predictions
3. **Build interface** - Create app for farmers to use
4. **Make predictions** - Recommend crops for real farms

### Current Limitations
- This project **only prepares data** - it doesn't make predictions yet
- To make actual predictions, ML models need to be trained (next phase)
- The system assumes data quality is important (which it is!)

---

## 🐛 Troubleshooting

### Problem: "Module not found: pandas"
**Solution:** Run `pip install pandas` in Command Prompt

### Problem: "CSV file not found"
**Solution:** Check that dataset files are in the `Datasets/` folder

### Problem: A cell shows error
**Solution:** 
1. Read the error message carefully
2. Check if you ran previous cells (they must run in order)
3. Verify data file paths are correct

---

## 📚 Additional Resources

### To Learn More:
- **Pandas Documentation:** https://pandas.pydata.org/docs/
- **Scikit-learn Guide:** https://scikit-learn.org/stable/
- **ML Concepts:** https://www.kaggle.com/learn

### Datasets Information:
- **Crop Recommendation:** Environmental and soil requirements for crops
- **Rainfall Data:** 30-year average rainfall patterns by district
- **Production Data:** Actual output data from Indian agriculture ministry

---

## 👥 Project Authors
- **Vishal Anand**
- **Aneesh Jain**

---

## 📞 Support

If you encounter issues:
1. Check this README file first
2. Review the comments in the code
3. Check data file formats and locations
4. Verify all required libraries are installed

---

## 🎯 Quick Start Checklist

- [ ] Download Python 3.7+
- [ ] Install required libraries: `pip install pandas numpy scikit-learn jupyter`
- [ ] Navigate to project folder
- [ ] Run `jupyter notebook`
- [ ] Open `Models/model.ipynb`
- [ ] Run cells from top to bottom
- [ ] Check outputs for data quality insights

---

## 📊 What Happens In Each Phase

```
Phase 1: DATA COLLECTION
          ↓
Phase 2: DATA CLEANING ← YOU ARE HERE
          ↓
Phase 3: MODEL TRAINING (Next)
          ↓
Phase 4: MODEL TESTING (Next)
          ↓
Phase 5: PREDICTION & DEPLOYMENT (Future)
```

---

## 🌟 Why This Matters

Agriculture feeds the world. Using AI to recommend the right crops:
- 🌱 Increases crop yields
- 💰 Improves farmer income
- 🌍 Reduces wasteful farming
- 🔍 Helps plan better crop rotation
- 📈 Supports food security

---

## 📄 Final Notes

This project demonstrates the critical importance of **data preprocessing** - arguably the most important step in machine learning. As data scientists say:

> "Garbage in, Garbage out" - High quality cleaned data leads to good AI models.

By properly cleaning and validating our agricultural data, we ensure that any ML models trained on this data will make reliable crop recommendations that actually help farmers!

---

**Last Updated:** March 2026  
**Status:** ✅ Data Preprocessing Complete - Ready for Model Training

---

## How to Read This README if You're New to Tech

1. **Start with "Project Overview"** - Get the big picture
2. **Read "What Does This Project Do"** - Understand the purpose
3. **Skip the technical parts initially** - Come back later
4. **Read "How to Use This Project"** - Learn how to run it
5. **Run the code** - See what happens
6. **Review "Understanding The Code Sections"** - Learn what each part does
7. **Check "Key Concepts Explained Simply"** - Understand the ideas

**Remember:** Technology might seem complicated, but breaking it down into small pieces makes it simple! 🚀

### by Vishal Anand and Aneesh Jain
