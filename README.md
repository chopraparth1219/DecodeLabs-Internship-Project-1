# DecodeLabs-Internship-Project-1
# Project 1: Advanced EDA & Feature Engineering

**DecodeLabs Industrial Training Kit – Data Science Project 1**

## Objective

Transform raw, chaotic transaction data into a mathematically clean dataset ready for machine learning algorithms.  
This project demonstrates:

- Statistical imputation of missing values (Median and KNN).
- Outlier detection using the Interquartile Range (IQR) and Z‑score methods.
- Outlier neutralisation via capping (Winsorization) – no row deletion.
- Engineering of three or more new predictive features.
- Fully vectorised Pandas/NumPy operations (no Python loops).

## Dataset

- **File:** `data1.csv`:  
  `OrderID`, `Date`, `customerID`, `Product`, `Quantity`, `UnitPrice`, `TotalPrice`.
- Missing values and outliers were synthetically added for demonstration purposes (real datasets naturally contain such imperfections).

## Steps Performed (IPO Architecture)

### 1. Input – Load and Inspect Data
- Load CSV using `pandas`.
- Identify missing values and initial outliers.

### 2. Process – Data Cleaning and Feature Engineering

#### a) Handling Missing Values
- **Global Median Imputation** – used for skewed numerical columns (`Quantity`, `UnitPrice`).
- **KNN Imputation** (optional) – implemented for comparison; captures multivariate relationships.

#### b) Outlier Detection and Neutralisation
- **IQR Method**:
  - Calculate Q1, Q3, and IQR.
  - Define lower and upper bounds: `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR`.
  - Cap extreme values to these bounds (Winsorization) – preserves row count.
- **Z‑score Method**:
  - Compute `z = |(x - μ) / σ|`.
  - Values with `z > 3` are considered outliers.
  - Clip them to `μ ± 3σ`.

#### c) Feature Engineering (Three or More New Predictive Features)
1. `TotalPrice_per_Quantity` – average price per unit (handles division by zero).
2. `IsHighValue` – binary flag (1 if `TotalPrice` > median, else 0).
3. `LogTotalPrice` – log transformation to reduce skewness.
4. Date‑based features (if `Date` column present): `Year`, `Month`, `DayOfWeek`.

### 3. Output – Cleaned Dataset and Statistics
- Final cleaned data saved as `data1_cleaned.csv`.
- Statistical summary (`.describe()`) printed to console.
- Missing value count after imputation = 0.

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, vectorised operations |
| `numpy` | Numerical computations (Z‑score, capping) |
| `scikit-learn` | `KNNImputer` for advanced imputation |

## How to Run

1. Clone the repository or download the script `project1_eda.py`.
2. Ensure `data1.csv` is in the same folder.
3. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn
