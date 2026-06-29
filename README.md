# Hospital Readmission Risk Analysis
### Tools: Python, Pandas, Matplotlib, Seaborn, Scikit-learn
### Author: Amir Ali | MS Business Analytics, Siena University
### Dataset: Diabetes 130-US Hospitals (1999-2008) | UCI Machine Learning Repository

---

## Project Overview
This project analyzes 30-day hospital readmission risk for diabetic patients 
using real clinical data from 130 US hospitals collected between 1999 and 2008. 
The goal is to identify which patient characteristics are most strongly associated 
with dangerous readmission within 30 days of discharge and build a predictive 
model to classify patients as high risk or low risk before they leave the hospital.

Hospital readmission within 30 days is one of the most costly and preventable 
problems in US healthcare. The Hospital Readmissions Reduction Program (HRRP) 
financially penalizes hospitals with excessive readmission rates — making early 
identification of high risk patients a critical operational and financial priority 
for every major hospital system in America.

---

## Industry Context — Why This Analysis Matters
The US healthcare industry is undergoing a fundamental shift from fee-for-service 
payment — where hospitals earn money per procedure — to Value Based Purchasing (VBP) 
— where hospitals earn money based on patient outcomes and care quality.

Three specific programs make hospital readmission reduction a direct financial priority:

**Hospital Readmissions Reduction Program (HRRP)** — the US government financially 
penalizes hospitals whose 30-day readmission rates exceed national benchmarks. 
Penalties can reach up to 3% of all Medicare payments — millions of dollars for 
large hospital systems.

**Value Based Purchasing (VBP)** — hospitals earn bonuses or face penalties based 
on outcome metrics including readmission rates, patient satisfaction, and chronic 
disease management quality.

**Star Ratings** — CMS rates every hospital on a 1-5 star scale visible to the 
public. Readmission rates directly impact star ratings which affect patient 
acquisition, insurance contracts, and reputation.

This analysis directly supports hospital efforts to identify high risk patients 
before discharge — enabling targeted interventions that reduce readmissions, 
improve star ratings, and avoid government penalties.

Additionally — medication adherence after discharge is one of the largest untracked 
drivers of readmission. Diabetic patients who cannot afford brand name insulin 
($300+ per month) may achieve adherence through biosimilar alternatives ($35 per 
month) — a clinical and financial consideration not captured in this dataset but 
critical to a complete readmission prevention strategy.

---

## Dataset
- Source: UCI Machine Learning Repository — Diabetes 130-US Hospitals Dataset
- 101,766 patient records from 130 US hospitals
- 50 original features representing patient demographics, clinical history, 
  medications, diagnoses, and hospital outcomes
- 10 years of data: 1999 to 2008
- Target variable: readmitted within 30 days (1 = dangerous, 0 = safe)
- 11.16% of patients were dangerously readmitted within 30 days

---

## Data Cleaning
Starting with 101,766 patients and 50 columns — ending with 101,745 
clean patients and 43 columns after the following steps:

- Dropped 7 columns with over 40% missing data or no analytical value 
  (weight, payer_code, medical_specialty, max_glu_serum, A1Cresult, 
  encounter_id, patient_nbr)
- Converted placeholder ? values to NaN for consistent missing value handling
- Filled missing race and secondary diagnosis values with Unknown 
  rather than deleting patients
- Deleted 21 corrupted rows where primary diagnosis was completely missing
- Simplified readmitted column from 3 categories (NO, >30, <30) to 
  binary (0 = not dangerous, 1 = dangerous readmission within 30 days)

---

## Key Business Questions
1. Which age group has the highest dangerous readmission rate?
2. Do patients with more prior inpatient visits get readmitted more?
3. Does longer hospital stay reduce readmission risk?
4. Does having more diagnoses increase readmission risk?
5. Which numeric features correlate most strongly with readmission?

---

## Key Findings

### Finding 1 — Age is a strong predictor with a surprising pattern
- The 20-30 age group had the HIGHEST readmission rate at 14.25%
- This contradicts the assumption that elderly patients are always highest risk
- Children (0-10) had the lowest rate at 1.86%
- Middle-aged and elderly groups ranged between 9-12% — relatively consistent
- Likely driven by young adults managing diabetes independently for the first 
  time without established self-management habits, higher Type 1 diabetes 
  prevalence, lifestyle factors, and insurance coverage gaps

### Finding 2 — Prior inpatient visits is the strongest numeric predictor
- Readmitted patients averaged 1.22 prior inpatient visits
- Non-readmitted patients averaged 0.56 prior inpatient visits
- Readmitted patients had MORE THAN DOUBLE the prior hospital visits
- Correlation with readmission: 0.17 — highest of all numeric features
- Past hospital behavior is the strongest available signal of future risk

### Finding 3 — Hospital stay and medication count are weak predictors
- Readmitted patients stayed 4.77 days vs 4.35 days for safe patients
- Readmitted patients received 16.90 medications vs 15.91 for safe patients
- Both differences are modest — these variables reflect patient complexity 
  rather than directly causing readmission (confounding effect)

### Finding 4 — Critical gap in diabetes monitoring
- 94.7% of patients never had their blood glucose tested during their visit
- 83.3% never had their HbA1c tested — the most important diabetes management test
- Research shows hospitals that routinely test HbA1c have lower readmission rates
- This represents a systemic gap in diabetes care quality across 130 US hospitals

---

## Visualizations
- Bar chart: 30-day readmission rate by age group — revealing the 
  surprising peak in the 20-30 age bracket
- Bar chart: Average prior inpatient visits by readmission status — 
  showing the doubling effect for high risk patients
- Boxplot: Medication count distribution by readmission status — 
  confirming medications as a weak predictor
- Correlation heatmap: All numeric feature relationships — validating 
  number_inpatient as the strongest numeric predictor

---

## Machine Learning Model
- Model: Logistic Regression with class balancing
- Features: All 42 available features after cleaning and encoding
- Encoding: Label encoding for categorical variables
- Scaling: StandardScaler for feature normalization
- Train/Test Split: 80% training (81,396 patients) / 20% testing (20,349 patients)
- Class imbalance handling: class_weight='balanced' to address the 
  9:1 ratio of safe to dangerous patients

### Model Results
| Metric | Safe Patients (0) | Readmitted Patients (1) |
|--------|------------------|------------------------|
| Precision | 0.92 | 0.17 |
| Recall | 0.69 | 0.51 |
| F1 Score | 0.79 | 0.26 |
| Support | 18,066 | 2,283 |
| Overall Accuracy | 67% | |

### Why accuracy is 67% and not higher
Individual feature correlations with readmission are modest — maximum 0.17 
for number_inpatient. This is consistent with published healthcare research 
on diabetic readmission and reflects the multifactorial nature of hospital 
readmission where no single variable dominates. The model captures combined 
feature interactions that individual correlations cannot reveal.

---

## Critical Limitation — Social Determinants of Health
This analysis is built entirely on clinical variables recorded inside 
the hospital. However readmission risk is significantly shaped by factors 
that happen outside the hospital — factors this dataset cannot capture.

These include:
- Income and financial barriers to medication after discharge
- Housing stability and living situation
- Family and social support systems
- Proximity to pharmacy and follow-up care
- Mental health status
- Dietary habits and lifestyle after discharge
- Whether patients understand and follow discharge instructions
- Medication adherence after leaving hospital
- Brand vs biosimilar drug affordability affecting medication adherence

Research shows social determinants of health explain 30-50% of health 
outcomes — yet they are absent from nearly all hospital clinical databases 
including this one. A patient discharged with perfect clinical care who 
cannot afford their insulin or has no family support is very likely to 
return within 30 days. This model cannot see that.

This is not a flaw unique to this project — it reflects a systemic gap 
in how healthcare data is collected in the United States.

---

## Business Recommendations
1. Flag patients aged 20-30 for intensive post-discharge follow up — 
   this group is counterintuitively the highest risk segment
2. Use prior inpatient visit count as a primary risk screening tool — 
   patients with 2 or more prior admissions in the past year need 
   case management support before discharge
3. Implement routine HbA1c testing for all diabetic admissions — 
   94.7% of patients in this dataset never received this critical test
4. Do not rely on hospital stay length or medication count as primary 
   risk indicators — these are weak predictors that reflect complexity 
   rather than causing readmission directly
5. Invest in collecting social determinants data — income, housing, 
   family support — to build a truly complete readmission risk model
6. Explore biosimilar insulin options for cost-sensitive patients — 
   medication affordability directly impacts post-discharge adherence 
   and therefore readmission risk

---

## Model Limitations and Future Work
- Class imbalance (9:1 ratio) limits recall for dangerous readmissions 
  despite balancing techniques
- Label encoding of categorical variables does not capture ordinal 
  relationships — future work should use one-hot encoding
- Diagnosis codes (diag_1, diag_2, diag_3) contain 700+ unique ICD-9 
  codes — grouping these into clinical categories would improve model performance
- Future models should test Random Forest and XGBoost which typically 
  handle class imbalance and categorical data better than logistic regression
- Incorporating social determinants of health data would be the single 
  biggest improvement to predictive accuracy
- Star rating impact and VBP performance metrics should be incorporated 
  as outcome measures in future iterations

---

## Technical Skills Demonstrated
- Large scale data cleaning and preprocessing (101,766 records, 50 features)
- Multiple missing value strategies (deletion, imputation, Unknown labeling)
- Exploratory data analysis with groupby aggregations
- Data visualization with Matplotlib and Seaborn
- Categorical encoding with LabelEncoder
- Feature scaling with StandardScaler
- Logistic regression with class imbalance handling
- Model evaluation: accuracy, precision, recall, F1 score
- Critical analysis of model limitations
- Business interpretation of analytical findings
- Healthcare domain knowledge — HRRP, VBP, Star Ratings, 
  Social Determinants of Health, Medication Adherence, Biosimilars
