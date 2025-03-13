"""
Mental Health Prevalence Analysis and Mapping for Washington State
- Author: [Your Name]
- Date: [Current Date]

This script analyzes and maps the prevalence of mental health conditions
among adolescents (14-17 years) in Washington State using ML predictions
constrained by demographic data from the American Community Survey (ACS).

Requirements:
- pandas, numpy, matplotlib, geopandas
- scikit-learn, xgboost, lightgbm, imblearn
- gurobipy for optimization
- census API access
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import itertools
import pickle
import os
import joblib
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Census API
from census import Census
import us

# Optimization library
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

# Machine learning models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb
import shap

# Configure API key for Census data
CENSUS_API_KEY = "ed75ad333b249bcbda60ef3fbae2fc722c7a7b5e"
c = Census(CENSUS_API_KEY)

#################################################
# CONFIGURATION AND HELPER FUNCTIONS
#################################################

# Define variable mappings for NSCH dataset
name2col = {
    # Mental health outcomes
    'Anxiety': 'K2Q33A',          # 1 yes, 2 no
    'Depression': 'K2Q32A',       # 1 yes, 2 no
    'ADHD': 'K2Q31A',             # 1 yes, 2 no
    'Behavior Problems': 'K2Q34A', # 1 yes, 2 no
    
    # Demographics
    'Age': 'SC_AGE_YEARS',         # 1~17
    'Grouped age': 'SC_AGE_GROUP', # 1~4
    'Race': 'race7_21',            # 1~7
    'Sex': 'SC_SEX',               # 1 male, 2 female
    'Highest education': 'HIGRADE_TVIS', # 1~4
    'Language': 'HHLanguage_21',   # 1 primarily English, 2 others
    'Family size': 'FamCount_21',  # 1~3
    'Insurance': 'InsGap_21',      # 1 yes, 2 no
    'Employment': 'EmploymentSt_21', # 1 full-time, 2 part-time, 3 unemployed
    'Family income': 'ACEincome_21', # 1~4
    'Family structure': 'famstruct5_21', # 1~5
    'Parents mental health': 'A1_MENTHEALTH',
    'Parents born in US': 'A1_BORN',
}

# Reverse mapping
col2name = {v: k for k, v in name2col.items()}

# Variables used to train the model
NSCH_ACS_variables = ['Grouped age', 'Sex', 'Race', 'Highest education', 
                      'Language', 'Family size', 'Insurance', 'Employment', 
                      'Family income', 'Family structure']

# Feature variables mapping used for analysis
name2col_feature = {k: v for k, v in name2col.items() if k in NSCH_ACS_variables}

# Dictionary for demographic value meanings
var2value2name = {
    'Race': {
        1: 'Hispanic',
        2: 'White',
        3: 'Black',
        4: 'Asian',
        5: 'Others'
    },
    'Sex': {
        1: 'male',
        2: 'female'
    },
    'Highest education': {
        1: 'Less than high school',
        2: 'High school or GED',
        3: 'Some college or technical school',
        4: 'College degree or higher',
    },
    'Language': {
        1: 'primarily English',
        2: 'others'
    },
    'Insurance': {
        1: 'Consistantly insured over the past year',
        2: 'Consistantly uninsured or had periods without coverage'
    },
    'Employment': {
        1: 'At least one caregiver employed full-time or part-time',
        2: 'Caregiver(s) unemployed or working without pay'
    },
    'Family income': {
        1: 'Above the poverty line',
        2: 'Below the poverty line'
    },
    'Family structure': {
        1: 'Two parents, currently married',
        2: 'Two parents, not currently married',
        3: 'Single parent (mother or father)',
        4: 'Other family type'
    },
}

# List of features used in analysis
used_vars = list(var2value2name.keys())
outcomes = ['Anxiety', 'Depression', 'ADHD', 'Behavior Problems']
model_types = ['LogisticRegression', 'XGBoost', 'RandomForest', 'GradientBoost', 'NaiveBayes', 'LightGBM']

# Best model per outcome based on performance
out2best_model = {
    'Anxiety': 'LogisticRegression',
    'Depression': 'LogisticRegression',
    'ADHD': 'GradientBoost',
    'Behavior Problems': 'LogisticRegression'
}


#################################################
# STEP 1: MACHINE LEARNING PREDICTION MODEL
#################################################

def prepare_nsch_data():
    """
    Load and prepare NSCH data for machine learning model training.
    """
    print("Step 1.1: Loading and preparing NSCH data...")
    
    # Load the NSCH data
    NSCH = pd.read_csv('2021 NSCH_Topical_CAHMI_DRC.csv')
    
    # Create age group column
    NSCH['SC_AGE_GROUP'] = pd.cut(NSCH['SC_AGE_YEARS'], 
                                  bins=[0, 4, 9, 13, 17], 
                                  labels=[1, 2, 3, 4])
    NSCH = NSCH.dropna(subset=['SC_AGE_GROUP'])
    NSCH['SC_AGE_GROUP'] = NSCH['SC_AGE_GROUP'].astype(int)
    
    # Keep only 14-17 years old children in the dataset
    NSCH = NSCH[NSCH['SC_AGE_GROUP'] == 4]
    
    # Recategorize variables
    NSCH['race7_21'] = NSCH['race7_21'].apply(lambda x: x if x in [1, 2, 3, 4] else 5)
    NSCH['EmploymentSt_21'] = NSCH['EmploymentSt_21'].apply(lambda x: 1 if x in [1, 2] else 2)
    NSCH['famstruct5_21'] = NSCH['famstruct5_21'].apply(lambda x: 4 if x == 5 else x)
    NSCH['FamCount_21'] = NSCH['FamCount_21'].apply(lambda x: 1 if x == 1 else 2 if x in [2, 3, 4] else 3)
    NSCH['ACEincome_21'] = NSCH['ACEincome_21'].apply(lambda x: 1 if x in [1, 2] else 2)
    
    # Exclude invalid data
    for var in NSCH_ACS_variables:
        NSCH = NSCH[NSCH[name2col[var]] != 99]
        NSCH = NSCH[NSCH[name2col[var]] != 95]
        
    print(f'The number of entries in NSCH dataset: {NSCH.shape[0]}')
    return NSCH

def train_mental_health_models(NSCH):
    """
    Train machine learning models to predict mental health outcomes.
    """
    print("Step 1.2: Training mental health prediction models...")
    
    variables = [name2col[var] for var in used_vars]
    out2model_type2model = {}
    out2X_train = {}
    out2X_test = {}
    out2y_train = {}
    out2y_test = {}
    
    for out in outcomes:
        print(f"Training models for {out}")
        X_ori = NSCH[variables]
        idxs_missing_x = X_ori.isin([99, 95, 'NaN']).any(axis=1)
        
        y_ori = NSCH[name2col[out]]
        y_ori = y_ori.replace(2, 0)  # Convert 'no' (2) to 0
        idxs_missing_y = y_ori.isin([99, 95])
        
        idxs_missing = idxs_missing_x | idxs_missing_y
        X = X_ori[~idxs_missing]
        y = y_ori[~idxs_missing]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Optional: SMOTE for imbalanced data
        # sm = SMOTE(random_state=42)
        # X_train, y_train = sm.fit_resample(X_train, y_train)
        
        out2X_train[out] = X_train
        out2X_test[out] = X_test
        out2y_train[out] = y_train
        out2y_test[out] = y_test
        
        for model_type in model_types:
            print(f'  Training {model_type} for {out}')
            if model_type == 'LogisticRegression':
                model = LogisticRegression(max_iter=1000)
            elif model_type == 'XGBoost':
                model = XGBClassifier()
            elif model_type == 'RandomForest':
                model = RandomForestClassifier()
            elif model_type == 'GradientBoost':
                model = GradientBoostingClassifier()
            elif model_type == 'NaiveBayes':
                model = GaussianNB()
            elif model_type == 'LightGBM':
                model = lgb.LGBMClassifier()
                
            model.fit(X_train, y_train)
            out2model_type2model.setdefault(out, {})[model_type] = model
    
    # Save the models
    os.makedirs('new_model', exist_ok=True)
    for out in out2model_type2model:
        for model_type in out2model_type2model[out]:
            model = out2model_type2model[out][model_type]
            joblib.dump(model, f'new_model/{out}_{model_type}.pkl')
    
    # Save data splits
    joblib.dump(out2X_train, 'new_model/out2X_train.pkl')
    joblib.dump(out2X_test, 'new_model/out2X_test.pkl')
    joblib.dump(out2y_train, 'new_model/out2y_train.pkl')
    joblib.dump(out2y_test, 'new_model/out2y_test.pkl')
    
    return out2model_type2model, out2X_train, out2X_test, out2y_train, out2y_test

def evaluate_models(out2model_type2model, out2X_test, out2y_test):
    """
    Evaluate model performance using ROC curves and AUC scores.
    """
    print("Step 1.3: Evaluating model performance...")
    
    model2out2auc = {}
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    
    for i, out in enumerate(outcomes):
        plt.subplot(2, 2, i + 1)
        X_test = out2X_test[out]
        y_test = out2y_test[out]
        
        for model_type in model_types:
            model = out2model_type2model[out][model_type]
            y_pred = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
            auc = roc_auc_score(y_test, y_pred)
            model2out2auc.setdefault(model_type, {})[out] = auc
            
            plt.plot(fpr, tpr, label=f"{model_type} (AUC={auc:.3f})", 
                     color=sns.color_palette()[model_types.index(model_type)])
            
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{out}')
        plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    plt.suptitle('ROC Curves for Mental Health Prediction Models')
    plt.savefig('model_roc_curves.png')
    plt.close()
    
    # Save AUC scores
    df = pd.DataFrame(model2out2auc)
    df = df.round(3)
    df.to_csv('model_auc.csv')
    
    return model2out2auc

def calculate_estimated_probabilities(out2model_type2model):
    """
    Calculate estimated probabilities for all demographic subgroups.
    """
    print("Step 1.4: Calculating estimated probabilities for demographic subgroups...")
    
    features = list(var2value2name.keys())
    features_col = [name2col[feature] for feature in features]
    
    # Get the dimensions of the joint distribution
    dimensions = tuple(len(var2value2name[feature]) for feature in features)
    print(f'Dimensions: {dimensions}')
    print(f'Total number of demographic subgroups: {np.prod(dimensions)}')
    
    # Calculate probabilities for each subgroup using the best model
    estimated_prob = np.zeros(dimensions)
    
    for idx, values in enumerate(itertools.product(*[var2value2name[feature].keys() for feature in features])):
        feature_dict = {feature_col: value for feature_col, value in zip(features_col, values)}
        # Using Depression model as an example
        prob = out2model_type2model['Depression']['LogisticRegression'].predict_proba(pd.DataFrame([feature_dict]))[0][1]
        estimated_prob[np.unravel_index(idx, dimensions)] = prob
    
    return estimated_prob

#################################################
# STEP 2: ACS DATA COLLECTION (COUNTY LEVEL)
#################################################

def collect_acs_demographic_data():
    """
    Collect demographic data from ACS for Washington state counties.
    """
    print("Step 2.1: Collecting ACS demographic data for Washington counties...")
    
    # Define variable lists for ACS data collection
    age_sex_var = ['B01001_006E', 'B01001_030E']
    age_sex_white_var = ['B01001H_006E', 'B01001H_021E']
    age_sex_black_var = ['B01001B_006E', 'B01001B_021E']
    age_sex_asian_var = ['B01001D_006E', 'B01001D_021E']
    age_sex_hispanic_var = ['B01001I_006E', 'B01001I_021E']
    
    target_var = age_sex_var + age_sex_white_var + age_sex_black_var + age_sex_asian_var + age_sex_hispanic_var
    target_var = ['NAME'] + target_var
    
    # Collect age, sex, and race data
    age_sex_race = c.acs5.state_county(
        fields=target_var,
        state_fips=us.states.WA.fips,
        county_fips="*",
        year=2022
    )
    age_sex_race_df = pd.DataFrame(age_sex_race)
    
    # Calculate "other" race populations
    age_sex_race_df['B01001O_006E'] = (
        age_sex_race_df['B01001_006E'] - 
        age_sex_race_df['B01001H_006E'] - 
        age_sex_race_df['B01001B_006E'] - 
        age_sex_race_df['B01001D_006E'] - 
        age_sex_race_df['B01001I_006E']
    )
    age_sex_race_df['B01001O_021E'] = (
        age_sex_race_df['B01001_030E'] - 
        age_sex_race_df['B01001H_021E'] - 
        age_sex_race_df['B01001B_021E'] - 
        age_sex_race_df['B01001D_021E'] - 
        age_sex_race_df['B01001I_021E']
    )
    
    # Add total population column
    age_sex_race_df['Total'] = age_sex_race_df['B01001_006E'] + age_sex_race_df['B01001_030E']
    
    # Collect employment and insurance data
    employ_insurance_var = ['NAME', 'B27011_002E', 'B27011_004E', 'B27011_007E', 'B27011_009E', 'B27011_012E']
    employ_insurance = c.acs5.state_county(
        fields=employ_insurance_var, 
        state_fips=us.states.WA.fips, 
        county_fips="*",
        year=2022
    )
    employ_insurance_df = pd.DataFrame(employ_insurance)
    
    # Collect language data
    language_var = ['NAME', 'B10054_001E', 'B10054_013E']
    language = c.acs5.state_county(
        fields=language_var, 
        state_fips=us.states.WA.fips, 
        county_fips="*",
        year=2022
    )
    language_df = pd.DataFrame(language)
    language_df['B10054_002E'] = language_df['B10054_001E'] - language_df['B10054_013E']
    language_df.loc[language_df['B10054_002E'] < 0, 'B10054_002E'] = 0
    language_df = language_df.rename(columns={'B10054_013E': 'B10054_003E'})
    
    # Collect education data
    education_var = ['NAME', 'B07009_001E', 'B07009_002E', 'B07009_003E', 'B07009_004E', 'B07009_005E', 'B07009_006E']
    education = c.acs5.state_county(
        fields=education_var, 
        state_fips=us.states.WA.fips, 
        county_fips="*",
        year=2022
    )
    education_df = pd.DataFrame(education)
    education_df['B07009_005E'] = education_df['B07009_005E'] + education_df['B07009_006E']
    education_df = education_df.drop('B07009_006E', axis=1)
    
    # Collect income data
    income_var = ['NAME', 'B17004_001E', 'B17004_002E', 'B17004_011E']
    income = c.acs5.state_county(
        fields=income_var, 
        state_fips=us.states.WA.fips, 
        county_fips="*",
        year=2022
    )
    income_df = pd.DataFrame(income)
    
    # Collect family structure data
    family_structure_var = ['NAME', 'B09005_001E', 'B09005_002E', 'B09005_003E', 'B09005_004E', 'B09005_005E']
    family_structure = c.acs5.state_county(
        fields=family_structure_var, 
        state_fips=us.states.WA.fips, 
        county_fips="*",
        year=2022
    )
    family_structure_df = pd.DataFrame(family_structure)
    family_structure_df['B09005_004E'] = family_structure_df['B09005_004E'] + family_structure_df['B09005_005E']
    family_structure_df = family_structure_df.drop('B09005_005E', axis=1)
    family_structure_df['B09005_005E'] = (
        family_structure_df['B09005_001E'] - 
        family_structure_df['B09005_002E'] - 
        family_structure_df['B09005_003E'] - 
        family_structure_df['B09005_004E']
    )
    family_structure_df['B09005_005E'] = family_structure_df['B09005_005E'].apply(lambda x: 0 if x < 0 else x)
    
    return (age_sex_race_df, employ_insurance_df, language_df, education_df, income_df, family_structure_df)

def calculate_marginal_distributions(age_sex_race_df, employ_insurance_df, language_df, education_df, income_df, family_structure_df):
    """
    Calculate marginal distributions from ACS data for each county.
    """
    print("Step 2.2: Calculating marginal distributions from ACS data...")
    
    # Create joint distribution for race, sex, age
    name2marginal_race_sex_age = {}
    for idx, row in age_sex_race_df.iterrows():
        marginal = {}
        for values in itertools.product(*[var2value2name[feature].keys() for feature in used_vars[0:2]]):
            row_name = 'B01001'
            row_name += {1: 'I_', 2: 'H_', 3: 'B_', 4: 'D_', 5: 'O_'}[values[0]]
            if values[1] == 1:
                row_name += '006E'  # Male 14-17
            elif values[1] == 2:
                row_name += '021E'  # Female 14-17
            
            marginal[values] = row[row_name]/row['Total'] if row['Total'] > 0 else 0
        name2marginal_race_sex_age[row['NAME']] = marginal

    # Create joint distribution for insurance and employment
    name2marginal_ins_emp = {}
    for idx, row in employ_insurance_df.iterrows():
        marginal = {}
        total = row['B27011_002E']
        if total > 0:
            marginal[(1, 1)] = row['B27011_004E'] / total  # Employed with insurance
            marginal[(1, 2)] = row['B27011_007E'] / total  # Employed without insurance
            marginal[(2, 1)] = row['B27011_009E'] / total  # Unemployed with insurance
            marginal[(2, 2)] = row['B27011_012E'] / total  # Unemployed without insurance
        else:
            marginal[(1, 1)] = 0
            marginal[(1, 2)] = 0
            marginal[(2, 1)] = 0
            marginal[(2, 2)] = 0
        name2marginal_ins_emp[row['NAME']] = marginal

    # Create marginal distribution for education
    name2marginal_edu = {}
    for idx, row in education_df.iterrows():
        marginal = {}
        total = row['B07009_001E']
        if total > 0:
            marginal[(1,)] = row['B07009_002E'] / total  # Less than high school
            marginal[(2,)] = row['B07009_003E'] / total  # High school or GED
            marginal[(3,)] = row['B07009_004E'] / total  # Some college or technical school
            marginal[(4,)] = row['B07009_005E'] / total  # College degree or higher
        else:
            marginal[(1,)] = 0
            marginal[(2,)] = 0
            marginal[(3,)] = 0
            marginal[(4,)] = 0
        name2marginal_edu[row['NAME']] = marginal

    # Create marginal distribution for language
    name2marginal_lang = {}
    for idx, row in language_df.iterrows():
        marginal = {}
        total = row['B10054_001E']
        if total > 0:
            marginal[(1,)] = row['B10054_002E'] / total  # Primarily English
            marginal[(2,)] = row['B10054_003E'] / total  # Others
        else:
            marginal[(1,)] = 0
            marginal[(2,)] = 0
        name2marginal_lang[row['NAME']] = marginal

    # Create marginal distribution for family structure
    name2marginal_fam_struct = {}
    for idx, row in family_structure_df.iterrows():
        marginal = {}
        total = row['B09005_001E']
        if total > 0:
            marginal[(1,)] = row['B09005_002E'] / total  # Two parents, currently married
            marginal[(2,)] = row['B09005_003E'] / total  # Two parents, not currently married
            marginal[(3,)] = row['B09005_004E'] / total  # Single parent
            marginal[(4,)] = row['B09005_005E'] / total  # Other family type
        else:
            marginal[(1,)] = 0
            marginal[(2,)] = 0
            marginal[(3,)] = 0
            marginal[(4,)] = 0
        name2marginal_fam_struct[row['NAME']] = marginal

    # Create marginal distribution for family income
    name2marginal_income = {}
    for idx, row in income_df.iterrows():
        marginal = {}
        total = row['B17004_001E']
        if total > 0:
            marginal[(1,)] = row['B17004_011E'] / total  # Below poverty line
            marginal[(2,)] = row['B17004_002E'] / total  # Above poverty line
        else:
            marginal[(1,)] = 0
            marginal[(2,)] = 0
        name2marginal_income[row['NAME']] = marginal

    return (name2marginal_race_sex_age, name2marginal_ins_emp, name2marginal_edu, 
            name2marginal_lang, name2marginal_fam_struct, name2marginal_income)

def load_hys_data():
    """
    Load Healthy Youth Survey (HYS) data for depression among 10th graders.
    """
    print("Step 2.3: Loading Healthy Youth Survey (HYS) data...")
    
    # Import HYS data
    hys_depression_10th = pd.read_csv('HYS/depression_10th_grade_2018.csv')
    
    # Format county names to match ACS data
    hys_depression_10th['County'] = hys_depression_10th['County'] + ' County, Washington'
    
    return hys_depression_10th

#################################################
# STEP 3: OPTIMIZATION MODEL FOR JOINT DISTRIBUTION
#################################################

def optimize_demographic_proportions(Y_target, dimensions, estimated_prob, lambda_param, 
                                     marginal_race_sex_age, marginal_ins_emp, marginal_edu, 
                                     marginal_lang, marginal_fam_struct, marginal_income):
    """
    Optimize joint demographic distribution using marginal constraints from ACS.
    """
    model = gp.Model("DemographicProportions")
    
    # Create variables for each element in the joint distribution tensor
    X_d = model.addVars(*dimensions, vtype=GRB.CONTINUOUS, name="X_d")
    
    # Objective: minimize squared difference between estimated and target prevalence with regularization
    objective = (gp.quicksum(X_d[i] * estimated_prob[i] for i in np.ndindex(dimensions)) - Y_target) ** 2
    penalty = lambda_param * gp.quicksum(X_d[i] ** 2 for i in np.ndindex(dimensions))
    model.setObjective(objective + penalty, GRB.MINIMIZE)

    # Add constraints for race-sex-age joint distribution
    idxs = [0, 1]
    feature_list = [used_vars[idx] for idx in idxs]
    for values in itertools.product(*[var2value2name[feature].keys() for feature in feature_list]):
        category_sum = gp.quicksum(
            X_d[i] for i in np.ndindex(dimensions) 
            if all(i[idx] == value - 1 for idx, value in zip(idxs, values))
        )
        model.addConstr(category_sum >= marginal_race_sex_age[values], 
                        name=f"marginal_race_sex_age_{values}")

    # Add constraints for education
    idxs = [2]
    feature_list = [used_vars[idx] for idx in idxs]
    for values in itertools.product(*[var2value2name[feature].keys() for feature in feature_list]):
        category_sum = gp.quicksum(
            X_d[i] for i in np.ndindex(dimensions) 
            if all(i[idx] == value - 1 for idx, value in zip(idxs, values))
        )
        model.addConstr(category_sum >= marginal_edu[values], 
                        name=f"marginal_edu_{values}")

    # Add constraints for language
    idxs = [3]
    feature_list = [used_vars[idx] for idx in idxs]
    for values in itertools.product(*[var2value2name[feature].keys() for feature in feature_list]):
        category_sum = gp.quicksum(
            X_d[i] for i in np.ndindex(dimensions) 
            if all(i[idx] == value - 1 for idx, value in zip(idxs, values))
        )
        model.addConstr(category_sum >= marginal_lang[values], 
                       name=f"marginal_lang_{values}")

    # Add constraints for employment and insurance
    idxs = [4, 5]
    feature_list = [used_vars[idx] for idx in idxs]
    for values in itertools.product(*[var2value2name[feature].keys() for feature in feature_list]):
        category_sum = gp.quicksum(
            X_d[i] for i in np.ndindex(dimensions) 
            if all(i[idx] == value - 1 for idx, value in zip(idxs, values))
        )
        model.addConstr(category_sum >= marginal_ins_emp[values], 
                       name=f"marginal_ins_emp_{values}")

    # Add constraints for family income
    idxs = [6]
    feature_list = [used_vars[idx] for idx in idxs]
    for values in itertools.product(*[var2value2name[feature].keys() for feature in feature_list]):
        category_sum = gp.quicksum(
            X_d[i] for i in np.ndindex(dimensions) 
            if all(i[idx] == value - 1 for idx, value in zip(idxs, values))
        )
        model.addConstr(category_sum >= marginal_income[values], 
                       name=f"marginal_income_{values}")

    # Add constraints for family structure
    idxs = [7]
    feature_list = [used_vars[idx] for idx in idxs]
    for values in itertools.product(*[var2value2name[feature].keys() for feature in feature_list]):
        category_sum = gp.quicksum(
            X_d[i] for i in np.ndindex(dimensions) 
            if all(i[idx] == value - 1 for idx, value in zip(idxs, values))
        )
        model.addConstr(category_sum >= marginal_fam_struct[values], 
                       name=f"marginal_fam_struct_{values}")

    # Normalization constraint: probabilities sum to 1
    model.addConstr(gp.quicksum(X_d[i] for i in np.ndindex(dimensions)) == 1, 
                   name="normalization")

    # Optimize the model
    model.optimize()

    # Retrieve the optimized proportions
    optimized_proportions = np.zeros(dimensions)
    for i in np.ndindex(dimensions):
        optimized_proportions[i] = X_d[i].X

    return optimized_proportions

def run_optimization_for_all_counties(age_sex_race_df, estimated_prob, marginal_distributions, hys_depression_10th):
    """
    Run optimization for all counties to estimate depression prevalence.
    """
    print("Step 3.1: Running optimization for all counties...")
    
    # Unpack marginal distributions
    (name2marginal_race_sex_age, name2marginal_ins_emp, name2marginal_edu, 
     name2marginal_lang, name2marginal_fam_struct, name2marginal_income) = marginal_distributions
    
    # Prepare HYS data
    depression_10th = pd.DataFrame(age_sex_race_df['NAME'].values, columns=['County'])
    depression_10th = depression_10th.merge(hys_depression_10th[['County', 'Percentage']], on='County', how='left')
    depression_10th['Percentage'] = depression_10th['Percentage'].fillna('40.0%')
    depression_10th['Percentage'] = depression_10th['Percentage'].replace('%', '', regex=True).astype(float)/100
    depression_10th = depression_10th[['County', 'Percentage']]
    
    # Set optimization parameters
    lambda_param = 1
    dimensions = estimated_prob.shape
    
    # Disable Gurobi output
    gp.setParam('OutputFlag', 0)
    
    # Create directory for results
    os.makedirs('proportions_depression_14_17', exist_ok=True)
    
    # Run optimization for each county
    for idx, row in age_sex_race_df.iterrows():
        county_name = row['NAME']
        print(f'Solving for {county_name}')
        
        # Get target depression rate from HYS data (multiplied by 0.5 as in original code)
        Y_target = depression_10th[depression_10th['County'] == county_name]['Percentage'].values[0] * 0.5
        
        # Get marginal distributions for this county
        marginal_race_sex_age = name2marginal_race_sex_age[county_name]
        marginal_ins_emp = name2marginal_ins_emp[county_name]
        marginal_edu = name2marginal_edu[county_name]
        marginal_lang = name2marginal_lang[county_name]
        marginal_fam_struct = name2marginal_fam_struct[county_name]
        marginal_income = name2marginal_income[county_name]
        
        # Run optimization
        optimized_proportions = optimize_demographic_proportions(
            Y_target, dimensions, estimated_prob, lambda_param,
            marginal_race_sex_age, marginal_ins_emp, marginal_edu,
            marginal_lang, marginal_fam_struct, marginal_income
        )
        
        # Save the optimized proportions
        with open(f'proportions_depression_14_17/{county_name}.pkl', 'wb') as f:
            pickle.dump(optimized_proportions, f)
    
    return depression_10th


#################################################
# STEP 4: MAP VISUALIZATION
#################################################

def calculate_estimated_prevalence(depression_10th, estimated_prob):
    """
    Calculate estimated depression prevalence using optimized demographic proportions.
    """
    print("Step 4.1: Calculating estimated depression prevalence for each county...")
    
    # Add column for estimated prevalence
    depression_10th['Estimated'] = np.nan
    
    # Iterate over counties and calculate estimated prevalence
    for idx, row in depression_10th.iterrows():
        county_name = row['County']
        
        # Load optimized proportions
        with open(f'proportions_depression_14_17/{county_name}.pkl', 'rb') as f:
            optimized_proportions = pickle.load(f)
        
        print(f'Estimating for {county_name}, Total proportion: {np.sum(optimized_proportions)}')
        
        # Calculate estimated prevalence as weighted sum
        est_prev = np.sum(optimized_proportions * estimated_prob)
        depression_10th.at[idx, 'Estimated'] = est_prev
    
    # Clean up county names for display
    depression_10th['County'] = depression_10th['County'].str.replace(' County, Washington', '')
    
    # Adjust reported rates to match original code (multiplied by 0.5)
    depression_10th['Percentage'] = depression_10th['Percentage'] * 0.5
    
    return depression_10th

def evaluate_model_performance(depression_10th):
    """
    Evaluate model performance by comparing estimated vs. reported prevalence.
    """
    print("Step 4.2: Evaluating model performance...")
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs(depression_10th['Percentage'] - depression_10th['Estimated']) / depression_10th['Percentage'])
    print(f'MAPE: {mape:.4f}')
    
    # Calculate Weighted Absolute Percentage Error (WAPE)
    wape = np.sum(np.abs(depression_10th['Percentage'] - depression_10th['Estimated'])) / np.sum(depression_10th['Percentage'])
    print(f'WAPE: {wape:.4f}')
    
    # Calculate Mean Percentage Error (MPE)
    mpe = np.mean((depression_10th['Percentage'] - depression_10th['Estimated']) / depression_10th['Percentage'])
    print(f'MPE: {mpe:.4f}')
    
    return mape, wape, mpe

def create_bar_chart(depression_10th):
    """
    Create bar chart comparing estimated vs. reported depression rates.
    """
    print("Step 4.3: Creating bar chart comparison...")
    
    plt.figure(figsize=(15, 10))
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    
    # Create side-by-side bars
    barWidth = 0.35
    r1 = np.arange(len(depression_10th))
    r2 = [x + barWidth for x in r1]
    
    plt.bar(r1, depression_10th['Percentage'], color='beige', width=barWidth, 
            edgecolor='grey', label='Adjusted HYS', alpha=0.7)
    plt.bar(r2, depression_10th['Estimated'], color='plum', width=barWidth, 
            edgecolor='grey', label='Model Estimation', alpha=0.7)
    
    # Add statewide average reference line
    plt.axhline(y=0.2, color='r', linestyle='--', label='Statewide from HYS')
    
    # Format chart
    plt.xlabel('County', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(depression_10th))], 
              depression_10th['County'], rotation=90)
    plt.ylabel('Depression Rate')
    plt.title('Comparisons of Reported and Estimated Depression Rates for Children Aged 14-17')
    plt.legend()
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('depression_comparison_chart.png')
    plt.close()

def create_choropleth_map(depression_10th):
    """
    Create choropleth map of estimated depression prevalence.
    """
    print("Step 4.4: Creating choropleth map...")
    
    # Access shapefile of Washington state counties
    wa_tract = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip")
    
    # Reproject shapefile to UTM Zone 10N (EPSG: 32610) for Washington state
    wa_tract = wa_tract.to_crs(epsg=32610)
    
    # Keep only Washington state counties
    wa_tract = wa_tract[wa_tract['STATEFP'] == '53']
    
    # Prepare county data
    wa_df = depression_10th.rename(columns={'County': 'NAME'})
    
    # Merge county data with geodataframe
    wa_merge = wa_tract.merge(wa_df, on="NAME", how="left")
    gplt = wa_merge[["STATEFP", "COUNTYFP", "NAME", "geometry", "Percentage", "Estimated"]]
    
    # Create map
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=300)
    gplt.plot(column="Estimated", ax=ax, cmap='OrRd', legend=True)
    
    # Add county boundaries
    gplt.boundary.plot(ax=ax, color='black', linewidth=1)
    
    # Add county labels
    for x, y, label in zip(gplt.geometry.centroid.x, gplt.geometry.centroid.y, gplt['NAME']):
        plt.text(x, y, label, fontsize=8, ha='center', color='black')
    
    # Set title
    ax.set_title('Estimated Prevalence of Depression in Washington with Children Aged 14-17', 
                fontdict={'fontsize': '25', 'fontweight': '3'})
    
    # Save map
    plt.savefig('depression_prevalence_map.png')
    plt.close()


#################################################
# MAIN EXECUTION FUNCTION
#################################################

def main():
    """
    Main execution function to run the entire analysis pipeline.
    """
    print("Starting Mental Health Prevalence Analysis for Washington State...")
    
    # Step 1: Machine Learning Prediction Model
    print("\n===== STEP 1: MACHINE LEARNING PREDICTION MODEL =====")
    
    # Step 1.1-1.2: Prepare data and train models
    NSCH = prepare_nsch_data()
    out2model_type2model, out2X_train, out2X_test, out2y_train, out2y_test = train_mental_health_models(NSCH)
    
    # Step 1.3: Evaluate models
    model2out2auc = evaluate_models(out2model_type2model, out2X_test, out2y_test)
    
    # Step 1.4: Calculate estimated probabilities
    estimated_prob = calculate_estimated_probabilities(out2model_type2model)
    
    # Step 2: ACS Data Collection
    print("\n===== STEP 2: ACS DATA COLLECTION =====")
    
    # Step 2.1: Collect ACS demographic data
    acs_data = collect_acs_demographic_data()
    age_sex_race_df, employ_insurance_df, language_df, education_df, income_df, family_structure_df = acs_data
    
    # Step 2.2: Calculate marginal distributions
    marginal_distributions = calculate_marginal_distributions(
        age_sex_race_df, employ_insurance_df, language_df, 
        education_df, income_df, family_structure_df
    )
    
    # Step 2.3: Load HYS data
    hys_depression_10th = load_hys_data()
    
    # Step 3: Optimization Model
    print("\n===== STEP 3: OPTIMIZATION MODEL =====")
    
    # Step 3.1: Run optimization for all counties
    depression_10th = run_optimization_for_all_counties(
        age_sex_race_df, estimated_prob, marginal_distributions, hys_depression_10th
    )
    
    # Step 4: Map Visualization
    print("\n===== STEP 4: MAP VISUALIZATION =====")
    
    # Step 4.1: Calculate estimated prevalence
    depression_10th = calculate_estimated_prevalence(depression_10th, estimated_prob)
    
    # Step 4.2: Evaluate model performance
    mape, wape, mpe = evaluate_model_performance(depression_10th)
    
    # Step 4.3: Create bar chart
    create_bar_chart(depression_10th)
    
    # Step 4.4: Create choropleth map
    create_choropleth_map(depression_10th)
    
    print("\nAnalysis complete! Results saved to output files.")

if __name__ == "__main__":
    main()