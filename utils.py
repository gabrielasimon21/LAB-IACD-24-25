import statistics
import pylidc as pl
import matplotlib.pyplot as plt
from pylidc.utils import consensus
import SimpleITK as sitk
#import radiomics as featureextractor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

def get_features():
    feature_extractor = featureextractor.RadiomicsFeatureExtractor()
    extra_annotations = pl.annotation_feature_names
    annotated_scans = pl.query(pl.Scan).filter(pl.Scan.annotations.any()).all()
    feature_data = []
    nodule_count = 1

    for scan_data in annotated_scans:
        patient_id = scan_data.patient_id
        nodules_group = scan_data.cluster_annotations()

        for annotation_set in nodules_group:
            if annotation_set:
                consensus_mask, _, _ = pl.utils.consensus(annotation_set, clevel=0.5, pad=[(20, 20), (20, 20), (0, 0)])
                image_data = sitk.GetImageFromArray(consensus_mask.astype(float))
                radiomic_features = feature_extractor.execute(image_data, image_data, label=1)
                radiomic_features['Patient_ID'] = patient_id
                radiomic_features['Nodule_ID'] = f'Nodule_{nodule_count}'
                nodule_count += 1

                def compute_value(values):
                    try:
                        return statistics.mode(values)
                    except statistics.StatisticsError:
                        return np.mean(values)

                def compute_average(values):
                    return np.mean(values)

                subtlety_metric = compute_value([annotation.subtlety for annotation in annotation_set])
                structure_metric = compute_value([annotation.internalStructure for annotation in annotation_set])
                calcification_metric = compute_value([annotation.calcification for annotation in annotation_set])
                shape_metric = compute_value([annotation.sphericity for annotation in annotation_set])
                margin_metric = compute_value([annotation.margin for annotation in annotation_set])
                lobulation_metric = compute_value([annotation.lobulation for annotation in annotation_set])
                spiculation_metric = compute_value([annotation.spiculation for annotation in annotation_set])
                texture_metric = compute_value([annotation.texture for annotation in annotation_set])
                malignancy_metric = compute_average([annotation.malignancy for annotation in annotation_set])

                for _ in extra_annotations:
                    radiomic_features['subtlety'] = subtlety_metric
                    radiomic_features['internal_structure'] = structure_metric
                    radiomic_features['sphericity'] = shape_metric
                    radiomic_features['margin'] = margin_metric
                    radiomic_features['lobulation'] = lobulation_metric
                    radiomic_features['spiculation'] = spiculation_metric
                    radiomic_features['texture'] = texture_metric
                    radiomic_features['malignancy'] = malignancy_metric

                feature_data.append(radiomic_features)

    feature_df = pd.DataFrame(feature_data)
    return feature_df


def pre_processing(input_df):
    unique_count = input_df.nunique()
    single_value_columns = unique_count[unique_count == 1].index
    filtered_df = input_df.drop(columns=single_value_columns)
    numeric_df = filtered_df.select_dtypes(include=[int, float])
    return numeric_df


def normalize_dataframe(input_df, target_col):
    data_features = input_df.drop(columns=[target_col])
    data_target = input_df[target_col].round().astype(int)
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data_features), columns=data_features.columns)
    final_normalized_df = pd.concat([normalized_data, data_target], axis=1)
    final_normalized_df = final_normalized_df[final_normalized_df['malignancy'] != 3]
    final_normalized_df.loc[:, "malignancy"] = final_normalized_df["malignancy"].apply(lambda x: 1 if x > 3 else 0)
    return final_normalized_df


def rename_columns(input_df):
    renamed_columns = {}
    for column_name in input_df.columns:
        simplified_name = column_name.split('_')[-1]
        renamed_columns[column_name] = simplified_name
    return input_df.rename(columns=renamed_columns)


def prepare_data(input_df, target_col, test_size=0.2):
    features = input_df.drop(columns=[target_col])
    target = input_df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, X_test, y_train_balanced, y_test

def load_models(file_path):
    # Loads models from JSON file
    with open(file_path, 'r') as f:
        return json.load(f)
    
def train_model(model_info, X_train, y_train):
    # Trains models
    model_class = globals()[model_info['model']] 
    clf = model_class(**model_info['params']) 
    clf.fit(X_train, y_train)  
    return clf

def plot_confusion_matrix(clf, X_test, y_test, model_name):

    grid_predictions = clf.predict(X_test)  # Makes predictions on the test data
    cm = confusion_matrix(y_test, grid_predictions)  # Generates confusion matrix

    # Displays the confusion matrix
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Malignous', 'Malignous'])
    disp.plot() 
    plt.title(model_name)
    plt.show()

def evaluate_models(models, X_train, y_train, X_test, y_test):

    accuracies = []

    for model_name, mp in models.items():
        if 'params' in mp:
            clf = train_model(mp, X_train, y_train)  # Trains the model
            plot_confusion_matrix(clf, X_test, y_test, model_name)  # Plots confusion matrix

            # Calculates accuracy
            cm = confusion_matrix(y_test, clf.predict(X_test))
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()  
            accuracies.append((model_name, accuracy))

    return accuracies

def plot_accuracy_comparison(accuracies):

    accuracy_df = pd.DataFrame(accuracies, columns=['Model', 'Accuracy'])
    accuracy_df['Accuracy'] = accuracy_df['Accuracy'] * 100  # Convert to percentage

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Model', y='Accuracy', data=accuracy_df)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accurac (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)  

    plt.ylim(80, 100)  
    
    plt.gca().set_yticklabels([f'{int(y)}%' for y in plt.gca().get_yticks()])

    plt.show()
