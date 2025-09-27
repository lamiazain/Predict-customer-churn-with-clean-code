"""
This file contains the necessary code to load the dataset,
preprocess it with feature engineering then training two
classification machine learning models (logestic
regression and random forest).
This file also contains the code to save the trained
models and generating predictions.
Also the necessary code to generate the classification reports for
the two models and the feature importance plot is provided
"""


# import libraries
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # avoid Qt GUI; must be set before pyplot import

# sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        "./data/marital_status_val_count.png",
        dpi=300,
        bbox_inches="tight")
    plt.close() 


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns
            that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for new features
    '''
    for column_name in category_lst:
        # compute mean response per category
        new_lst = []
        category_groups = df.groupby(column_name)[response].mean()
        #####################
        for val in df[column_name]:
            new_lst.append(category_groups.loc[val])
        df[column_name + "_Churn"] = new_lst

    return df


def perform_feature_engineering(df, response):
    '''
    It splits the data frame with selected features into train and test inputs and outputs
    input:
              df: pandas dataframe
              response: string of response name
             [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    new_df = pd.DataFrame()
    new_df = df[response]

    x_train, x_test, y_train, y_test = train_test_split(
        new_df, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='liblinear', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    # Training models
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Random Forest ROC
    rfc_disp = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)

    # Logistic Regression ROC
    lrc_disp = RocCurveDisplay.from_estimator(
        lrc, x_test, y_test, ax=ax, alpha=0.8)

    # save plot
    plt.savefig("./data/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()   # frees memory, avoids warnings

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


def generate_predictions(cv_rfc_pth, lrc_pth, x_train, x_test):
    '''
    Loads the saved models and generate predictions
    inputs:
        cv_rfc_pth: location of the saved random forest model
        lrc_pth: location of the saved logestic regression model
        X_train: Training model inputs
        X_test: Testing inputs
    outputs:
        train_preds_rf: Generated predictions of the random forest model given the training inputs
        test_preds_rf: Generated predictions of the random forest model given the test inputs
        train_preds_lr: Generated predictions of the logestic regression model given the
        training inputs
        test_preds_lr: Generated predictions of the logestic regression model given the tedt inputs
        cv_rfc_model: the saved random forest model
        lr_model: the saved logestic regression model

    '''

    rf_model_loaded = joblib.load(cv_rfc_pth)
    lr_model_loaded = joblib.load(lrc_pth)

    train_preds_rf = rf_model_loaded.predict(x_train)
    test_preds_rf = rf_model_loaded.predict(x_test)

    train_preds_lr = lr_model_loaded.predict(x_train)
    test_preds_lr = lr_model_loaded.predict(x_test)

    return (train_preds_rf, test_preds_rf, train_preds_lr,
            test_preds_lr, rf_model_loaded, lr_model_loaded)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        "./data/classification_report_rf.png",
        dpi=300,
        bbox_inches="tight")
    plt.close()

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        "./data/classification_report_lr.png",
        dpi=300,
        bbox_inches="tight")
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    RFC_MODEL_PTH = './models/rfc_model.pkl'
    LR_MODEL_PTH = './models/logistic_model.pkl'
    FEATURE_IMPORTANCE_PLOT_PTH = './data/feture_importance_plot.png'

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Load the dataset
    raw_df = import_data("./data/bank_data.csv")

    # Perform data analysis
    perform_eda(raw_df)

    # Encoding the dataframe
    encoded_df = encoder_helper(raw_df, cat_columns, 'Churn')

    # Perform feature engineering
    in_train, in_test, out_train, out_test = perform_feature_engineering(
        encoded_df, keep_cols)

    # Uncomment the following line to train the two models again
    train_models(in_train, in_test, out_train, out_test)

    # Generate predictions and save models
    out = generate_predictions(RFC_MODEL_PTH, LR_MODEL_PTH, in_train, in_test)

    train_preds_rf_out = out[0]
    test_preds_rf_out = out[1]
    train_preds_lr_out = out[2]
    test_preds_lr_out = out[3]
    rf_model_out = out[4]
    lr_model_out = out[5]

    # Generate the classification report for the two models
    classification_report_image(
        out_train,
        out_test,
        train_preds_lr_out,
        train_preds_rf_out,
        test_preds_lr_out,
        test_preds_rf_out)

    # Generate the feature importance plot for the random forest model
    feature_importance_plot(
        rf_model_out,
        in_train,
        FEATURE_IMPORTANCE_PLOT_PTH)
