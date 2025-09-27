'''
This file:
 - Contains unit tests for the *churn_library.py* functions.
 - Logs any errors and INFO messages.
'''
import logging
from pathlib import Path
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(f"Dataframe contains information and of size {df.shape}")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, in_df):
    '''
    test perform eda function
    '''
    value_counts_pth = Path("./data/marital_status_val_count.png")

    perform_eda(in_df)

    try:
        assert "Churn" in in_df.columns
        assert in_df["Churn"].dtype == "int64"
        logging.info(
            f"Dataframe has a column Chrun and of type {
                in_df['Churn'].dtype}")

    except AssertionError as err:
        logging.error("Incorrect column name or data type isn't int")
        raise err

    try:
        assert value_counts_pth.exists()
        assert value_counts_pth.stat().st_size > 0
        logging.info(f"Value counts plot exists at {
                     value_counts_pth} and not empty")
    except AssertionError:
        logging.error(f"ERROR: Value counts at {
                      value_counts_pth} doesn't exist OR empty")

    return in_df


def test_encoder_helper(encoder_helper, df, category_lst):
    '''
    test encoder helper
    '''
    encoded_df = encoder_helper(df, category_lst, "Churn")

    for column_name in category_lst:
        try:
            assert column_name + "_Churn" in encoded_df.columns
            logging.info(
                f"SUCCESS: Column {
                    column_name +
                    '_Churn'} exists in the encoded dataframe")
        except AssertionError as err:
            logging.error(
                f"Column {
                    column_name +
                    '_Churn'} doesn't exist in the encoded dataframe")
            raise err
    return encoded_df


def test_perform_feature_engineering(
        perform_feature_engineering, e_df, k_columns):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        e_df, k_columns)
    try:
        assert len(x_train) + len(x_test) == len(e_df)
        assert len(y_train) + len(y_test) == len(e_df)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert list(x_train.columns) == k_columns
        logging.info("Successful split of selected features in dataset")

    except AssertionError as err:
        logging.error(f"EORROR is encoubtered -> {err}")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(x_train, x_test, y_train, y_test)
    roc_path = Path("./data/roc_curves.png")
    rf_path = Path("./models/rfc_model.pkl")
    lr_path = Path("./models/logistic_model.pkl")

    output_list = [roc_path, rf_path, lr_path]

    for output in output_list:
        try:
            assert output.exists()
            assert output.stat().st_size > 0
            logging.info(f"File exists at {output} and not empty")
        except AssertionError:
            logging.error(f"ERROR: File at {output} doesn't exist OR empty")


def test_generate_predictions(
        generate_predictions,
        cv_rfc_pth,
        lrc_pth,
        x_train,
        x_test):
    '''
    Tests that model is correctly loaded and tested
    '''
    (train_preds_rf, test_preds_rf, train_preds_lr,
     test_preds_lr, rf_model_loaded, lr_model_loaded) = generate_predictions(
        cv_rfc_pth, lrc_pth, x_train, x_test)

    try:
        assert len(train_preds_rf) == len(train_preds_lr)
        assert len(test_preds_rf) == len(test_preds_lr)
        assert len(train_preds_rf) > 0
        assert len(test_preds_rf) > 0
        logging.info("Two models generated predections")
    except AssertionError:
        logging.error("ERROR: Predections generated empty or size issue")

    return (train_preds_rf, test_preds_rf, train_preds_lr,
            test_preds_lr, rf_model_loaded, lr_model_loaded)


def test_classification_report_image(
        classification_report_image,
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    '''
    Test that the two classification reports are generated and saved
    '''
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    class_report_lr_pth = Path("./data/classification_report_lr.png")
    class_report_rf_pth = Path("./data/classification_report_rf.png")
    output_list = [class_report_lr_pth, class_report_rf_pth]

    for output in output_list:
        try:
            assert output.exists()
            assert output.stat().st_size > 0
            logging.info(f"Classification report exists at {
                         output} and not empty")
        except AssertionError:
            logging.error(f"ERROR: Classification report at {
                          output} doesn't exist OR empty")


def test_feature_importance_plot(
        feature_importance_plot,
        model,
        x_data,
        output_pth):
    '''
    Test that the feature_importance_plot is generated and saved
    '''
    feature_importance_plot(model, x_data, output_pth)
    output_pth = Path(output_pth)

    try:
        assert output_pth.exists()
        assert output_pth.stat().st_size > 0
        logging.info(f"Features importance plot exists at {
                     output_pth} and not empty")
    except AssertionError:
        logging.error(f"ERROR: Features importance plot at {
                      output_pth} doesn't exist OR empty")


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

    raw_data = test_import(cl.import_data)
    churn_df = test_eda(cl.perform_eda, raw_data)
    encoded_out_df = test_encoder_helper(
        cl.encoder_helper, churn_df, cat_columns)
    in_train, in_test, out_train, out_test = test_perform_feature_engineering(
        cl.perform_feature_engineering,
        encoded_out_df,
        keep_cols
    )
    # test_train_models(cl.train_models, in_train, in_test, out_train, out_test)
    out = test_generate_predictions(
        cl.generate_predictions,
        RFC_MODEL_PTH,
        LR_MODEL_PTH,
        in_train,
        in_test)
    train_preds_rf_out = out[0]
    test_preds_rf_out = out[1]
    train_preds_lr_out = out[2]
    test_preds_lr_out = out[3]
    rf_model_out = out[4]
    lr_model_out = out[5]

    test_classification_report_image(cl.classification_report_image,
                                     out_train,
                                     out_test,
                                     train_preds_lr_out,
                                     train_preds_rf_out,
                                     test_preds_lr_out,
                                     test_preds_rf_out)
    test_feature_importance_plot(
        cl.feature_importance_plot,
        rf_model_out,
        in_train,
        FEATURE_IMPORTANCE_PLOT_PTH)
