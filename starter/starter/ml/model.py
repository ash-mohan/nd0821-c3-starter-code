from sklearn.metrics import fbeta_score, precision_score, recall_score, \
    accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    accuracy = accuracy_score(y, preds)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta, accuracy


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : GradientBoostingClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    predictions = model.predict(X)

    return predictions


def slice_performance(feature, model, data, X, y):
    """ Test model performance on specific slice of data

    Inputs
    ------
    feature : str
        Feature to perform slice performance on
    model : GradientBoostingClassifier
        Trained machine learning model.
    data : pandas DataFrame
        Data frame to extract slices from
    X : np.array
        processed data used for prediction.
    y : np.array
        binary labels
    Returns
    -------
    None

    """

    df = data.reset_index(drop=True)

    with open("slice_output.txt", "w") as f:
        f.write(f"Computing model performance on slices of {feature} "
                f"feature\n\n")

        for val in df[feature].unique():
            f.write(f"Value for {feature}: {val}\n")

            slice = df.loc[df[feature] == val, :]
            X_slice = X[slice.index.values, :]
            y_slice = y[slice.index.values]

            y_pred = inference(model, X_slice)
            precision, recall, fbeta, accuracy = compute_model_metrics(
                y_slice, y_pred)

            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F-beta score: {fbeta}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write("-----------------\n")

    return None


def export_model_files(model, encoder, lb):
    """ Export model files and save to path.

    Inputs
    ------
    model : GradientBoostingClassifier
        Trained machine learning model.
    encoder : OneHotEncoder
        Encoder used for categorical variables
    lb: LabelBinarizer
        Label Binarizer used for labels
    Returns
    -------
    None
    """

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('encoders/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    with open('lb/lb.pkl', 'wb') as f:
        pickle.dump(lb, f)

    return None


def import_model_files(model_path, encoder_path, lb_path):
    """ Import model files.

    Inputs
    ------
    model_path : path
        Path to model.pkl file
    encoder_path : path
        Path to encoder.pkl used for categorical variables
    lb_path: path
        path to lb.pkl used for labels
    Returns
    -------
    model : GradientBoostingClassifier
        Trained machine learning model.
    encoder : OneHotEncoder
        Encoder used for categorical variables
    lb: LabelBinarizer
        Label Binarizer used for labels
    """

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    with open(lb_path, 'rb') as f:
        lb = pickle.load(f)

    return model, encoder, lb
