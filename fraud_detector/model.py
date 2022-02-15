from fraud_detector.data import load_data, preprocess_data
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from fraud_detector.utils import get_abs_path
import sklearn
import joblib


def train(
    data_path: str = "data/dataset_TakeHome.csv", model_dir: str = "models"
) -> sklearn.pipeline.Pipeline:
    """Trains a model on the data.

    Args:
        data_path (str): The name of the file to load. Should either be absolute or relative to the base repo.

    Returns:
        sklearn.pipeline.Pipeline: trained model
    """
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    # NOTE picked SGD based on sklearn algorithm cheet sheet
    # https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    model.fit(X_train, y_train)
    print(f"Accuracy: {model.score(X_test, y_test)}")
    model_save_path = get_abs_path(f"{model_dir}/model.joblib")
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    return model


if __name__ == "__main__":
    train()
