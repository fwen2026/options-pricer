import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

from src.training.preprocess_data import load_data, compute_normalization_params, normalize_X, normalize_y
from src.constants import ELASTIC_MODEL_PATH, TRAIN_DATA_PATH


def train_model(model, X_train, y_train):
    """
    Training script for a sklearn model. Has hyperparameter tuning.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(estimator=model, param_grid={
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }, cv=cv)

    grid.fit(X_train, y_train)
    print("Hyperparameters: " + str(grid.best_params_))
    return grid.best_estimator_


def predict(model, X_test):
    """
    Prediction script for a sklearn model
    """
    return model.predict(X_test)


if __name__ == "__main__":
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    norm_params = compute_normalization_params(X_train, y_train)

    X_norm = normalize_X(X_train, norm_params)
    y_norm = normalize_y(y_train, norm_params)

    model = ElasticNet(max_iter=10000)

    model = train_model(model, X_norm, y_norm)

    # Save model
    joblib.dump({
        'model': model,
        'norm_params': norm_params
    }, ELASTIC_MODEL_PATH)