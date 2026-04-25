import joblib
import xgboost as xgb

from src.training.preprocess_data import compute_normalization_params, load_data, normalize_y
from src.constants import EVAL_DATA_PATH, TRAIN_DATA_PATH, XGBOOST_MODEL_PATH

def train_model(model, X_train, y_train, X_eval, y_eval):
    model.fit(X_train, 
              y_train,
              eval_set=[(X_eval, y_eval)],
              verbose=50
              )

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    X_eval, y_eval = load_data(EVAL_DATA_PATH)
    
    norm_params = compute_normalization_params(X_train, y_train)
    y_train = normalize_y(y_train, norm_params)
    y_eval = normalize_y(y_eval, norm_params)

    # Train model
    model = xgb.XGBRegressor(
        booster = "gbtree",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        objective="reg:squarederror",
        random_state=42
    )
    train_model(model, X_train, y_train, X_eval, y_eval)

    # Save model
    model.save_model(XGBOOST_MODEL_PATH)
    joblib.dump(norm_params, XGBOOST_MODEL_PATH.replace(".ubj", "_norm_params.joblib"))