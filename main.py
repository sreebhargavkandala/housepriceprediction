import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def feature_engineering(data):
    data['LivingAreaPerRoom'] = data['squareMeters'] / data['numberOfRooms']
    features = [
        'squareMeters', 'numberOfRooms', 'hasYard', 'hasPool', 'floors', 'cityCode', 'cityPartRange',
        'numPrevOwners', 'made', 'isNewBuilt', 'hasStormProtector', 'basement', 'attic', 'garage',
        'hasStorageRoom', 'hasGuestRoom', 'LivingAreaPerRoom'
    ]
    return data[features], data['price']

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mae, r2, mape

if __name__ == '__main__':
    # Load and process data
    data = load_data('data/ParisHousing.csv')
    X, y = feature_engineering(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train and tune the model
    best_model, best_params = hyperparameter_tuning(X_train_scaled, y_train)
    
    # Evaluate the model
    mae, r2, mape = evaluate_model(best_model, X_test_scaled, y_test)
    print(f'MAE: {mae}')
    print(f'R2: {r2}')
    print(f'MAPE: {mape}')
    
    # Save the model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
