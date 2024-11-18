from sklearn.preprocessing import MinMaxScaler

def create_features(X, y, numerical_columns, categorical_columns):
    X_transformed = X.copy()

    scaler = MinMaxScaler()
    X_transformed[numerical_columns] = scaler.fit_transform(X_transformed[numerical_columns])
    
    for col in categorical_columns:
        target_means = y.groupby(X[col]).mean()
        X_transformed[col] = X[col].map(target_means)
    
    return X_transformed


