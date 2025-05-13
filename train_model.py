import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


# Loading data
df = pd.read_csv('Dataset/dataset.csv')

# Defining features
categorical_features = ['Building Typology']
numerical_features_damage = ['Level of Inundation(m)', 'Hours of Inundation(hrs)', 'Combined effect of level and hour of inundation']

# Damage pipeline
preprocessor_damage = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_damage),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

damage_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_damage),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Separating features and target
X_damage = df[categorical_features + numerical_features_damage]
y_damage = df['Ratio of Damage(%)']

# Train-test split
X1_train, X1_test, y1_train, y1_test = train_test_split(X_damage, y_damage, test_size=0.2, random_state=42)

# Training the damage model
damage_pipeline.fit(X1_train, y1_train)

# Evaluating the damage model on test set
y1_pred = damage_pipeline.predict(X1_test)
damage_r2 = r2_score(y1_test, y1_pred)
print(f"📊 R² score for Damage Prediction Model: {damage_r2:.4f}")

# Loss pipeline
numerical_features_loss = ['Level of Inundation(m)', 'Hours of Inundation(hrs)', 'Combined effect of level and hour of inundation', 'Ratio of Damage(%)']

preprocessor_loss = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_loss),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

loss_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_loss),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Preparing loss model data
X_loss = X_damage.copy()
X_loss['Ratio of Damage(%)'] = df['Ratio of Damage(%)']
y_loss = df['Loss(NPR)']
X2_train, X2_test, y2_train, y2_test = train_test_split(X_loss, y_loss, test_size=0.2, random_state=42)

# Training the loss model
loss_pipeline.fit(X2_train, y2_train)

# Evaluating the loss model on test set
y2_pred = loss_pipeline.predict(X2_test)
loss_r2 = r2_score(y2_test, y2_pred)
print(f"📊 R² score for Loss Prediction Model: {loss_r2:.4f}")

# Final training on the entire dataset for both models
damage_pipeline.fit(X_damage, y_damage)
loss_pipeline.fit(X_loss, y_loss)

# Saving models
joblib.dump(damage_pipeline, 'models/damage_pipeline_v1.pkl')
joblib.dump(loss_pipeline, 'models/loss_pipeline_v1.pkl')

# Saving model evaluation metrics
metrics = {
    'damage_r2': damage_r2,
    'loss_r2': loss_r2
}
joblib.dump(metrics, 'models/model_metrics_v1.pkl')

#Saving typologies
typologies = df['Building Typology'].unique().tolist()
joblib.dump(typologies, 'models/typologies_v1.pkl')

print("✅ Models trained and saved.")
