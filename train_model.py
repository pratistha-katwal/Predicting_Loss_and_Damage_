import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib



# Load data
df = pd.read_csv('Dataset/dataset.csv')
df['Level of Inundation(m)'] = df['Level of Inundation(cm)'] / 100
df['hr_lv'] = df['Level of Inundation(m)'] * df['Hours of Inundation(hrs)']

# Define features
categorical_features = ['Building Typology']
numerical_features_damage = ['Level of Inundation(m)', 'Hours of Inundation(hrs)', 'hr_lv']

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

X_damage = df[categorical_features + numerical_features_damage]
y_damage = df['Ratio of Damage(%)']
X1_train, X1_test, y1_train, y1_test = train_test_split(X_damage, y_damage, test_size=0.2, random_state=42)
damage_pipeline.fit(X1_train, y1_train)

# Evaluate damage model
y1_pred = damage_pipeline.predict(X1_test)
damage_r2 = r2_score(y1_test, y1_pred)
print(f"📊 R² score for Damage Prediction Model: {damage_r2:.4f}")

# Loss pipeline
numerical_features_loss = ['Level of Inundation(m)', 'Hours of Inundation(hrs)', 'hr_lv', 'Ratio of Damage(%)']

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

X_loss = X_damage.copy()
X_loss['Ratio of Damage(%)'] = df['Ratio of Damage(%)']
y_loss = df['Loss(NPR)']
X2_train, X2_test, y2_train, y2_test = train_test_split(X_loss, y_loss, test_size=0.2, random_state=42)
loss_pipeline.fit(X2_train, y2_train)

# Evaluate loss model
y2_pred = loss_pipeline.predict(X2_test)
loss_r2 = r2_score(y2_test, y2_pred)
print(f"📊 R² score for Loss Prediction Model: {loss_r2:.4f}")

# Save models
joblib.dump(damage_pipeline, 'models/damage_pipeline.pkl')
joblib.dump(loss_pipeline, 'models/loss_pipeline.pkl')

print("✅ Models trained and saved.")

metrics = {
    'damage_r2': damage_r2,
    'loss_r2': loss_r2
}
joblib.dump(metrics, 'models/model_metrics.pkl')

