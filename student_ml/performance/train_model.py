import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_nn():
    # Adjusted path for running from current dir
    data_path = 'Student_Performance.csv' if os.path.exists('Student_Performance.csv') else 'student_ml/Student_Performance.csv'
    df = pd.read_csv(data_path)  

    # Preprocessing
    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])

    # Feature Engineering
    # 1. Study-Sleep Interaction (Studying is more effective with sleep)
    df['Study_Sleep_Interaction'] = df['Hours Studied'] * df['Sleep Hours']
    
    # 2. Study Efficiency (High previous scores + Study hours)
    df['Study_Efficiency'] = (df['Hours Studied'] * df['Previous Scores']) / 100
    
    # 3. Effort Consistency (Sample papers * Study hours)
    df['Effort_Consistency'] = df['Hours Studied'] * df['Sample Question Papers Practiced']

    X = df.drop('Performance Index', axis=1).values
    y = df['Performance Index'].values

    # Scale features 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feed-forward neural network
    model = keras.Sequential([
        layers.Input(shape=(X.shape[1],)),          
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)                               
    ])

    model.compile(
        optimizer='adam',
        loss='mse',                                   
        metrics=['mae']
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,                
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ]
    )

    # Evaluate
    y_pred = model.predict(X_test).flatten()
    print("\nFinal evaluation:")
    print(f"R² score:     {r2_score(y_test, y_pred):.4f}")
    print(f"MAE:          {mean_absolute_error(y_test, y_pred):.4f}")

    # Saving model and scaler 
    save_dir = 'performance' if os.path.exists('performance') else 'student_ml/performance'
    model.save(f'{save_dir}/nn_model.keras')
    joblib.dump(scaler, f'{save_dir}/scaler.pkl')

    print(f"Neural network model and scaler saved to {save_dir}!")

if __name__ == "__main__":
    train_nn()