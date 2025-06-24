import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv("data/processed/tesla_lagged_96.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df = df[['date'] + list(numeric_columns)]
    
    return df

def add_technical_indicators(df):
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close']).diff()
    
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = SMAIndicator(close=df['Close'], window=window).sma_indicator()
        df[f'ema_{window}'] = EMAIndicator(close=df['Close'], window=window).ema_indicator()
        df[f'ma_ratio_{window}'] = df['Close'] / df[f'sma_{window}']
    
    rsi = RSIIndicator(close=df['Close'])
    df['rsi'] = rsi.rsi()
    
    bb = BollingerBands(close=df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    
    for window in [5, 7]:
        df[f'sentiment_ma_{window}'] = df['avg_sentiment'].rolling(window=window).mean()
        df[f'sentiment_std_{window}'] = df['avg_sentiment'].rolling(window=window).std()
        df[f'sentiment_momentum_{window}'] = df['avg_sentiment'].diff(window)
        df[f'sentiment_volatility_{window}'] = df['avg_sentiment'].rolling(window=window).std() / df['avg_sentiment'].rolling(window=window).mean()
        df[f'sentiment_trend_{window}'] = df['avg_sentiment'].rolling(window=window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    df['target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    df['target_7d'] = df['Close'].shift(-7) / df['Close'] - 1
    df['target_direction'] = np.where(df['target_5d'] > 0, 1, 0)
    df['target_direction_7d'] = np.where(df['target_7d'] > 0, 1, 0)
    
    df = df.fillna(0)
    
    return df

def add_sentiment_features(df):
    df['sentiment_ma_5'] = df['avg_sentiment'].rolling(window=5).mean()
    df['sentiment_ma_7'] = df['avg_sentiment'].rolling(window=7).mean()
    
    df['sentiment_std_5'] = df['avg_sentiment'].rolling(window=5).std()
    df['sentiment_std_7'] = df['avg_sentiment'].rolling(window=7).std()
    
    df['sentiment_momentum_5'] = df['avg_sentiment'].diff(5)
    df['sentiment_momentum_7'] = df['avg_sentiment'].diff(7)
    
    df['sentiment_acceleration'] = df['sentiment_momentum_5'].diff()
    
    df['sentiment_mean_reversion_5'] = (df['avg_sentiment'] - df['sentiment_ma_5']) / df['sentiment_std_5']
    df['sentiment_mean_reversion_7'] = (df['avg_sentiment'] - df['sentiment_ma_7']) / df['sentiment_std_7']
    
    df['sentiment_trend_5'] = df['avg_sentiment'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['sentiment_trend_7'] = df['avg_sentiment'].rolling(window=7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    df['sentiment_volatility_ratio'] = df['sentiment_std_5'] / df['sentiment_ma_5']
    
    df['sentiment_momentum_ratio'] = df['sentiment_momentum_5'] / df['sentiment_ma_5']
    
    df['sentiment_price_impact_5'] = df['avg_sentiment'] * df['returns'].rolling(window=5).mean()
    df['sentiment_price_impact_7'] = df['avg_sentiment'] * df['returns'].rolling(window=7).mean()
    
    df['sentiment_volatility_impact_5'] = df['avg_sentiment'] * df['returns'].rolling(window=5).std()
    df['sentiment_volatility_impact_7'] = df['avg_sentiment'] * df['returns'].rolling(window=7).std()
    
    df['sentiment_trend_impact_5'] = df['avg_sentiment'] * df['Close'].pct_change(5)
    df['sentiment_trend_impact_7'] = df['avg_sentiment'] * df['Close'].pct_change(7)
    
    df = df.fillna(0)
    
    return df

def prepare_sequences(df):
    feature_columns = [col for col in df.columns if col not in ['Date', 'target_5d', 'target_7d', 'target_direction', 'target_direction_7d']]
    
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(np.int64) // 10**9
    
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    joblib.dump(scaler, 'feature_scaler.joblib')
    sequence_length = 10
    X, y5, y7, yd5, yd7 = [], [], [], [], []
    
    for i in range(len(df) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length)])
        y5.append(df['target_5d'].iloc[i + sequence_length])
        y7.append(df['target_7d'].iloc[i + sequence_length])
        yd5.append(df['target_direction'].iloc[i + sequence_length])
        yd7.append(df['target_direction_7d'].iloc[i + sequence_length])
    
    return np.array(X), np.array(y5), np.array(y7), np.array(yd5), np.array(yd7)

def build_model(sequence_length, n_features):
    inputs = Input(shape=(sequence_length, n_features))
    
    x1 = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)
    
    x1 = Bidirectional(LSTM(64, return_sequences=True))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)
    
    x1 = Bidirectional(LSTM(32))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)
    
    x2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dropout(0.3)(x2)
    
    combined = Concatenate()([x1, x2])
    
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    residual = x
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Add()([x, residual])
    
    x5 = Dense(128, activation='relu')(x)
    x5 = BatchNormalization()(x5)
    x5 = Dropout(0.3)(x5)
    output_5d = Dense(1, name='output_5d')(x5)
    
    x7 = Dense(128, activation='relu')(x)
    x7 = BatchNormalization()(x7)
    x7 = Dropout(0.3)(x7)
    output_7d = Dense(1, name='output_7d')(x7)
    
    xd = Dense(128, activation='relu')(x)
    xd = BatchNormalization()(xd)
    xd = Dropout(0.3)(xd)
    xd = Dense(64, activation='relu')(xd)
    xd = BatchNormalization()(xd)
    xd = Dropout(0.3)(xd)
    output_direction = Dense(1, activation='sigmoid', name='output_direction')(xd)

    xd7 = Dense(128, activation='relu')(x)
    xd7 = BatchNormalization()(xd7)
    xd7 = Dropout(0.3)(xd7)
    xd7 = Dense(64, activation='relu')(xd7)
    xd7 = BatchNormalization()(xd7)
    xd7 = Dropout(0.3)(xd7)
    output_direction_7d = Dense(1, activation='sigmoid', name='output_direction_7d')(xd7)
    
    model = Model(inputs=inputs, outputs=[output_5d, output_7d, output_direction, output_direction_7d])
    
    optimizer = Adam(learning_rate=0.0003)
    model.compile(
        optimizer=optimizer,
        loss={
            'output_5d': 'huber',
            'output_7d': 'huber',
            'output_direction': 'binary_crossentropy',
            'output_direction_7d': 'binary_crossentropy'
        },
        loss_weights={
            'output_5d': 1.0,
            'output_7d': 1.0,
            'output_direction': 3.0,
            'output_direction_7d': 3.0
        },
        metrics={
            'output_5d': ['mae', 'mse'],
            'output_7d': ['mae', 'mse'],
            'output_direction': 'accuracy',
            'output_direction_7d': 'accuracy'
        }
    )
    
    return model

def ensemble_predict(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X, verbose=0)
        predictions.append(pred)
    
    avg_predictions = [
        np.mean([p[0] for p in predictions], axis=0),
        np.mean([p[1] for p in predictions], axis=0),
        np.mean([p[2] for p in predictions], axis=0),
        np.mean([p[3] for p in predictions], axis=0)
    ]
    
    return avg_predictions

def plot_predictions_vs_actual(actual, predicted, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_training_history(history, fold):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output_direction_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_output_direction_accuracy'], label='Validation Accuracy')
    plt.title(f'Direction Accuracy - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_fold_{fold}.png')
    plt.close()

def train_and_evaluate():
    df = load_data()
    df = add_technical_indicators(df)
    df = add_sentiment_features(df)
    
    X, y5, y7, yd5, yd7 = prepare_sequences(df)
    
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y5_train, y5_val = y5[:train_size], y5[train_size:]
    y7_train, y7_val = y7[:train_size], y7[train_size:]
    yd5_train, yd5_val = yd5[:train_size], yd5[train_size:]
    yd7_train, yd7_val = yd7[:train_size], yd7[train_size:]
    
    n_models = 5
    all_models = []
    all_metrics = []
    
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}")
        model = build_model(X_train.shape[1], X_train.shape[2])
        
        history = model.fit(
            X_train, [y5_train, y7_train, yd5_train, yd7_train],
            epochs=200,
            batch_size=64,
            validation_data=(X_val, [y5_val, y7_val, yd5_val, yd7_val]),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001),
                ModelCheckpoint(f'best_model_fold_{i+1}.h5', monitor='val_loss', save_best_only=True)
            ],
            verbose=1
        )
        
        plot_training_history(history, i+1)
        
        predictions = model.predict(X_val)
        metrics = {
            '5d_mae': mean_absolute_error(y5_val, predictions[0]),
            '5d_rmse': np.sqrt(mean_squared_error(y5_val, predictions[0])),
            '7d_mae': mean_absolute_error(y7_val, predictions[1]),
            '7d_rmse': np.sqrt(mean_squared_error(y7_val, predictions[1])),
            'direction_accuracy': accuracy_score(yd5_val, predictions[2] > 0.5),
            'direction_7d_accuracy': accuracy_score(yd7_val, predictions[3] > 0.5)
        }
        all_models.append(model)
        all_metrics.append(metrics)
        
        print(f"\nModel {i+1} Metrics:")
        print(f"5-Day MAE: {metrics['5d_mae']:.4f}")
        print(f"5-Day RMSE: {metrics['5d_rmse']:.4f}")
        print(f"7-Day MAE: {metrics['7d_mae']:.4f}")
        print(f"7-Day RMSE: {metrics['7d_rmse']:.4f}")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        print(f"Direction 7d Accuracy: {metrics['direction_7d_accuracy']:.4f}")
    
    ensemble_predictions = ensemble_predict(all_models, X_val)
    
    plot_predictions_vs_actual(y5_val, ensemble_predictions[0], '5-Day Predictions', 'predictions_vs_actual_5-day_predictions_fold_ensemble.png')
    plot_predictions_vs_actual(y7_val, ensemble_predictions[1], '7-Day Predictions', 'predictions_vs_actual_7-day_predictions_fold_ensemble.png')
    
    ensemble_metrics = {
        '5d_mae': mean_absolute_error(y5_val, ensemble_predictions[0]),
        '5d_rmse': np.sqrt(mean_squared_error(y5_val, ensemble_predictions[0])),
        '7d_mae': mean_absolute_error(y7_val, ensemble_predictions[1]),
        '7d_rmse': np.sqrt(mean_squared_error(y7_val, ensemble_predictions[1])),
        'direction_accuracy': accuracy_score(yd5_val, ensemble_predictions[2] > 0.5),
        'direction_7d_accuracy': accuracy_score(yd7_val, ensemble_predictions[3] > 0.5)
    }
    
    print("\nEnsemble Model Metrics:")
    print(f"5-Day MAE: {ensemble_metrics['5d_mae']:.4f}")
    print(f"5-Day RMSE: {ensemble_metrics['5d_rmse']:.4f}")
    print(f"7-Day MAE: {ensemble_metrics['7d_mae']:.4f}")
    print(f"7-Day RMSE: {ensemble_metrics['7d_rmse']:.4f}")
    print(f"Direction Accuracy: {ensemble_metrics['direction_accuracy']:.4f}")
    print(f"Direction 7d Accuracy: {ensemble_metrics['direction_7d_accuracy']:.4f}")
    
    avg_metrics = pd.DataFrame(all_metrics).mean()
    print("\nAverage Metrics Across All Models:")
    print(f"5-Day MAE: {avg_metrics['5d_mae']:.4f}")
    print(f"5-Day RMSE: {avg_metrics['5d_rmse']:.4f}")
    print(f"7-Day MAE: {avg_metrics['7d_mae']:.4f}")
    print(f"7-Day RMSE: {avg_metrics['7d_rmse']:.4f}")
    print(f"Direction Accuracy: {avg_metrics['direction_accuracy']:.4f}")
    print(f"Direction 7d Accuracy: {avg_metrics['direction_7d_accuracy']:.4f}")
    
    return avg_metrics

if __name__ == "__main__":
    train_and_evaluate()