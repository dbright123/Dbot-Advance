import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD DATA FROM EXCEL
# ============================================================================

def load_data_from_excel(file_path, sheet_name=0):
    """
    Load trading data from Excel file
    
    Expected columns: time/date, hour, minute, day, month, open, high, low, close, volume, spread
    Or the function will try to extract time features from a datetime column
    """
    print(f"Loading data from: {file_path}")
    
    # Read Excel file (handles large files efficiently)
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    print(f"Initial data shape: {df.shape}")
    print(f"Columns found: {df.columns.tolist()}")
    
    # Standardize column names (make lowercase and remove spaces)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Check if we have a datetime column
    datetime_cols = ['time', 'date', 'datetime', 'timestamp']
    datetime_col = None
    
    for col in datetime_cols:
        if col in df.columns:
            datetime_col = col
            break
    
    # If datetime column exists, extract time features
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        if 'hour' not in df.columns:
            df['hour'] = df[datetime_col].dt.hour
        if 'minute' not in df.columns:
            df['minute'] = df[datetime_col].dt.minute
        if 'day' not in df.columns:
            df['day'] = df[datetime_col].dt.day
        if 'month' not in df.columns:
            df['month'] = df[datetime_col].dt.month
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df[datetime_col].dt.dayofweek
    
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nWARNING: Missing required columns: {missing_cols}")
        print("Please ensure your Excel file has: open, high, low, close columns")
        return None
    
    # Handle volume column (might be named differently)
    volume_cols = ['volume', 'tick_volume', 'vol']
    volume_col = None
    for col in volume_cols:
        if col in df.columns:
            volume_col = col
            break
    
    if volume_col and volume_col != 'tick_volume':
        df['tick_volume'] = df[volume_col]
    elif 'tick_volume' not in df.columns:
        print("WARNING: No volume column found. Creating dummy volume data.")
        df['tick_volume'] = 1000  # Default value
    
    # Calculate spread if not present
    if 'spread' not in df.columns:
        df['spread'] = df['high'] - df['low']
        print("INFO: Spread column not found. Calculated from high-low.")
    
    # Ensure all required time features exist
    time_features = ['hour', 'minute', 'day', 'month']
    for feature in time_features:
        if feature not in df.columns:
            print(f"WARNING: {feature} column not found. Setting to 0.")
            df[feature] = 0
    
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = 0
    
    # Remove any rows with NaN in critical columns
    critical_cols = ['open', 'high', 'low', 'close']
    df = df.dropna(subset=critical_cols)
    
    print(f"\nData loaded successfully!")
    print(f"Final shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}" if datetime_col else "No datetime index")
    
    return df

# ============================================================================
# STEP 2: CLUSTER PRICES FOR SUPPORT/RESISTANCE
# ============================================================================

def identify_support_resistance(df, n_clusters=5):
    """Use K-Means clustering to identify support and resistance levels"""
    prices = df['close'].values.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['price_cluster'] = kmeans.fit_predict(prices)
    
    # Get cluster centers (support/resistance levels)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Assign nearest cluster center to each row
    df['nearest_sr_level'] = df['close'].apply(
        lambda x: min(cluster_centers, key=lambda level: abs(level - x))
    )
    
    return df, cluster_centers

# ============================================================================
# STEP 3: CREATE BUY/SELL LABELS
# ============================================================================

def create_labels(df, cluster_centers, tolerance=0.0005):
    """
    Create buy/sell labels based on price action around support/resistance
    Label 0: Hold/No action
    Label 1: Buy signal
    Label 2: Sell signal
    """
    labels = np.zeros(len(df))
    
    # Sort cluster centers to identify support and resistance
    sr_levels = sorted(cluster_centers)
    
    for i in range(1, len(df) - 1):
        current_price = df.iloc[i]['close']
        prev_price = df.iloc[i-1]['close']
        next_price = df.iloc[i+1]['close']
        
        # Find nearest support and resistance
        lower_levels = [level for level in sr_levels if level < current_price]
        upper_levels = [level for level in sr_levels if level > current_price]
        
        support = lower_levels[-1] if lower_levels else sr_levels[0]
        resistance = upper_levels[0] if upper_levels else sr_levels[-1]
        
        # Buy signal (1): Price at support and moving up to resistance
        if abs(current_price - support) <= tolerance * current_price:
            # Check if price moves towards resistance in next periods
            future_high = df.iloc[i:min(i+20, len(df))]['high'].max()
            if future_high >= resistance * (1 - tolerance):
                labels[i] = 1
        
        # Sell signal (2): Price at resistance and moving down
        elif abs(current_price - resistance) <= tolerance * current_price:
            # Check if price moves down
            future_low = df.iloc[i:min(i+20, len(df))]['low'].min()
            if future_low <= support * (1 + tolerance):
                labels[i] = 2
        
        # Breakout buy (1): Price breaks resistance upward
        elif prev_price < resistance and current_price > resistance:
            # Confirm breakout
            if next_price > resistance:
                labels[i] = 1
        
        # Breakdown sell (2): Price breaks support downward
        elif prev_price > support and current_price < support:
            # Confirm breakdown
            if next_price < support:
                labels[i] = 2
    
    df['label'] = labels.astype(int)
    return df

# ============================================================================
# STEP 4: ADD TECHNICAL INDICATORS
# ============================================================================

def add_technical_indicators(df):
    """Add RSI, MACD, and other technical indicators"""
    
    # RSI (Relative Strength Index)
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['close'])
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(4)
    
    # Volume indicators
    df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
    
    # Price rate of change
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

# ============================================================================
# STEP 5: VISUALIZE BUY/SELL SIGNALS
# ============================================================================

def plot_signals(df, cluster_centers):
    """Plot price chart with buy/sell signals"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot close price
    ax.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.6)
    
    # Plot support/resistance levels
    for level in cluster_centers:
        ax.axhline(y=level, color='gray', linestyle='--', alpha=0.3)
    
    # Plot buy signals
    buy_signals = df[df['label'] == 1]
    ax.scatter(buy_signals.index, buy_signals['close'], 
               color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    
    # Plot sell signals
    sell_signals = df[df['label'] == 2]
    ax.scatter(sell_signals.index, sell_signals['close'], 
               color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax.set_xlabel('Index')
    ax.set_ylabel('Price')
    ax.set_title('Trading Signals: Buy and Sell Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trading_signals.png', dpi=300)
    plt.show()
    
    print(f"\nSignal Distribution:")
    print(f"Hold/No Action (0): {len(df[df['label'] == 0])}")
    print(f"Buy Signals (1): {len(df[df['label'] == 1])}")
    print(f"Sell Signals (2): {len(df[df['label'] == 2])}")

# ============================================================================
# STEP 6: TRAIN CLASSIFICATION MODELS
# ============================================================================

def train_classification_models(df):
    """Train multiple classification models"""
    
    # Prepare features and labels
    feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread',
                    'hour', 'minute', 'day', 'month', 'day_of_week',
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'sma_20', 'sma_50', 'ema_12',
                    'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                    'momentum', 'volume_ratio', 'roc', 'price_cluster']
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*70)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*70)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300)
        plt.show()
    
    return results, X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ============================================================================
# STEP 7: TRAIN DEEP NEURAL NETWORK
# ============================================================================

def train_deep_neural_network(X_train, X_test, y_train, y_test):
    """Train a deep neural network for classification"""
    
    print("\n" + "="*70)
    print("TRAINING DEEP NEURAL NETWORK")
    print("="*70)
    
    # Convert labels to categorical
    num_classes = len(np.unique(y_train))
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Build model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n\nDeep Neural Network Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - Deep Neural Network')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_DNN.png', dpi=300)
    plt.show()
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dnn_training_history.png', dpi=300)
    plt.show()
    
    return model, accuracy, cm

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(excel_file_path='trading_data.xlsx', sheet_name=0):
    """
    Main execution function
    
    Parameters:
    -----------
    excel_file_path : str
        Path to your Excel file containing trading data
    sheet_name : str or int
        Sheet name or index to read from Excel file
    """
    print("="*70)
    print("TRADING SIGNAL CLASSIFICATION SYSTEM")
    print("="*70)
    
    # Step 1: Load data from Excel
    print("\nStep 1: Loading data from Excel file...")
    df = load_data_from_excel(excel_file_path, sheet_name)
    
    if df is None:
        print("Failed to load data. Please check your Excel file.")
        return
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Step 2: Identify support/resistance
    print("\nStep 2: Identifying support and resistance levels...")
    df, cluster_centers = identify_support_resistance(df, n_clusters=5)
    print(f"Support/Resistance levels: {cluster_centers}")
    
    # Step 3: Create labels
    print("\nStep 3: Creating buy/sell labels...")
    df = create_labels(df, cluster_centers)
    
    # Step 4: Add technical indicators
    print("\nStep 4: Adding technical indicators...")
    df = add_technical_indicators(df)
    print(f"Final dataset shape: {df.shape}")
    
    # Step 5: Visualize signals
    print("\nStep 5: Visualizing trading signals...")
    plot_signals(df, cluster_centers)
    
    # Step 6: Train classification models
    print("\nStep 6: Training classification models...")
    results, X_train, X_test, y_train, y_test, scaler = train_classification_models(df)
    
    # Step 7: Train deep neural network
    print("\nStep 7: Training deep neural network...")
    dnn_model, dnn_accuracy, dnn_cm = train_deep_neural_network(
        X_train, X_test, y_train, y_test
    )
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f}")
    print(f"Deep Neural Network: {dnn_accuracy:.4f}")
    
    print("\nAll models trained successfully!")
    print("Confusion matrices and plots saved to current directory.")
    
    return df, results, dnn_model, scaler

if __name__ == "__main__":
    # CHANGE THIS TO YOUR EXCEL FILE PATH
    excel_file = 'trading_data.xlsx'  # Replace with your file path
    
    # If your data is in a specific sheet, specify it here
    # sheet_name can be the sheet name (string) or sheet index (integer, 0-based)
    sheet = 0  # or 'Sheet1', etc.
    
    df, results, dnn_model, scaler = main(excel_file_path=excel_file, sheet_name=sheet)