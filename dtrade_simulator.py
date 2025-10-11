import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
MODEL_PATH_AUDUSD = 'GeneratedAUDUSD lstm_best.keras'
MODEL_PATH_NZDUSD = 'GeneratedNZDUSD lstm_best.keras'
DATA_PATH_AUDUSD_H1 = 'GeneratedAUDUSD dbot.csv'
DATA_PATH_NZDUSD_H1 = 'GeneratedNZDUSD dbot.csv'
SEQ_LEN = 240
PRED_STEPS = 5
SYMBOL_TO_SIMULATE = 'AUDUSD'
N_CLUSTERS = 3
RISK_REWARD_RATIO = 1.5
ATR_PERIOD = 14
MIN_CLUSTER_DISTANCE = 0.002  # Minimum distance between clusters to be significant

def calculate_atr(df, period=14):
    """Calculate Average True Range for dynamic position sizing"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(window=period).mean()
    # Fill initial NaN values
    df['atr'].fillna(method='bfill', inplace=True)
    return df

# --- Load Model and Data ---
def load_data_and_model(symbol):
    """Loads the appropriate data and model based on the symbol."""
    if symbol == 'AUDUSD':
        model_path = MODEL_PATH_AUDUSD
        data_path_h1 = DATA_PATH_AUDUSD_H1
    elif symbol == 'NZDUSD':
        model_path = MODEL_PATH_NZDUSD
        data_path_h1 = DATA_PATH_NZDUSD_H1
    else:
        raise ValueError("Symbol must be 'AUDUSD' or 'NZDUSD'")

    required_files = [model_path, data_path_h1]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            print("Please ensure all necessary files are in the same directory as this script.")
            exit()

    print(f"Loading H1 data from {data_path_h1}...")
    df_h1 = pd.read_csv(data_path_h1)
    
    # Calculate ATR
    df_h1 = calculate_atr(df_h1, ATR_PERIOD)
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return df_h1, model

# --- Trading Simulation ---
def run_simulation(data_h1, model):
    """Runs the backtesting trading simulation with improved logic."""
    print("\nStarting trading simulation...")
    in_trade = False
    trade_entry_price = 0
    trade_entry_tick = 0
    trade_type = None
    take_profit = 0
    stop_loss = 0
    trade_history = []
    balance = 1000  # Increased starting balance for better P/L visibility

    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 8))

    # Start the loop to ensure there's enough H1 data for the first prediction
    start_index = max(SEQ_LEN, ATR_PERIOD)
    if start_index >= len(data_h1):
        print("Error: Not enough H1 data to start the simulation.")
        return

    for i in range(start_index, len(data_h1)):
        current_price = data_h1['close'].iloc[i]
        previous_close = data_h1['close'].iloc[i-1] if i > 0 else current_price
        current_atr = data_h1['atr'].iloc[i]

        # --- Improved K-Means Clustering for Support/Resistance ---
        # Use high and low prices for better S/R levels
        high_low_data = np.vstack([
            data_h1['high'].iloc[i - SEQ_LEN:i].values,
            data_h1['low'].iloc[i - SEQ_LEN:i].values
        ]).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto').fit(high_low_data)
        cluster_centers = sorted([center[0] for center in kmeans.cluster_centers_])
        
        # Filter clusters by significance
        significant_clusters = []
        for j, center in enumerate(cluster_centers):
            if j == 0 or abs(center - cluster_centers[j-1]) > MIN_CLUSTER_DISTANCE:
                significant_clusters.append(center)
        
        # Find nearest support and resistance
        supports = [c for c in significant_clusters if c < current_price]
        resistances = [c for c in significant_clusters if c > current_price]
        
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None

        # --- Use H1 data for prediction ---
        prediction_data_sequence = data_h1[['open', 'high', 'low', 'close']].iloc[i - SEQ_LEN:i].values
        model_input = np.reshape(prediction_data_sequence, (1, SEQ_LEN, 4))

        predictions = model.predict(model_input, verbose=0)[0]
        first_pred_price = predictions[0]
        last_pred_price = predictions[-1]
        pred_direction = 'bullish' if last_pred_price > current_price else 'bearish'

        # --- IMPROVED ENTRY LOGIC ---
        buy_condition = False
        sell_condition = False
        
        # Long entry: Price breaks above resistance with confirmation
        if (nearest_resistance and 
            previous_close < nearest_resistance and 
            current_price > nearest_resistance and
            first_pred_price > current_price and  # Prediction confirms upward momentum
            pred_direction == 'bullish'):
            
            buy_condition = True
        
        # Short entry: Price breaks below support with confirmation  
        elif (nearest_support and
              previous_close > nearest_support and
              current_price < nearest_support and
              first_pred_price < current_price and  # Prediction confirms downward momentum
              pred_direction == 'bearish'):
            
            sell_condition = True

        # --- Trading Logic ---

        # Handle existing trades
        if in_trade:
            trade_closed = False
            profit_loss = 0
            
            # Dynamic trailing stops
            if trade_type == 'long':
                # Move stop loss to breakeven + small profit when price moves favorably
                if current_price > trade_entry_price + (current_atr * 0.5):
                    new_stop = trade_entry_price + (current_atr * 0.2)
                    if new_stop > stop_loss:
                        stop_loss = new_stop
                        print(f"Tick {i}: Trailing SL updated to {stop_loss:.5f}")
                
                # Check exit conditions
                if current_price >= take_profit:
                    print(f"Tick {i}: *** TAKE PROFIT HIT (LONG) at {current_price:.5f} ***")
                    profit_loss = (current_price - trade_entry_price) * 10000  # Increased multiplier for better P/L
                    trade_closed = True
                elif current_price <= stop_loss:
                    print(f"Tick {i}: *** STOP LOSS HIT (LONG) at {current_price:.5f} ***")
                    profit_loss = (current_price - trade_entry_price) * 10000
                    trade_closed = True
                    
            elif trade_type == 'short':
                # Move stop loss to breakeven + small profit when price moves favorably
                if current_price < trade_entry_price - (current_atr * 0.5):
                    new_stop = trade_entry_price - (current_atr * 0.2)
                    if new_stop < stop_loss:
                        stop_loss = new_stop
                        print(f"Tick {i}: Trailing SL updated to {stop_loss:.5f}")
                
                # Check exit conditions
                if current_price <= take_profit:
                    print(f"Tick {i}: *** TAKE PROFIT HIT (SHORT) at {current_price:.5f} ***")
                    profit_loss = (trade_entry_price - current_price) * 10000
                    trade_closed = True
                elif current_price >= stop_loss:
                    print(f"Tick {i}: *** STOP LOSS HIT (SHORT) at {current_price:.5f} ***")
                    profit_loss = (trade_entry_price - current_price) * 10000
                    trade_closed = True

            if trade_closed:
                balance += profit_loss
                trade_history.append({
                    'entry_tick': trade_entry_tick,
                    'exit_tick': i,
                    'entry_price': trade_entry_price,
                    'exit_price': current_price,
                    'trade_type': trade_type,
                    'profit_loss': profit_loss,
                    'balance': balance
                })
                print(f"P/L: ${profit_loss:.2f}, New Balance: ${balance:.2f}")
                in_trade = False
                trade_type = None

        # Handle new entries if no trade is active
        if not in_trade:
            if buy_condition:
                in_trade = True
                trade_type = 'long'
                trade_entry_price = current_price
                trade_entry_tick = i
                
                # Dynamic stops based on ATR
                stop_loss = trade_entry_price - (current_atr * 1.0)
                take_profit = trade_entry_price + (current_atr * RISK_REWARD_RATIO)
                
                print(f"\nTick {i}: --- NEW LONG TRADE ---")
                print(f"Entry: {trade_entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                print(f"Resistance: {nearest_resistance:.5f}, ATR: {current_atr:.5f}")
                
            elif sell_condition:
                in_trade = True
                trade_type = 'short'
                trade_entry_price = current_price
                trade_entry_tick = i
                
                # Dynamic stops based on ATR
                stop_loss = trade_entry_price + (current_atr * 1.0)
                take_profit = trade_entry_price - (current_atr * RISK_REWARD_RATIO)
                
                print(f"\nTick {i}: --- NEW SHORT TRADE ---")
                print(f"Entry: {trade_entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
                print(f"Support: {nearest_support:.5f}, ATR: {current_atr:.5f}")

        # --- Plotting ---
        ax.clear()
        plot_start = max(start_index, i - 100)  # Show last 100 bars for clarity
        
        # Price line
        ax.plot(range(i - plot_start + 1), 
                data_h1['close'].iloc[plot_start:i+1].values, 
                label='Close Price', color='black', linewidth=1, zorder=10)
        
        # Plot significant support/resistance levels
        for level in significant_clusters:
            color = 'green' if level < current_price else 'red'
            linestyle = '--' if level != nearest_support and level != nearest_resistance else '-'
            linewidth = 1 if level != nearest_support and level != nearest_resistance else 2
            
            if level == nearest_support or level == nearest_resistance:
                label = 'Nearest S/R'
            else:
                label = None
                
            ax.axhline(y=level, color=color, linestyle=linestyle, 
                      linewidth=linewidth, alpha=0.7, zorder=5, label=label)
        
        # Mark current price
        ax.axhline(y=current_price, color='blue', linestyle=':', alpha=0.5, label='Current Price')
        
        # Mark trade entries and current positions
        if in_trade:
            entry_idx = trade_entry_tick - plot_start
            ax.plot(entry_idx, trade_entry_price, 
                   'g^' if trade_type == 'long' else 'rv', 
                   markersize=10, label=f'{trade_type.upper()} Entry', zorder=20)
            
            # Plot stop loss and take profit lines
            ax.axhline(y=stop_loss, color='red', linestyle='-', alpha=0.7, label='Stop Loss')
            ax.axhline(y=take_profit, color='green', linestyle='-', alpha=0.7, label='Take Profit')

        ax.set_title(f'{SYMBOL_TO_SIMULATE} Trading Simulation (Tick: {i}) - Balance: ${balance:.2f}')
        ax.set_xlabel('Bars')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    print("\n--- Simulation Finished ---")
    print(f"Total Ticks Simulated: {len(data_h1) - start_index}")
    print(f"Final Balance: ${balance:.2f}")
    
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        print("\n--- Trade History ---")
        print(trade_df.to_string(index=False))
        
        # --- Performance Summary ---
        print("\n--- Performance Summary ---")
        print(f"Total Trades: {len(trade_df)}")
        print(f"Winning Trades: {len(trade_df[trade_df['profit_loss'] > 0])}")
        print(f"Losing Trades: {len(trade_df[trade_df['profit_loss'] <= 0])}")
        if len(trade_df) > 0:
            win_rate = (len(trade_df[trade_df['profit_loss'] > 0]) / len(trade_df)) * 100
            print(f"Win Rate: {win_rate:.2f}%")
        if len(trade_df[trade_df['profit_loss'] > 0]) > 0:
            avg_win = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].mean()
            print(f"Average Win: ${avg_win:.2f}")
        if len(trade_df[trade_df['profit_loss'] <= 0]) > 0:
            avg_loss = trade_df[trade_df['profit_loss'] <= 0]['profit_loss'].mean()
            print(f"Average Loss: ${avg_loss:.2f}")
            
        # Calculate profit factor
        gross_profit = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(trade_df[trade_df['profit_loss'] <= 0]['profit_loss'].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
            print(f"Profit Factor: {profit_factor:.2f}")
        else:
            print("Profit Factor: Infinite (No losses)")
            
        # Maximum drawdown
        trade_df['cumulative_max'] = trade_df['balance'].cummax()
        trade_df['drawdown'] = (trade_df['cumulative_max'] - trade_df['balance']) / trade_df['cumulative_max'] * 100
        max_drawdown = trade_df['drawdown'].max()
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")

if __name__ == '__main__':
    market_data_h1, lstm_model = load_data_and_model(SYMBOL_TO_SIMULATE)
    run_simulation(market_data_h1, lstm_model)