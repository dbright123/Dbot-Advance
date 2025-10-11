import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
# Adjust these parameters as needed
MODEL_PATH_AUDUSD = 'GeneratedAUDUSD lstm_best.keras'
MODEL_PATH_NZDUSD = 'GeneratedNZDUSD lstm_best.keras'
DATA_PATH_AUDUSD_H1 = 'GeneratedAUDUSD dbot.csv'
DATA_PATH_NZDUSD_H1 = 'GeneratedNZDUSD dbot.csv'
DATA_PATH_AUDUSD_D1 = 'GeneratedAUDUSD test.csv'
DATA_PATH_NZDUSD_D1 = 'GeneratedNZDUSD test.csv'
SEQ_LEN = 240  # Sequence length used for training the model
PRED_STEPS = 5  # Number of steps the model predicts
SYMBOL_TO_SIMULATE = 'AUDUSD'  # Change to 'NZDUSD' to simulate the other pair

# --- Load Model and Data ---
def load_data_and_model(symbol):
    """Loads the appropriate data and model based on the symbol."""
    if symbol == 'AUDUSD':
        model_path = MODEL_PATH_AUDUSD
        data_path_h1 = DATA_PATH_AUDUSD_H1
        data_path_d1 = DATA_PATH_AUDUSD_D1
    elif symbol == 'NZDUSD':
        model_path = MODEL_PATH_NZDUSD
        data_path_h1 = DATA_PATH_NZDUSD_H1
        data_path_d1 = DATA_PATH_NZDUSD_D1
    else:
        raise ValueError("Symbol must be 'AUDUSD' or 'NZDUSD'")

    required_files = [model_path, data_path_h1, data_path_d1]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            print("Please ensure all necessary files are in the same directory as this script.")
            exit()

    print(f"Loading H1 data from {data_path_h1}...")
    df_h1 = pd.read_csv(data_path_h1)
    print(f"Loading D1 data from {data_path_d1}...")
    df_d1 = pd.read_csv(data_path_d1)
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return df_h1, df_d1, model

# --- Data Preparation ---
def create_sequences(data, seq_len):
    """Creates a sequence from the last `seq_len` data points."""
    if len(data) < seq_len:
        return None
    return np.array([data[-seq_len:]])

# --- Trading Simulation ---
def run_simulation(data_h1, data_d1, model):
    """Runs the backtesting trading simulation."""
    print("\nStarting trading simulation...")
    in_trade = False
    trade_entry_price = 0
    trade_type = None  # 'long' or 'short'
    trade_history = []
    balance = 100  # Starting balance for P/L calculation

    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 8))

    # Start the loop later to ensure there's enough D1 data for prediction
    start_index = SEQ_LEN * 24 
    if start_index >= len(data_h1):
        print("Error: Not enough H1 data to start the simulation based on SEQ_LEN.")
        return

    for i in range(start_index, len(data_h1)):
        current_price = data_h1['close'].iloc[i]
        
        # --- Use D1 data for prediction ---
        # Map current H1 bar to the corresponding D1 bar
        d1_index = i // 24
        if d1_index < SEQ_LEN:
            continue # Not enough D1 history yet

        prediction_data_sequence = data_d1[['open', 'high', 'low', 'close']].iloc[d1_index - SEQ_LEN:d1_index].values
        
        # Prepare input for the model
        model_input = np.reshape(prediction_data_sequence, (1, SEQ_LEN, 4))

        # Get predictions based on D1 data
        predictions = model.predict(model_input, verbose=0)[0]
        first_pred = predictions[0]
        tp_price = np.max(predictions)
        sl_price = np.min(predictions)

        # --- Trading Logic (executed on H1 data) ---
        if not in_trade:
            # Entry conditions
            if first_pred > current_price:  # Predicted price is higher, consider a long trade
                in_trade = True
                trade_type = 'long'
                trade_entry_price = current_price
                print(f"\nTick {i}: --- NEW LONG TRADE ---")
                print(f"Entry Price: {trade_entry_price:.5f}")
            elif first_pred < current_price:  # Predicted price is lower, consider a short trade
                in_trade = True
                trade_type = 'short'
                trade_entry_price = current_price
                print(f"\nTick {i}: --- NEW SHORT TRADE ---")
                print(f"Entry Price: {trade_entry_price:.5f}")

        if in_trade:
            # Update TP and SL based on new predictions
            if trade_type == 'long':
                take_profit = tp_price
                stop_loss = sl_price
            else:  # Short trade
                take_profit = sl_price
                stop_loss = tp_price

            # --- Breakeven Logic ---
            if trade_type == 'long':
                trigger_price = (take_profit + trade_entry_price) / 2
                if current_price > trigger_price:
                    breakeven_price = (current_price + trade_entry_price) / 2
                    # Move stop loss to lock in some profit or breakeven
                    if breakeven_price > stop_loss:
                        stop_loss = breakeven_price
                        print(f"Tick {i}: --- BREAKEVEN TRIGGERED (LONG) --- New SL: {stop_loss:.5f}")

            elif trade_type == 'short':
                trigger_price = (take_profit + trade_entry_price) / 2
                if current_price < trigger_price:
                    breakeven_price = (current_price + trade_entry_price) / 2
                    # Move stop loss to lock in some profit or breakeven
                    if breakeven_price < stop_loss:
                        stop_loss = breakeven_price
                        print(f"Tick {i}: --- BREAKEVEN TRIGGERED (SHORT) --- New SL: {stop_loss:.5f}")


            print(f"Tick {i}: Current Price: {current_price:.5f}, TP: {take_profit:.5f}, SL: {stop_loss:.5f}")

            # Check for exit conditions
            trade_closed = False
            profit_loss = 0
            if trade_type == 'long':
                if current_price >= take_profit:
                    print(f"Tick {i}: *** TAKE PROFIT HIT (LONG) at {current_price:.5f} ***")
                    profit_loss = (current_price - trade_entry_price) * 1000
                    trade_closed = True
                elif current_price <= stop_loss:
                    print(f"Tick {i}: *** STOP LOSS HIT (LONG) at {current_price:.5f} ***")
                    profit_loss = (current_price - trade_entry_price) * 1000
                    trade_closed = True
                elif take_profit < trade_entry_price or stop_loss > trade_entry_price:
                     print(f"Tick {i}: *** INVALID TP/SL, CLOSING TRADE (LONG) at {current_price:.5f} ***")
                     profit_loss = (current_price - trade_entry_price) * 1000
                     trade_closed = True

            elif trade_type == 'short':
                if current_price <= take_profit:
                    print(f"Tick {i}: *** TAKE PROFIT HIT (SHORT) at {current_price:.5f} ***")
                    profit_loss = (trade_entry_price - current_price) * 1000
                    trade_closed = True
                elif current_price >= stop_loss:
                    print(f"Tick {i}: *** STOP LOSS HIT (SHORT) at {current_price:.5f} ***")
                    profit_loss = (trade_entry_price - current_price) * 1000
                    trade_closed = True
                elif take_profit > trade_entry_price or stop_loss < trade_entry_price:
                    print(f"Tick {i}: *** INVALID TP/SL, CLOSING TRADE (SHORT) at {current_price:.5f} ***")
                    profit_loss = (trade_entry_price - current_price) * 1000
                    trade_closed = True


            if trade_closed:
                balance += profit_loss
                trade_history.append({
                    'entry_tick': len(trade_history) * (i - start_index), # Simplified tick
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

        # --- Plotting ---
        ax.clear()
        # Plot recent price history
        plot_start = max(start_index, i - 200)
        ax.plot(data_h1['close'].iloc[plot_start:i+1].values, label='Close Price (H1)', color='blue')

        if in_trade:
            # Plot trade entry
            ax.plot(i - plot_start, trade_entry_price, 'go', markersize=8, label=f'{trade_type.upper()} Entry')
            # Plot TP and SL lines
            ax.axhline(y=take_profit, color='g', linestyle='--', label='Take Profit (from D1)')
            ax.axhline(y=stop_loss, color='r', linestyle='--', label='Stop Loss (from D1)')

        ax.set_title(f'{SYMBOL_TO_SIMULATE} Trading Simulation (Tick: {i}) - Balance: ${balance:.2f}')
        ax.set_xlabel('Time (H1 Ticks)')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        plt.grid(True)
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    print("\n--- Simulation Finished ---")
    print(f"Total Ticks Simulated: {len(data_h1) - start_index}")
    print(f"Final Balance: ${balance:.2f}")
    
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        print("\n--- Trade History ---")
        print(trade_df.to_string())
        
        # --- Performance Summary ---
        print("\n--- Performance Summary ---")
        print(f"Total Trades: {len(trade_df)}")
        print(f"Winning Trades: {len(trade_df[trade_df['profit_loss'] > 0])}")
        print(f"Losing Trades: {len(trade_df[trade_df['profit_loss'] <= 0])}")
        if len(trade_df) > 0:
          print(f"Win Rate: {(len(trade_df[trade_df['profit_loss'] > 0]) / len(trade_df)) * 100:.2f}%")
        if len(trade_df[trade_df['profit_loss'] > 0]) > 0:
            print(f"Average Win: ${trade_df[trade_df['profit_loss'] > 0]['profit_loss'].mean():.2f}")
        if len(trade_df[trade_df['profit_loss'] <= 0]) > 0:
            print(f"Average Loss: ${trade_df[trade_df['profit_loss'] <= 0]['profit_loss'].mean():.2f}")


if __name__ == '__main__':
    market_data_h1, market_data_d1, lstm_model = load_data_and_model(SYMBOL_TO_SIMULATE)
    run_simulation(market_data_h1, market_data_d1, lstm_model)

