import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import os
import time
import MetaTrader5 as mt5 # <-- Import the MT5 library
from scaler3d2d import transform_data, inverse_transform_data

# --- Configuration ---

# --- TRADING PARAMETERS ---
SYMBOL_TO_TRADE = 'GBPUSD'
TIMEFRAME = mt5.TIMEFRAME_H1 # Using H1 timeframe
VOLUME = 0.01 # Lot size for trades
MAGIC_NUMBER = 123456 # Unique identifier for this bot's trades

# --- STRATEGY PARAMETERS ---
SEQ_LEN = 240
PRED_STEPS = 5
N_CLUSTERS = 5
SL_BUFFER_PIPS = 20
PRICE_NEAR_CLUSTER_PIPS = 10
BREAKEVEN_PROFIT_PIPS = 10
TP_RISK_REWARD_RATIO = 5

# --- Load Model ---
def load_trading_model(symbol):
    
    model_path = f'Generated{symbol} lstm_best.keras'

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        exit()
        
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return model

# --- Helper Functions (Pip calculations are the same) ---
def get_pip_value(symbol):
    """Returns the pip value for a given symbol."""
    if 'JPY' in symbol: return 0.01
    return 0.0001 # Simplified for Forex, adjust if needed

def pips_to_price(pips, symbol):
    return pips * get_pip_value(symbol)

def price_to_pips(price_diff, symbol):
    return price_diff / get_pip_value(symbol)

def get_nearest_cluster(current_price, cluster_centers):
    distances = [abs(c[0] - current_price) for c in cluster_centers]
    nearest_idx = distances.index(min(distances))
    return cluster_centers[nearest_idx][0], nearest_idx

def is_price_near_cluster(current_price, cluster_value, tolerance_pips, symbol):
    tolerance = pips_to_price(tolerance_pips, symbol)
    return abs(current_price - cluster_value) <= tolerance

# --- MT5 Interaction Functions ---

def get_mt5_data(symbol, timeframe, count):
    """Fetches historical bar data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        print(f"Error fetching data for {symbol} from MT5.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def check_for_open_trade(symbol, magic_number):
    """Checks if a trade managed by this bot is already open."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == magic_number:
            return pos # Return the position object if found
    return None

def place_order(symbol, trade_type, volume, sl, tp, magic_number):
    """Sends a trade order to MT5."""
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask if trade_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic_number,
        "comment": "dtrade_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order send failed, retcode={result.retcode}")
        return None
    
    print(f"SUCCESS: Order sent for {symbol}, ticket {result.order}")
    return result

def modify_position_sl(position, new_sl, magic_number):
    """Modifies the stop loss of an existing position (for breakeven)."""
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp, # Keep the original TP
        "magic": magic_number,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to modify SL, retcode={result.retcode}")
    else:
        print(f"SUCCESS: Position {position.ticket} SL moved to breakeven: {new_sl:.5f}")


# --- Main Trading Loop ---
def run_live_trader():
    """The main function to run the live trading bot."""
    print("Starting DTrade Live Bot...")
    
    # --- INITIALIZATION ---
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
        
    print("MT5 Connection Successful")
    model = load_trading_model(SYMBOL_TO_TRADE)
    
    # Initialize plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 8))

    # --- THE LIVE LOOP ---
    while True:
        try:
            # --- 1. GET DATA & CURRENT PRICE ---
            # Fetch enough data to create sequences
            data_h1 = get_mt5_data(SYMBOL_TO_TRADE, TIMEFRAME, SEQ_LEN + 200) # Fetch more for context
            if data_h1 is None:
                time.sleep(5)
                continue
            
            current_price = mt5.symbol_info_tick(SYMBOL_TO_TRADE).bid
            
            # --- 2. CHECK TRADE STATE ---
            open_position = check_for_open_trade(SYMBOL_TO_TRADE, MAGIC_NUMBER)
            in_trade = open_position is not None
            
            # --- 3. TRADE MANAGEMENT (IF IN A TRADE) ---
            if in_trade:
                trade_entry_price = open_position.price_open
                trade_type = 'long' if open_position.type == mt5.ORDER_TYPE_BUY else 'short'
                
                if trade_type == 'long':
                    current_profit_pips = price_to_pips(current_price - trade_entry_price, SYMBOL_TO_TRADE)
                else:
                    current_profit_pips = price_to_pips(trade_entry_price - current_price, SYMBOL_TO_TRADE)
                
                # Breakeven Logic
                if current_profit_pips >= BREAKEVEN_PROFIT_PIPS:
                    # Check if SL is already at or past breakeven
                    breakeven_price = trade_entry_price + pips_to_price(1, SYMBOL_TO_TRADE) # 1 pip profit BE
                    if (trade_type == 'long' and open_position.sl < breakeven_price) or \
                       (trade_type == 'short' and open_position.sl > breakeven_price):
                        print(f"🛡️ BREAKEVEN triggered. Moving SL for position {open_position.ticket}")
                        modify_position_sl(open_position, breakeven_price, MAGIC_NUMBER)
            
            # --- 4. TRADE ENTRY LOGIC (IF NOT IN A TRADE) ---
            if not in_trade:
                # Use the most recent data for clusters and prediction
                recent_data = data_h1.tail(SEQ_LEN)
                
                cluster_data = recent_data['close'].values.reshape(-1, 1)
                kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto').fit(cluster_data)
                cluster_centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
                
                # Prepare data for prediction
                prediction_data_sequence = data_h1[['open', 'high', 'low', 'close']].tail(SEQ_LEN).values
                model_input = np.reshape(prediction_data_sequence, (1, SEQ_LEN, 4))
                model_input, _ = transform_data(model_input, scaler_x_filename=f'{SYMBOL_TO_TRADE} scaler_x.joblib')
                
                predictions = model.predict(model_input, verbose=0)
                _, predictions = inverse_transform_data(scaled_y=predictions, scaler_y_filename=f'{SYMBOL_TO_TRADE} scaler_y.joblib')
                predictions = predictions[0]

                nearest_cluster, _ = get_nearest_cluster(current_price, cluster_centers)
                is_near_support_resistance = is_price_near_cluster(current_price, nearest_cluster, PRICE_NEAR_CLUSTER_PIPS, SYMBOL_TO_TRADE)

                # LONG ENTRY CONDITION
                if (np.all(predictions > nearest_cluster) and is_near_support_resistance):
                    stop_loss = nearest_cluster - pips_to_price(SL_BUFFER_PIPS, SYMBOL_TO_TRADE)
                    sl_distance = current_price - stop_loss
                    take_profit = current_price + (sl_distance * TP_RISK_REWARD_RATIO)
                    
                    print(f"\n🟢 LONG SIGNAL DETECTED at {current_price:.5f}")
                    print(f"   Support Cluster: {nearest_cluster:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    place_order(SYMBOL_TO_TRADE, mt5.ORDER_TYPE_BUY, VOLUME, stop_loss, take_profit, MAGIC_NUMBER)

                # SHORT ENTRY CONDITION
                elif (np.all(predictions < nearest_cluster) and is_near_support_resistance):
                    stop_loss = nearest_cluster + pips_to_price(SL_BUFFER_PIPS, SYMBOL_TO_TRADE)
                    sl_distance = stop_loss - current_price
                    take_profit = current_price - (sl_distance * TP_RISK_REWARD_RATIO)

                    print(f"\n🔴 SHORT SIGNAL DETECTED at {current_price:.5f}")
                    print(f"   Resistance Cluster: {nearest_cluster:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    place_order(SYMBOL_TO_TRADE, mt5.ORDER_TYPE_SELL, VOLUME, stop_loss, take_profit, MAGIC_NUMBER)

            # --- 5. PLOTTING AND VISUALIZATION ---
            ax.clear()
            plot_data = data_h1.tail(200) # Plot the last 200 candles
            ax.plot(plot_data.index, plot_data['close'], label='Close Price', color='blue', linewidth=1)
            
            # Plot Clusters
            for center in cluster_centers:
                ax.axhline(y=center[0], color='orange', linestyle='--', linewidth=1, alpha=0.8)
            ax.axhline(y=nearest_cluster, color='purple', linestyle='-', linewidth=1.5, label=f'Nearest Cluster')

            # Plot trade info if in a trade
            if in_trade:
                entry_price = open_position.price_open
                sl = open_position.sl
                tp = open_position.tp
                trade_color = 'green' if open_position.type == mt5.ORDER_TYPE_BUY else 'red'
                
                ax.axhline(y=entry_price, color=trade_color, linestyle='-', linewidth=1, label=f'Entry: {entry_price:.5f}')
                ax.axhline(y=sl, color='red', linestyle='--', linewidth=1, label=f'SL: {sl:.5f}')
                ax.axhline(y=tp, color='green', linestyle='--', linewidth=1, label=f'TP: {tp:.5f}')

            ax.set_title(f'{SYMBOL_TO_TRADE} Live Chart | Current Price: {current_price:.5f}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.pause(1) # Pause to allow chart to update

            # Wait before the next check. For H1, checking every few minutes is fine.
            print("...waiting for next candle...")
            time.sleep(60) # Check every 1 minutes

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            time.sleep(60) # Wait a minute before retrying on error

if __name__ == '__main__':
    run_live_trader()