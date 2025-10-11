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
SL_BUFFER_PIPS = 5  # Pips below/above cluster for stop loss (1 pip = 0.0001)
PRICE_NEAR_CLUSTER_PIPS = 10  # How close price needs to be to cluster for entry (in pips)
BREAKEVEN_PROFIT_PIPS = 5  # Pips in profit before moving SL to breakeven
TP_RISK_REWARD_RATIO = 4  # Take profit is 4x the stop loss distance

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
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return df_h1, model

# --- Helper Functions ---
def pips_to_price(pips):
    """Convert pips to price value (1 pip = 0.0001 for AUDUSD/NZDUSD)"""
    return pips * 0.0001

def get_nearest_cluster(current_price, cluster_centers):
    """Find the cluster center nearest to current price."""
    distances = [abs(c[0] - current_price) for c in cluster_centers]
    nearest_idx = distances.index(min(distances))
    return cluster_centers[nearest_idx][0], nearest_idx

def get_next_cluster_above(current_cluster_value, cluster_centers):
    """Get the next cluster above the current one."""
    centers_above = [c[0] for c in cluster_centers if c[0] > current_cluster_value]
    if centers_above:
        return min(centers_above)
    return None

def get_next_cluster_below(current_cluster_value, cluster_centers):
    """Get the next cluster below the current one."""
    centers_below = [c[0] for c in cluster_centers if c[0] < current_cluster_value]
    if centers_below:
        return max(centers_below)
    return None

def is_price_near_cluster(current_price, cluster_value, tolerance_pips):
    """Check if price is at or very close to a cluster level."""
    tolerance = pips_to_price(tolerance_pips)
    return abs(current_price - cluster_value) <= tolerance

# --- Trading Simulation ---
def run_simulation(data_h1, model):
    """Runs the backtesting trading simulation."""
    print("\nStarting trading simulation...")
    print(f"SL Buffer: {SL_BUFFER_PIPS} pips | Entry Tolerance: {PRICE_NEAR_CLUSTER_PIPS} pips")
    print(f"Breakeven: {BREAKEVEN_PROFIT_PIPS} pips | Risk:Reward Ratio: 1:{TP_RISK_REWARD_RATIO}\n")
    
    in_trade = False
    trade_entry_price = 0
    trade_entry_tick = 0
    trade_type = None
    breakeven_set = False
    take_profit = 0  # STATIC - set at entry
    stop_loss = 0
    trade_history = []
    balance = 100

    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 8))

    start_index = SEQ_LEN
    if start_index >= len(data_h1):
        print("Error: Not enough H1 data to start the simulation based on SEQ_LEN.")
        return

    for i in range(start_index, len(data_h1)):
        current_price = data_h1['close'].iloc[i]

        # --- K-Means Clustering for S/R Levels ---
        cluster_data = data_h1['close'].iloc[i - SEQ_LEN:i].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init='auto').fit(cluster_data)
        cluster_centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        # --- Model Prediction ---
        prediction_data_sequence = data_h1[['open', 'high', 'low', 'close']].iloc[i - SEQ_LEN:i].values
        model_input = np.reshape(prediction_data_sequence, (1, SEQ_LEN, 4))
        predictions = model.predict(model_input, verbose=0)[0]
        
        first_pred = predictions[0]
        last_pred = predictions[-1]

        # Get nearest cluster to current price
        nearest_cluster, cluster_idx = get_nearest_cluster(current_price, cluster_centers)

        # --- Handle Existing Trades ---
        if in_trade:
            # Breakeven Logic: Move SL to entry once in profit by BREAKEVEN_PROFIT_PIPS
            if not breakeven_set:
                breakeven_threshold = pips_to_price(BREAKEVEN_PROFIT_PIPS)
                
                if trade_type == 'long':
                    # Check if current price is at least 5 pips above entry
                    if current_price >= trade_entry_price + breakeven_threshold:
                        stop_loss = trade_entry_price
                        breakeven_set = True
                        profit_pips = (current_price - trade_entry_price) / 0.0001
                        print(f"Tick {i}: 🛡️ BREAKEVEN (LONG) - In profit by {profit_pips:.1f} pips - SL moved to: {stop_loss:.5f}")
                        
                elif trade_type == 'short':
                    # Check if current price is at least 5 pips below entry
                    if current_price <= trade_entry_price - breakeven_threshold:
                        stop_loss = trade_entry_price
                        breakeven_set = True
                        profit_pips = (trade_entry_price - current_price) / 0.0001
                        print(f"Tick {i}: 🛡️ BREAKEVEN (SHORT) - In profit by {profit_pips:.1f} pips - SL moved to: {stop_loss:.5f}")

            # --- Exit Conditions (TP/SL) ---
            trade_closed = False
            profit_loss = 0
            exit_reason = ""
            
            if trade_type == 'long':
                if current_price >= take_profit:
                    exit_reason = "TAKE PROFIT HIT (LONG)"
                    profit_loss = (current_price - trade_entry_price) * 1000
                    trade_closed = True
                elif current_price <= stop_loss:
                    exit_reason = "STOP LOSS HIT (LONG)"
                    profit_loss = (current_price - trade_entry_price) * 1000
                    trade_closed = True
            elif trade_type == 'short':
                if current_price <= take_profit:
                    exit_reason = "TAKE PROFIT HIT (SHORT)"
                    profit_loss = (trade_entry_price - current_price) * 1000
                    trade_closed = True
                elif current_price >= stop_loss:
                    exit_reason = "STOP LOSS HIT (SHORT)"
                    profit_loss = (trade_entry_price - current_price) * 1000
                    trade_closed = True

            if trade_closed:
                print(f"Tick {i}: *** {exit_reason} at {current_price:.5f} ***")
                balance += profit_loss
                trade_history.append({
                    'entry_tick': trade_entry_tick,
                    'exit_tick': i,
                    'entry_price': trade_entry_price,
                    'exit_price': current_price,
                    'trade_type': trade_type,
                    'profit_loss': profit_loss,
                    'balance': balance,
                    'exit_reason': exit_reason
                })
                print(f"P/L: ${profit_loss:.2f} | New Balance: ${balance:.2f}\n")
                in_trade = False
                trade_type = None

        # --- New Entry Logic ---
        if not in_trade:
            # LONG ENTRY CONDITIONS:
            # 1. ALL predicted prices must be above the nearest cluster (strong breakout signal)
            # 2. Current price is AT or NEAR the cluster from below (at support level)
            # This ensures we're catching a breakout, not consolidation
            if (np.all(predictions > nearest_cluster) and 
                is_price_near_cluster(current_price, nearest_cluster, PRICE_NEAR_CLUSTER_PIPS) and
                current_price <= nearest_cluster + pips_to_price(PRICE_NEAR_CLUSTER_PIPS)):
                
                in_trade = True
                trade_type = 'long'
                trade_entry_price = current_price
                trade_entry_tick = i
                
                # Calculate SL first
                stop_loss = nearest_cluster - pips_to_price(SL_BUFFER_PIPS)
                sl_distance = trade_entry_price - stop_loss
                
                # TP is 4x the SL distance (risk:reward = 1:4)
                take_profit = trade_entry_price + (sl_distance * TP_RISK_REWARD_RATIO)
                breakeven_set = False
                
                print(f"\n{'='*60}")
                print(f"Tick {i}: 🟢 LONG ENTRY (BREAKOUT CONFIRMED)")
                print(f"{'='*60}")
                print(f"Current Price:    {current_price:.5f}")
                print(f"Entry Price:      {trade_entry_price:.5f}")
                print(f"Support Cluster:  {nearest_cluster:.5f}")
                print(f"Stop Loss:        {stop_loss:.5f} ({SL_BUFFER_PIPS} pips below support)")
                print(f"SL Distance:      {(sl_distance / 0.0001):.1f} pips")
                print(f"Take Profit:      {take_profit:.5f} (STATIC - 4x SL)")
                print(f"TP Distance:      {((take_profit - trade_entry_price) / 0.0001):.1f} pips")
                print(f"Risk:Reward:      1:{TP_RISK_REWARD_RATIO}")
                print(f"\nAll 5 Predictions Above Cluster:")
                for idx, pred in enumerate(predictions):
                    print(f"  Pred[{idx}]: {pred:.5f} (above {nearest_cluster:.5f} ✓)")
                print(f"{'='*60}\n")
                    
            # SHORT ENTRY CONDITIONS:
            # 1. ALL predicted prices must be below the nearest cluster (strong breakout signal)
            # 2. Current price is AT or NEAR the cluster from above (at resistance level)
            # This ensures we're catching a breakout, not consolidation
            elif (np.all(predictions < nearest_cluster) and 
                  is_price_near_cluster(current_price, nearest_cluster, PRICE_NEAR_CLUSTER_PIPS) and
                  current_price >= nearest_cluster - pips_to_price(PRICE_NEAR_CLUSTER_PIPS)):
                
                in_trade = True
                trade_type = 'short'
                trade_entry_price = current_price
                trade_entry_tick = i
                
                # Calculate SL first
                stop_loss = nearest_cluster + pips_to_price(SL_BUFFER_PIPS)
                sl_distance = stop_loss - trade_entry_price
                
                # TP is 4x the SL distance (risk:reward = 1:4)
                take_profit = trade_entry_price - (sl_distance * TP_RISK_REWARD_RATIO)
                breakeven_set = False
                
                print(f"\n{'='*60}")
                print(f"Tick {i}: 🔴 SHORT ENTRY (BREAKOUT CONFIRMED)")
                print(f"{'='*60}")
                print(f"Current Price:         {current_price:.5f}")
                print(f"Entry Price:           {trade_entry_price:.5f}")
                print(f"Resistance Cluster:    {nearest_cluster:.5f}")
                print(f"Stop Loss:             {stop_loss:.5f} ({SL_BUFFER_PIPS} pips above resistance)")
                print(f"SL Distance:           {(sl_distance / 0.0001):.1f} pips")
                print(f"Take Profit:           {take_profit:.5f} (STATIC - 4x SL)")
                print(f"TP Distance:           {((trade_entry_price - take_profit) / 0.0001):.1f} pips")
                print(f"Risk:Reward:           1:{TP_RISK_REWARD_RATIO}")
                print(f"\nAll 5 Predictions Below Cluster:")
                for idx, pred in enumerate(predictions):
                    print(f"  Pred[{idx}]: {pred:.5f} (below {nearest_cluster:.5f} ✓)")
                print(f"{'='*60}\n")

        # --- Plotting ---
        ax.clear()
        plot_start = max(start_index, i - 200)  # Show last 200 candles
        plot_range = range(plot_start, i+1)
        ax.plot(plot_range, data_h1['close'].iloc[plot_start:i+1].values, 
                label='Close Price (H1)', color='blue', linewidth=1.5, zorder=10)
        
        # Plot ALL Cluster Centers as S/R Lines with labels
        for j, center in enumerate(cluster_centers):
            ax.axhline(y=center[0], color='orange', linestyle='--', linewidth=1.5, 
                      alpha=0.7, zorder=3)
            ax.text(plot_start + 5, center[0], f'S/R: {center[0]:.5f}', 
                   fontsize=8, color='orange', va='bottom')
        
        # Highlight nearest cluster
        ax.axhline(y=nearest_cluster, color='purple', linestyle='-', linewidth=2, 
                  label=f'Nearest Cluster: {nearest_cluster:.5f}', zorder=4)
            
        if in_trade:
            # Mark entry point
            entry_x = trade_entry_tick if trade_entry_tick >= plot_start else plot_start
            ax.plot(entry_x, trade_entry_price, 
                   'go' if trade_type == 'long' else 'ro', 
                   markersize=12, label=f'{trade_type.upper()} Entry', zorder=15)
            
            # Show TP and SL lines
            ax.axhline(y=take_profit, color='green', linestyle=':', linewidth=2.5, 
                      label=f'TP (STATIC): {take_profit:.5f}', zorder=5)
            ax.axhline(y=stop_loss, color='red', linestyle='--', linewidth=2.5, 
                      label=f'SL: {stop_loss:.5f}', zorder=5)

        ax.set_title(f'{SYMBOL_TO_SIMULATE} - Tick: {i} | Balance: ${balance:.2f} | Current: {current_price:.5f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Tick Number', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    print("\n" + "="*60)
    print("SIMULATION FINISHED")
    print("="*60)
    print(f"Total Ticks Simulated: {len(data_h1) - start_index}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Net P/L: ${balance - 100:.2f}")
    
    if trade_history:
        trade_df = pd.DataFrame(trade_history)
        print("\n" + "="*60)
        print("TRADE HISTORY")
        print("="*60)
        print(trade_df.to_string(index=False))
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        total_trades = len(trade_df)
        winning_trades = len(trade_df[trade_df['profit_loss'] > 0])
        losing_trades = len(trade_df[trade_df['profit_loss'] <= 0])
        
        print(f"Total Trades:     {total_trades}")
        print(f"Winning Trades:   {winning_trades}")
        print(f"Losing Trades:    {losing_trades}")
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            print(f"Win Rate:         {win_rate:.2f}%")
        
        if winning_trades > 0:
            avg_win = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].mean()
            print(f"Average Win:      ${avg_win:.2f}")
        
        if losing_trades > 0:
            avg_loss = trade_df[trade_df['profit_loss'] <= 0]['profit_loss'].mean()
            print(f"Average Loss:     ${avg_loss:.2f}")
        
        total_pnl = trade_df['profit_loss'].sum()
        print(f"Total P/L:        ${total_pnl:.2f}")
        
        if winning_trades > 0 and losing_trades > 0:
            avg_win = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].mean()
            avg_loss = abs(trade_df[trade_df['profit_loss'] <= 0]['profit_loss'].mean())
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            print(f"Profit Factor:    {profit_factor:.2f}")
        
        print("="*60)
    else:
        print("\n⚠️  No trades were executed during this simulation.")
        print("Consider adjusting entry parameters:")
        print(f"  - SL_BUFFER_PIPS (currently {SL_BUFFER_PIPS})")
        print(f"  - PRICE_NEAR_CLUSTER_PIPS (currently {PRICE_NEAR_CLUSTER_PIPS})")
        print(f"  - N_CLUSTERS (currently {N_CLUSTERS})")

if __name__ == '__main__':
    market_data_h1, lstm_model = load_data_and_model(SYMBOL_TO_SIMULATE)
    run_simulation(market_data_h1, lstm_model)