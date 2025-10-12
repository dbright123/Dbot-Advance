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
N_CLUSTERS = 5
SL_BUFFER_PIPS = 20  # Pips below/above cluster for stop loss (1 pip = 0.0001)
PRICE_NEAR_CLUSTER_PIPS = 10  # How close price needs to be to cluster for entry (in pips)
BREAKEVEN_PROFIT_PIPS = 10  # Pips in profit before moving SL to breakeven
TP_RISK_REWARD_RATIO = 5  # Take profit is 5x the stop loss distance

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
    current_breakeven = 0
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
            # Calculate current profit in pips
            if trade_type == 'long':
                current_profit_pips = (current_price - trade_entry_price) / 0.0001
            else:
                current_profit_pips = (trade_entry_price - current_price) / 0.0001
            
            # Breakeven Logic: Calculate new breakeven using (close price + entry price) / 2
            if current_profit_pips >= BREAKEVEN_PROFIT_PIPS:
                # Calculate new breakeven using your formula
                new_breakeven = (current_price + trade_entry_price) / 2
                
                # For LONG trades: new breakeven must be greater than current breakeven AND less than current price
                if trade_type == 'long':
                    if new_breakeven > current_breakeven and new_breakeven < current_price:
                        current_breakeven = new_breakeven
                        stop_loss = current_breakeven
                        profit_pips = (current_price - trade_entry_price) / 0.0001
                        print(f"Tick {i}: 🛡️ BREAKEVEN (LONG) - In profit by {profit_pips:.1f} pips - SL moved to: {stop_loss:.5f}")
                
                # For SHORT trades: new breakeven must be less than current breakeven AND greater than current price
                elif trade_type == 'short':
                    if (current_breakeven == 0 or new_breakeven < current_breakeven) and new_breakeven > current_price:
                        current_breakeven = new_breakeven
                        stop_loss = current_breakeven
                        profit_pips = (trade_entry_price - current_price) / 0.0001
                        print(f"Tick {i}: 🛡️ BREAKEVEN (SHORT) - In profit by {profit_pips:.1f} pips - SL moved to: {stop_loss:.5f}")

            # --- Exit Conditions (TP/SL) ---
            trade_closed = False
            profit_loss = 0
            exit_reason = ""
            
            if trade_type == 'long':
                if current_price >= take_profit:
                    exit_reason = "TAKE PROFIT HIT (LONG)"
                    profit_loss = (take_profit - trade_entry_price) * 1000 # Use TP for exact calculation
                    trade_closed = True
                elif current_price <= stop_loss:
                    exit_reason = "STOP LOSS HIT (LONG)"
                    profit_loss = (stop_loss - trade_entry_price) * 1000 # Use SL for exact calculation
                    trade_closed = True
            elif trade_type == 'short':
                if current_price <= take_profit:
                    exit_reason = "TAKE PROFIT HIT (SHORT)"
                    profit_loss = (trade_entry_price - take_profit) * 1000 # Use TP for exact calculation
                    trade_closed = True
                elif current_price >= stop_loss:
                    exit_reason = "STOP LOSS HIT (SHORT)"
                    profit_loss = (trade_entry_price - stop_loss) * 1000 # Use SL for exact calculation
                    trade_closed = True

            if trade_closed:
                exit_price = take_profit if "TAKE PROFIT" in exit_reason else stop_loss
                print(f"Tick {i}: *** {exit_reason} at {exit_price:.5f} ***")
                balance += profit_loss
                trade_history.append({
                    'entry_tick': trade_entry_tick,
                    'exit_tick': i,
                    'entry_price': trade_entry_price,
                    'exit_price': exit_price,
                    'trade_type': trade_type,
                    'profit_loss': profit_loss,
                    'balance': balance,
                    'exit_reason': exit_reason
                })
                print(f"P/L: ${profit_loss:.2f} | New Balance: ${balance:.2f}\n")
                in_trade = False
                trade_type = None
                current_breakeven = 0
        
        # --- New Entry Logic (CORRECTED) ---
        if not in_trade:
            # LONG ENTRY (CORRECTED LOGIC):
            # When model predicts ALL above a support cluster, we BUY.
            # Current price is AT or NEAR the cluster.
            if (np.all(predictions > nearest_cluster) and 
                is_price_near_cluster(current_price, nearest_cluster, PRICE_NEAR_CLUSTER_PIPS) and
                current_price >= nearest_cluster - pips_to_price(PRICE_NEAR_CLUSTER_PIPS)):
                
                in_trade = True
                trade_type = 'long' # CORRECT: Buying when model predicts price will go up
                trade_entry_price = current_price
                trade_entry_tick = i
                
                # SL is placed below the support cluster
                stop_loss = nearest_cluster - pips_to_price(SL_BUFFER_PIPS)
                sl_distance = trade_entry_price - stop_loss
                
                # TP is calculated based on the risk:reward ratio
                take_profit = trade_entry_price + (sl_distance * TP_RISK_REWARD_RATIO)
                current_breakeven = 0 # Reset breakeven

                print(f"\n{'='*60}")
                print(f"Tick {i}: 🟢 LONG ENTRY (Following Bullish Prediction)")
                print(f"{'='*60}")
                print(f"Current Price:         {current_price:.5f}")
                print(f"Entry Price:           {trade_entry_price:.5f}")
                print(f"Support Cluster:       {nearest_cluster:.5f}")
                print(f"Stop Loss:             {stop_loss:.5f} ({SL_BUFFER_PIPS} pips below support)")
                print(f"SL Distance:           {(sl_distance / 0.0001):.1f} pips")
                print(f"Take Profit:           {take_profit:.5f} (STATIC - {TP_RISK_REWARD_RATIO}x SL)")
                print(f"TP Distance:           {((take_profit - trade_entry_price) / 0.0001):.1f} pips")
                print(f"Risk:Reward:           1:{TP_RISK_REWARD_RATIO}")
                print(f"\nModel Predictions (ALL ABOVE cluster - we follow by going LONG):")
                for idx, pred in enumerate(predictions):
                    print(f"  Pred[{idx}]: {pred:.5f} (above {nearest_cluster:.5f})")
                print(f"{'='*60}\n")
                    
            # SHORT ENTRY (CORRECTED LOGIC):
            # When model predicts ALL below a resistance cluster, we SELL.
            # Current price is AT or NEAR the cluster.
            elif (np.all(predictions < nearest_cluster) and 
                  is_price_near_cluster(current_price, nearest_cluster, PRICE_NEAR_CLUSTER_PIPS) and
                  current_price <= nearest_cluster + pips_to_price(PRICE_NEAR_CLUSTER_PIPS)):
                
                in_trade = True
                trade_type = 'short' # CORRECT: Selling when model predicts price will go down
                trade_entry_price = current_price
                trade_entry_tick = i
                
                # SL is placed above the resistance cluster
                stop_loss = nearest_cluster + pips_to_price(SL_BUFFER_PIPS)
                sl_distance = stop_loss - trade_entry_price
                
                # TP is calculated based on the risk:reward ratio
                take_profit = trade_entry_price - (sl_distance * TP_RISK_REWARD_RATIO)
                current_breakeven = 0 # Reset breakeven
                
                print(f"\n{'='*60}")
                print(f"Tick {i}: 🔴 SHORT ENTRY (Following Bearish Prediction)")
                print(f"{'='*60}")
                print(f"Current Price:      {current_price:.5f}")
                print(f"Entry Price:        {trade_entry_price:.5f}")
                print(f"Resistance Cluster: {nearest_cluster:.5f}")
                print(f"Stop Loss:          {stop_loss:.5f} ({SL_BUFFER_PIPS} pips above resistance)")
                print(f"SL Distance:        {(sl_distance / 0.0001):.1f} pips")
                print(f"Take Profit:        {take_profit:.5f} (STATIC - {TP_RISK_REWARD_RATIO}x SL)")
                print(f"TP Distance:        {((trade_entry_price - take_profit) / 0.0001):.1f} pips")
                print(f"Risk:Reward:        1:{TP_RISK_REWARD_RATIO}")
                print(f"\nModel Predictions (ALL BELOW cluster - we follow by going SHORT):")
                for idx, pred in enumerate(predictions):
                    print(f"  Pred[{idx}]: {pred:.5f} (below {nearest_cluster:.5f})")
                print(f"{'='*60}\n")

        # --- Plotting ---
        ax.clear()
        plot_start = max(start_index, i - 200)  # Show last 200 candles
        plot_range = range(plot_start, i+1)
        
        # Plot historical close prices
        actual_prices = data_h1['close'].iloc[plot_start:i+1].values
        ax.plot(plot_range, actual_prices, 
                label='Actual Close Price', color='blue', linewidth=2, zorder=10)
        
        # Plot first predicted close price (overlay on the same timeline)
        ax.plot(i, first_pred, 'ro', markersize=8, 
               label='Model 1st Prediction', zorder=11, alpha=0.7)
        
        # Draw a line connecting current price to first prediction for clarity
        ax.plot([i, i], [current_price, first_pred], 'r--', linewidth=1, alpha=0.5)
        
        # Add text showing prediction vs actual
        pred_diff = first_pred - current_price
        pred_diff_pips = pred_diff / 0.0001
        ax.text(i, first_pred, f'  Pred: {first_pred:.5f}\n  ({pred_diff_pips:+.1f} pips)', 
               fontsize=7, color='red', va='center')
        
        # Plot ALL Cluster Centers as S/R Lines with labels
        for j, center in enumerate(cluster_centers):
            ax.axhline(y=center[0], color='orange', linestyle='--', linewidth=1.5, 
                      alpha=0.7, zorder=3)
            ax.text(plot_start + 5, center[0], f'S/R: {center[0]:.5f}', 
                   fontsize=8, color='orange', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        # Highlight nearest cluster
        ax.axhline(y=nearest_cluster, color='purple', linestyle='-', linewidth=2, 
                  label=f'Nearest Cluster: {nearest_cluster:.5f}', zorder=4)
        
        # Add a text box showing prediction status
        pred_status = "ALL predictions ABOVE cluster → LONG signal" if np.all(predictions > nearest_cluster) else \
                      "ALL predictions BELOW cluster → SHORT signal" if np.all(predictions < nearest_cluster) else \
                      "Mixed predictions (no entry signal)"
        
        # Show model accuracy info
        accuracy_text = f'Prediction Status: {pred_status}\n'
        accuracy_text += f'Current: {current_price:.5f}\n'
        accuracy_text += f'1st Pred: {first_pred:.5f} ({pred_diff_pips:+.1f} pips)'
        
        ax.text(0.02, 0.98, accuracy_text, 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
        if in_trade:
            # Mark entry point
            entry_x = trade_entry_tick if trade_entry_tick >= plot_start else plot_start
            ax.plot(entry_x, trade_entry_price, 
                   'go' if trade_type == 'long' else 'ro', 
                   markersize=12, label=f'{trade_type.upper()} Entry', zorder=15)
            
            # Show TP and SL lines
            ax.axhline(y=take_profit, color='green', linestyle=':', linewidth=2.5, 
                      label=f'TP: {take_profit:.5f}', zorder=5)
            ax.axhline(y=stop_loss, color='red', linestyle='--', linewidth=2.5, 
                      label=f'SL: {stop_loss:.5f}', zorder=5)
            
            # Show current breakeven level if active
            if current_breakeven > 0:
                ax.axhline(y=current_breakeven, color='cyan', linestyle='-', linewidth=2, 
                          label=f'Breakeven SL: {current_breakeven:.5f}', zorder=4)
            
            # Show current P/L
            if trade_type == 'long':
                current_pl = (current_price - trade_entry_price) * 1000
                current_pl_pips = (current_price - trade_entry_price) / 0.0001
            else:
                current_pl = (trade_entry_price - current_price) * 1000
                current_pl_pips = (trade_entry_price - current_price) / 0.0001
            
            pl_color = 'green' if current_pl > 0 else 'red'
            
            # Show breakeven status
            breakeven_status = f"Breakeven Active: {current_breakeven:.5f}" if current_breakeven > 0 else "Breakeven: Not set"
            
            ax.text(0.02, 0.88, f'Current P/L: ${current_pl:.2f} ({current_pl_pips:.1f} pips)\n{breakeven_status}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   color=pl_color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'{SYMBOL_TO_SIMULATE} - Tick: {i} | Balance: ${balance:.2f}', 
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
            if avg_loss > 0:
                profit_factor = avg_win / avg_loss
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