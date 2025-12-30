import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ALERT_CSV, ZONE_NAMES

# Keep track of detection counts for each zone
zone_counts = {z: [] for z in ZONE_NAMES}

# Initialize with some default data
for zone in ZONE_NAMES:
    zone_counts[zone] = [0] * 10  # Initialize with 10 zero counts

def log_alerts(alerts):
    """Append alerts to CSV log."""
    if not alerts:
        return
    df = pd.DataFrame(alerts)
    try:
        df.to_csv(
            ALERT_CSV,
            mode="a",
            header=not pd.io.common.file_exists(ALERT_CSV),
            index=False
        )
    except Exception as e:
        print("Error writing alerts log:", e)

def update_heatmap(detections_per_zone):
    """Update count history for each zone."""
    for zone in ZONE_NAMES:
        count = len(detections_per_zone.get(zone, []))
        zone_counts[zone].append(count)
        # Keep only last 100 readings
        if len(zone_counts[zone]) > 100:
            zone_counts[zone].pop(0)

def get_zone_counts():
    """Return the latest count for each zone as a list."""
    return [zone_counts[z][-1] if zone_counts[z] else 0 for z in ZONE_NAMES]

# -------- HEATMAP --------
def get_heatmap_matrix():
    """Convert zone counts into a 2x3 matrix for heatmap."""
    counts = get_zone_counts()
    return np.array(counts).reshape(2, 3)

def plot_heatmap():
    """Generate a matplotlib heatmap figure and return it."""
    plt.close('all')
    
    matrix = get_heatmap_matrix()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    fig.set_facecolor("#0D0D1A")
    ax.set_facecolor("#0D0D1A")

    sns.heatmap(
        matrix,
        annot=True, fmt="d",
        cmap="magma",
        cbar=True,
        linewidths=2,
        linecolor="white",
        xticklabels=ZONE_NAMES[:3],
        yticklabels=ZONE_NAMES[3:],
        annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'},
        cbar_kws={'label': 'Number of People'}
    )

    ax.set_title("People Density Heatmap", fontsize=18, color="white", pad=20)
    ax.tick_params(colors="white", labelsize=12)
    
    return fig

# -------- BAR CHART --------
def plot_zone_bars():
    """Generate a bar chart showing people count per zone."""
    plt.close('all')
    
    counts = get_zone_counts()
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    fig.set_facecolor("#0D0D1A")
    ax.set_facecolor("#0D0D1A")

    # Create gradient colors
    colors = sns.color_palette("husl", len(ZONE_NAMES))
    bars = ax.bar(ZONE_NAMES, counts, color=colors, width=0.6)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}',
            ha='center', va='bottom',
            color='white', fontweight='bold', fontsize=12
        )

    ax.set_title("People Count per Zone", fontsize=18, color="white", pad=20)
    ax.set_xlabel("Zones", fontsize=14, color="white", labelpad=10)
    ax.set_ylabel("Count", fontsize=14, color="white", labelpad=10)
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(colors="white", labelsize=12)
    return fig
