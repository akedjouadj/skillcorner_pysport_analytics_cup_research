import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def add_metatdata_to_off_ball_rating(off_ball_rating_df, match_id):
    """
    Enrich OffBR output DataFrame with player names, positions, team names, and match outcomes
    """
    
    match_metadata = json.load(open(f"data/match_{match_id}/match.json", encoding='utf-8'))

    # add player name and position
    player_id_name_map = {}
    player_id_position_map = {}
    player_id_position_group_map = {}
    player_id_playing_time_map = {}
    rated_player_ids = off_ball_rating_df['player_id'].unique().tolist()
    for player in match_metadata['players']:
        if player['id'] in rated_player_ids:
            player_id_name_map[player['id']] = f"{player['first_name']} {player['last_name']}"
            player_id_position_map[player['id']] = player['player_role']['name']
            player_id_position_group_map[player['id']] = player['player_role']['position_group']
            player_id_playing_time_map[player['id']] = player['playing_time']['total']['minutes_played']

    off_ball_rating_df.insert(1, 'player_name', off_ball_rating_df['player_id'].map(player_id_name_map))
    off_ball_rating_df.insert(2, 'position', off_ball_rating_df['player_id'].map(player_id_position_map))
    off_ball_rating_df.insert(3, 'position_group', off_ball_rating_df['player_id'].map(player_id_position_group_map))
    off_ball_rating_df.insert(4, 'playing_time', off_ball_rating_df['player_id'].map(player_id_playing_time_map))
    
    # add team name
    home_team_id = match_metadata['home_team']['id']
    home_team_name = match_metadata['home_team']['name']
    away_team_id = match_metadata['away_team']['id']
    away_team_name = match_metadata['away_team']['name']
    team_id_name_map = {
        home_team_id: home_team_name,
        away_team_id: away_team_name
    }
    off_ball_rating_df.insert(5, 'team_name', off_ball_rating_df['team_id'].map(team_id_name_map))

    # add match outcome
    home_team_score = match_metadata['home_team_score']
    away_team_score = match_metadata['away_team_score']
    team_id_outcome_map = {}
    if home_team_score > away_team_score:
        team_id_outcome_map[home_team_id] = 'W'
        team_id_outcome_map[away_team_id] = 'L'
    elif home_team_score < away_team_score:
        team_id_outcome_map[home_team_id] = 'L'
        team_id_outcome_map[away_team_id] = 'W'
    else:
        team_id_outcome_map[home_team_id] = 'D'
        team_id_outcome_map[away_team_id] = 'D'

    off_ball_rating_df.insert(6, 'match_outcome', off_ball_rating_df['team_id'].map(team_id_outcome_map))

    return off_ball_rating_df


# ==================== VALIDATION 1: TACTICAL COHERENCE ====================

def validate_tactical_coherence(combined_df: pd.DataFrame) -> Dict:
    """
    Validate OffBR by position: Do attackers have higher OCR? Defenders higher SCR?
    
    Args:
        combined_df: Combined DataFrame with all matches
        
    Returns:
        Dictionary with statistics and insights
    """
    print("=" * 80)
    print("VALIDATION 1: TACTICAL COHERENCE")
    print("=" * 80)
    
    # Group by position
    position_stats = combined_df.groupby('position_group').agg({
        'avg_option_creation_rating': ['mean', 'std', 'count'],
        'avg_space_control_rating': ['mean', 'std', 'count']
    }).round(4)
    
    print("\n📊 Average Ratings by Position Group:")
    print(position_stats)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # OCR by position
    ax1 = axes[0]
    position_order = combined_df.groupby('position_group')['avg_option_creation_rating'].mean().sort_values(ascending=False).index
    sns.boxplot(data=combined_df, y='position_group', x='avg_option_creation_rating', 
                order=position_order, ax=ax1, palette='Blues_d')
    ax1.set_xlabel('Option Creation Rating (OCR)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Position Group', fontsize=12, fontweight='bold')
    ax1.set_title('OCR Distribution by Position\n(Higher = Better Off-Ball Attacking Movement)', 
                  fontsize=13, fontweight='bold')
    ax1.axvline(combined_df['avg_option_creation_rating'].mean(), color='red', 
                linestyle='--', alpha=0.5, label='Overall Mean')
    ax1.legend()
    
    # SCR by position
    ax2 = axes[1]
    position_order_scr = combined_df.groupby('position_group')['avg_space_control_rating'].mean().sort_values(ascending=False).index
    sns.boxplot(data=combined_df, y='position_group', x='avg_space_control_rating', 
                order=position_order_scr, ax=ax2, palette='Reds_d')
    ax2.set_xlabel('Space Control Rating (SCR)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Position Group', fontsize=12, fontweight='bold')
    ax2.set_title('SCR Distribution by Position\n(Higher = Better Defensive Positioning)', 
                  fontsize=13, fontweight='bold')
    ax2.axvline(combined_df['avg_space_control_rating'].mean(), color='red', 
                linestyle='--', alpha=0.5, label='Overall Mean')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/tactical_coherence_validation.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: outputs/plots/tactical_coherence_validation.png")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    
    # Check if forwards have highest OCR
    ocr_by_pos = combined_df.groupby('position_group')['avg_option_creation_rating'].mean().sort_values(ascending=False)
    top_ocr_position = ocr_by_pos.index[0]
    print(f"  • Highest OCR: {top_ocr_position} ({ocr_by_pos.iloc[0]:.4f})")
    
    # Check if defenders have highest SCR
    scr_by_pos = combined_df.groupby('position_group')['avg_space_control_rating'].mean().sort_values(ascending=False)
    top_scr_position = scr_by_pos.index[0]
    print(f"  • Highest SCR: {top_scr_position} ({scr_by_pos.iloc[0]:.4f})")
    
    # Check midfielders balance
    for position in combined_df['position_group'].unique():
        mid_ocr = combined_df[combined_df['position_group'] == position]['avg_option_creation_rating'].mean()
        mid_scr = combined_df[combined_df['position_group'] == position]['avg_space_control_rating'].mean()
        balance_ratio = min(mid_ocr, mid_scr) / max(mid_ocr, mid_scr)
        sample_size = combined_df[combined_df['position_group'] == position].shape[0]
        print(f"  • position: {position}, sample size: {sample_size} balance ratio: {balance_ratio:.2f} (closer to 1.0 = more balanced)")
    
    return {
        'position_stats': position_stats,
        'top_ocr_position': top_ocr_position,
        'top_scr_position': top_scr_position
    }

# ==================== VALIDATION 2: TEMPORAL EVOLUTION ====================

def analyze_temporal_evolution(combined_df: pd.DataFrame, 
                                match_seqs: Dict[int, dict]) -> Dict:
    """
    Analyze temporal evolution of OffBR during matches (fatigue effect)
    
    Args:
        combined_df: Combined DataFrame with all matches
        sample_size: Number of players to show in detailed plots
        
    Returns:
        Dictionary with statistics and insights
    """
    
    print("\n" + "=" * 80)
    print("VALIDATION 2: TEMPORAL EVOLUTION (FATIGUE ANALYSIS)")
    print("=" * 80)
    
    # Analyze OCR evolution
    print("\n📈 Analyzing OCR temporal patterns...")
    
    all_ocr_trends = []
    all_scr_trends = []
    
    for idx, row in combined_df.iterrows():
        match_id = row['match_id']
        player_id = row['player_id']
        ocr_seq = match_seqs[match_id][player_id]['seq_option_creation_rating']
        scr_seq = match_seqs[match_id][player_id]['seq_space_control_rating']
        
        if len(ocr_seq) > 10:  # Minimum sequence length
            # Normalize time to percentage of match
            time_pct = np.linspace(0, 100, len(ocr_seq))
            
            # Split into bins (e.g., 0-15min, 15-30min, etc.)
            bins = [0, 25, 50, 75, 100]
            for i in range(len(bins) - 1):
                mask = (time_pct >= bins[i]) & (time_pct < bins[i+1])
                if np.sum(mask) > 0:
                    all_ocr_trends.append({
                        'player_id': row['player_id'],
                        'position_group': row['position_group'],
                        'period': f"{bins[i]}-{bins[i+1]}%",
                        'avg_ocr': np.mean(np.array(ocr_seq)[mask]),
                        'period_num': i
                    })
        
        if len(scr_seq) > 10:
            time_pct = np.linspace(0, 100, len(scr_seq))
            bins = [0, 25, 50, 75, 100]
            for i in range(len(bins) - 1):
                mask = (time_pct >= bins[i]) & (time_pct < bins[i+1])
                if np.sum(mask) > 0:
                    all_scr_trends.append({
                        'player_id': row['player_id'],
                        'position_group': row['position_group'],
                        'period': f"{bins[i]}-{bins[i+1]}%",
                        'avg_scr': np.mean(np.array(scr_seq)[mask]),
                        'period_num': i
                    })
    
    ocr_trends_df = pd.DataFrame(all_ocr_trends)
    scr_trends_df = pd.DataFrame(all_scr_trends)
    
    # Aggregate by period
    ocr_by_period = ocr_trends_df.groupby('period_num')['avg_ocr'].agg(['mean', 'std']).reset_index()
    scr_by_period = scr_trends_df.groupby('period_num')['avg_scr'].agg(['mean', 'std']).reset_index()
    
    print("\n📊 Average OCR by Match Period:")
    for _, row in ocr_by_period.iterrows():
        period_label = ['0-25%', '25-50%', '50-75%', '75-100%'][int(row['period_num'])]
        print(f"  {period_label}: {row['mean']:.4f} (±{row['std']:.4f})")
    
    print("\n📊 Average SCR by Match Period:")
    for _, row in scr_by_period.iterrows():
        period_label = ['0-25%', '25-50%', '50-75%', '75-100%'][int(row['period_num'])]
        print(f"  {period_label}: {row['mean']:.4f} (±{row['std']:.4f})")
    
    # Calculate trend (linear regression)
    ocr_slope, ocr_intercept, ocr_pearson_corr, _, _ = stats.linregress(
        ocr_by_period['period_num'], ocr_by_period['mean']
    )
    scr_slope, scr_intercept, scr_pearson_corr, _, _ = stats.linregress(
        scr_by_period['period_num'], scr_by_period['mean']
    )
    
    print(f"\n🔬 Temporal Trend Analysis:")
    print(f"  OCR trend: slope={ocr_slope:.6f}, pearson_corr={ocr_pearson_corr:.3f}")
    print(f"  SCR trend: slope={scr_slope:.6f}, pearson_corr={scr_pearson_corr:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    
    # OCR temporal evolution (aggregate)
    ax1 = axes[0]
    period_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    ax1.plot(ocr_by_period['period_num'], ocr_by_period['mean'], 
             marker='o', linewidth=2, markersize=10, color='steelblue', label='Mean OCR')
    ax1.fill_between(ocr_by_period['period_num'], 
                      ocr_by_period['mean'] - ocr_by_period['std'],
                      ocr_by_period['mean'] + ocr_by_period['std'],
                      alpha=0.3, color='steelblue')
    ax1.set_xlabel('Match Period', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Option Creation Rating', fontsize=12, fontweight='bold')
    ax1.set_title('OCR Evolution During Match\n(Aggregated Across All Players)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(period_labels)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # SCR temporal evolution (aggregate)
    ax2 = axes[1]
    ax2.plot(scr_by_period['period_num'], scr_by_period['mean'], 
             marker='o', linewidth=2, markersize=10, color='crimson', label='Mean SCR')
    ax2.fill_between(scr_by_period['period_num'], 
                      scr_by_period['mean'] - scr_by_period['std'],
                      scr_by_period['mean'] + scr_by_period['std'],
                      alpha=0.3, color='crimson')
    ax2.set_xlabel('Match Period', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Space Control Rating', fontsize=12, fontweight='bold')
    ax2.set_title('SCR Evolution During Match\n(Aggregated Across All Players)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(period_labels)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/temporal_evolution_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: outputs/plots/temporal_evolution_analysis.png")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    
    if abs(ocr_slope) > 0.0001:
        direction = "increases" if ocr_slope > 0 else "decreases"
        print(f"  • OCR {direction} over match duration (slope={ocr_slope:.6f})")
    else:
        print(f"  • OCR remains relatively stable throughout the match")
    
    if abs(scr_slope) > 0.0001:
        direction = "increases" if scr_slope > 0 else "decreases"
        print(f"  • SCR {direction} over match duration (slope={scr_slope:.6f})")
    else:
        print(f"  • SCR remains relatively stable throughout the match")
    
    return {
        'ocr_by_period': ocr_by_period,
        'scr_by_period': scr_by_period,
        'ocr_trend': {'slope': ocr_slope, 'r': ocr_pearson_corr},
        'scr_trend': {'slope': scr_slope, 'r': scr_pearson_corr}
    }
