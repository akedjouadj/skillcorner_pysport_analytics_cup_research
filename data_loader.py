import pandas as pd
import json
from kloppy import skillcorner
from tqdm import tqdm
from typing import Dict

import config
import utils

def load_and_process_tracking_df(match_id: int) -> pd.DataFrame:
    """
    Load and process tracking data for a given match from SkillCorner
    """
    
    print(f"Loading tracking data from skillcorner for match {match_id}...")
    dataset_raw = skillcorner.load_open_data(
        match_id=match_id,
        coordinates="skillcorner",  # or specify a different coordinate system
    )
    
    dataset = dataset_raw.to_df(engine="pandas")

    print("Converting tracking data in OffBR calculation format. This might take up to 20 seconds...")
    print(f"Processing {len(dataset)} frames...")
    player_tracking_df_ids = [int(pid.split('_')[0]) for pid in dataset.columns[9:]]
    player_tracking_df_ids = list(set(player_tracking_df_ids))
    
    match = json.load(open(f"data/match_{match_id}/match.json", encoding='utf-8'))
    home_team_id = match["home_team"]["id"]
    away_team_id = match["away_team"]["id"]
    
    home_team_player_ids = [player["id"] for player in match["players"] if player["id"] in player_tracking_df_ids and player["team_id"] == home_team_id]
    
    tracking_df = []
    total_frames = 0
    
    for _, frame in tqdm(dataset.iterrows()):
        
        frame_id = frame["frame_id"]
        time_stamp = frame["timestamp"]
        period = frame["period_id"]
        ball_x = frame["ball_x"]
        ball_y = frame["ball_y"]
        possession_team_id = frame["ball_owning_team_id"]
        ball_carrier_id = utils.find_ball_carrier(frame, match, player_tracking_df_ids)

        # prepare attacking direction
        attacking_directions = match["home_team_side"]
        if period == 1:
            attacking_direction = attacking_directions[0]
        else:
            attacking_direction = attacking_directions[1]
        
        if attacking_direction == "right_to_left":
            attacking_direction = "left"
        elif attacking_direction == "left_to_right":
            attacking_direction = "right"
            
        for player in match["players"]:
            
            if player["id"] not in player_tracking_df_ids:
                continue
            
            player_id = player["id"]

            team_id = home_team_id if player_id in home_team_player_ids else away_team_id
            # change attacking direction for away team
            if team_id == away_team_id:
                attacking_direction = "left" if attacking_direction == "right" else "right" 
            
            x = frame[f"{player_id}_x"]
            y = frame[f"{player_id}_y"]
            
            tracking_df.append({
                "frame_id": frame_id,
                "time_stamp": time_stamp,
                "period": period,
                "attacking_direction": attacking_direction,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "team_id": team_id,
                "player_id": player_id,
                "x": x,
                "y": y,
                "possession_team_id": possession_team_id,
                "is_ball_carrier": player_id == ball_carrier_id
            })
    
        total_frames += 1
        
    tracking_df = pd.DataFrame(tracking_df)

    # adjust coordinates to have (0,0) at the bottom left corner
    tracking_df["x"] = tracking_df["x"] + config.FIELD_LENGTH / 2
    tracking_df["y"] = tracking_df["y"] + config.FIELD_WIDTH / 2

    # drop rows with missing values
    tracking_df = tracking_df.dropna().reset_index(drop=True)

    return tracking_df

def load_all_matches_outputs(match_dataframes: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all match output dataframes with match_id
    
    Args:
        match_dataframes: List of DataFrames, one per match
        
    Returns:
        Combined DataFrame with match_id column
    """
    all_data = []
    for match_id, df in match_dataframes.items():
        df_copy = df.copy()
        df_copy['match_id'] = match_id
        all_data.append(df_copy)
    
    return pd.concat(all_data, ignore_index=True)