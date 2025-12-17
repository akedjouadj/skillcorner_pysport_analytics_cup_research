"""
Offensive rating calculation: Option Creation Rating (OCR)
Measures how well players create passing options through movement
"""

import numpy as np
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Circle, Rectangle, Arc
from typing import Dict, Tuple, List

import config
import utils
from xt_grid import XTGrid


class OffenseRating:
    """
    Calculate Option Creation Rating (OCR) for off-ball offensive movements
    """
    
    def __init__(self, field_length: float):
        """
        Initialize offense rating calculator
        
        Args:
            field_length: Length of the field in meters
        """
        self.field_length = field_length
    
    def calc_passing_probability(self,
                                   receiver_pos: Tuple[float, float],
                                   passer_pos: Tuple[float, float],
                                   defenders: List[Tuple[float, float]]) -> float:
        """
        Calculate probability of successful pass reception using geometric proxy
        
        P(receive) ∝ exp(-d(P,B)/d_max) × Π[defenders](1 - exp(-α·angle/β))
        
        Args:
            receiver_pos: Position of potential receiver (x, y)
            passer_pos: Position of ball carrier (x, y)
            defenders: List of defender positions [(x, y), ...]
            
        Returns:
            Probability value between 0 and 1
        """
        # Distance component
        dist = utils.distance(receiver_pos, passer_pos)
        
        if dist < 0.1:  # Too close
            return 0.0
        
        distance_factor = np.exp(-dist / config.D_MAX)
        
        # Defender obstruction component
        defender_factor = 1.0
        
        for defender_pos in defenders:
            # Calculate angle at receiver between passer and defender
            angle = utils.angle_between_three_points(
                passer_pos, receiver_pos, defender_pos
            )
            
            # Distance from defender to pass line
            dist_to_line = utils.point_to_line_distance(
                defender_pos, passer_pos, receiver_pos
            )
            
            # Defender impact (higher impact when close to line and small angle)
            if dist_to_line < 3.0:  # Only consider defenders close to pass line
                obstruction = 1.0 - np.exp(-config.ALPHA * angle / config.BETA)
                defender_factor *= obstruction
        
        probability = distance_factor * defender_factor
        
        return np.clip(probability, 0.0, 1.0)
    
    # Plotting
    def plot_passing_probabilities(self, receivers, passer, defenders):
        probs = [self.calc_passing_probability(r, passer, defenders) for r in receivers]

        fig, ax = plt.subplots(figsize=(12, 7))
        utils.draw_pitch(ax)

        # Team A receivers (green circles)
        for r, p in zip(receivers, probs):
            ax.scatter(r[0], r[1], c="green", s=80)
            ax.text(r[0], r[1]+1, f"{p:.2f}", color="red", ha="center", fontsize=15)

            # Draw pass lines
            ax.plot([passer[0], r[0]], [passer[1], r[1]], color="gray", linestyle="--", alpha=0.4)

        # Passer (green square)
        ax.scatter(passer[0], passer[1], c="green", s=140, marker="s", label="Passeur")

        # Defenders (blue circles)
        for d in defenders:
            ax.scatter(d[0], d[1], c="blue", s=80)

        # Make legend
        ax.scatter([], [], c="green", s=80, marker="o", label="Team A (receivers)")
        ax.scatter([], [], c="blue", s=80, marker="o", label="Team B (defenders)")
        ax.legend(loc="upper right")

        ax.set_xlim(0, config.FIELD_LENGTH)
        ax.set_ylim(0, config.FIELD_WIDTH)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

    def calc_pov(self, frame_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Passing Option Value (POV) for all players in a frame
        
        POV = (P(receive) × xT(receiver_position))^(1/2) # geometric mean
        
        Args:
            frame_data: DataFrame containing all players in one frame
            
        Returns:
            Dictionary mapping player_id to POV value
        """
        pov_dict = {}
        
        # Get possession team
        possession_team_id = frame_data['possession_team_id'].iloc[0]
        
        # Get ball carrier
        ball_carrier = frame_data[frame_data['is_ball_carrier'] == True]
        
        if len(ball_carrier) == 0:
            # No clear ball carrier, return zeros
            return {pid: 0.0 for pid in frame_data['player_id'].unique()}
        
        ball_carrier = ball_carrier.iloc[0]
        passer_pos = (ball_carrier['x'], ball_carrier['y'])
        
        # Get teammates (excluding ball carrier)
        teammates = frame_data[
            (frame_data['team_id'] == possession_team_id) &
            (frame_data['is_ball_carrier'] == False)
        ]
        
        # Get opponents
        opponents = frame_data[frame_data['team_id'] != possession_team_id]
        defender_positions = [(row['x'], row['y']) for _, row in opponents.iterrows()]
        
        # Calculate POV for each teammate
        for _, player in teammates.iterrows():
            receiver_pos = (player['x'], player['y'])
            
            # Check for offside
            attacking_direction = 'right' if passer_pos[0] < self.field_length / 2 else 'left'
            if utils.is_offside(receiver_pos, passer_pos, defender_positions, attacking_direction):
                pov_dict[player['player_id']] = 0.0
                continue
            
            # Calculate passing probability
            pass_prob = self.calc_passing_probability(
                receiver_pos, passer_pos, defender_positions
            )
            
            # Get xT value at receiver position
            xt_grid = XTGrid(config.FIELD_LENGTH, config.FIELD_WIDTH, attacking_direction=attacking_direction)
            xt_value = xt_grid.get_xt_value(receiver_pos[0], receiver_pos[1])
            
            # POV
            pov_dict[player['player_id']] = math.sqrt(pass_prob * xt_value)
        
        # Set POV to 0 for ball carrier and opponents
        for _, player in frame_data.iterrows():
            if player['player_id'] not in pov_dict:
                pov_dict[player['player_id']] = 0.0
        
        return pov_dict

    def plot_pov(self, home_team_id, away_team_id, frame, pov_dict):
        fig, ax = plt.subplots(figsize=(12,7))
        utils.draw_pitch(ax)

        teamA = frame[frame.team_id==home_team_id]
        teamB = frame[frame.team_id==away_team_id]
        passer = frame[frame.is_ball_carrier==True].iloc[0]

        # receivers
        for _, r in teamA.iterrows():
            if r.player_id == passer.player_id:
                continue
            ax.scatter(r.x, r.y, c="green", s=80)
            pov = pov_dict[r.player_id]
            ax.text(r.x, r.y+1, f"{pov:.3f}", color='red', ha="center")

            # pass line
            ax.plot([passer.x, r.x], [passer.y, r.y], color="gray", linestyle="--", alpha=0.4)

        # passer
        ax.scatter(passer.x, passer.y, c="green", s=140, marker="s", label="Passeur")

        # defenders
        for _, d in teamB.iterrows():
            ax.scatter(d.x, d.y, c="blue", s=80)

        ax.scatter([],[],c="green",s=80,label="Team A (receivers)")
        ax.scatter([],[],c="blue",s=80,label="Team B (defenders)")
        ax.legend()

        ax.set_xlim(0, config.FIELD_LENGTH)
        ax.set_ylim(0, config.FIELD_WIDTH)
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
    
    def calc_movement_threat(self,
                             frame_t: pd.DataFrame,
                             frame_t_minus_1: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate movement threat as difference in POV between consecutive frames
        
        Movement Threat = POV_t - POV_{t-1}
        
        Args:
            frame_t: Current frame data
            frame_t_minus_1: Previous frame data
            
        Returns:
            Dictionary mapping player_id to movement threat value
        """
        pov_t = self.calc_pov(frame_t)
        pov_t_minus_1 = self.calc_pov(frame_t_minus_1)
        
        movement_threat = {}
        
        for player_id in pov_t.keys():
            if player_id in pov_t_minus_1:
                threat = pov_t[player_id] - pov_t_minus_1[player_id]
                # Only count positive movements (improvements in option value)
                movement_threat[player_id] = max(0.0, threat)
            else:
                movement_threat[player_id] = 0.0
        
        return movement_threat
    
    def get_ocr(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Option Creation Rating (OCR) for all players across entire match
        
        Args:
            tracking_df: Full tracking DataFrame for the match
            
        Returns:
            DataFrame with columns: player_id, team_id, ocr, playing_time
        """
        # Get unique frames sorted by time
        frames = sorted(tracking_df['frame_id'].unique())

        # downsample to T-second intervals if needed
        period = config.T
        fps = config.FPS
        frame_interval = int(period * fps)
        frames = frames[::frame_interval]
        
        # Initialize accumulator for each player
        ocr_accumulator = {}
        playing_time = {}
        team_mapping = {}
        
        # Process consecutive frames
        print("Calculating Option Creation Rating (OCR) for all players at a 2-second interval...")
        for frame_id in tqdm(frames):
            
            frame_data = tracking_df[tracking_df['frame_id'] == frame_id]
            
            # Calculate pov dict for the frame
            pov_dict = self.calc_pov(frame_data)
            
            for player_id, pov_value in pov_dict.items():
                if player_id not in ocr_accumulator:
                    ocr_accumulator[player_id] = []
                    playing_time[player_id] = 0.0
                    
                    # Store team mapping
                    player_team = frame_data[frame_data['player_id'] == player_id]['team_id'].iloc[0]
                    team_mapping[player_id] = player_team
                
                ocr_accumulator[player_id].append(pov_value)
                playing_time[player_id] = frame_id/fps/60 # in minutes
        
        # Create result DataFrame
        results_df = []
        results_seq = {}
        for player_id, pov_values in ocr_accumulator.items():
            avg_pov = np.mean(pov_values)
            # Normalize by playing time (per 90 minutes)
            time_played = playing_time[player_id]
            
            results_df.append({
                'player_id': player_id,
                'team_id': team_mapping[player_id],
                'avg_option_creation_rating': avg_pov,
                'playing_time': time_played
            })

            results_seq[player_id] = {'seq_option_creation_rating': pov_values}
                
        
        return pd.DataFrame(results_df), results_seq
    
if __name__ == "__main__":
    
    tracking_df = pd.read_csv("data/match_1886347/tracking_df.csv")
    
    # translater les coordonnées pour que l'origine soit au coin bas gauche et non au centre
    tracking_df["x"] = tracking_df["x"] + config.FIELD_LENGTH / 2
    tracking_df["y"] = tracking_df["y"] + config.FIELD_WIDTH / 2

    """frame = tracking_df[tracking_df["frame_id"]==1000].copy()

    # check if a player is in possession of the ball
    if frame["is_ball_carrier"].sum() == 0:
        raise ValueError("No player in possession of the ball in this frame.")


    # translater les coordonnées pour que l'origine soit au coin bas gauche et non au centre
    frame["x"] = frame["x"] + config.FIELD_LENGTH / 2
    frame["y"] = frame["y"] + config.FIELD_WIDTH / 2

    passer = frame[frame["is_ball_carrier"] == True][["x", "y"]].values[0]
    team_A_id = frame[frame["is_ball_carrier"] == True]["team_id"].values[0]
    team_B_id = frame[frame["team_id"] != team_A_id]["team_id"].values[0]
    receivers = frame[(frame["team_id"] == team_A_id) & (frame["is_ball_carrier"] == False)][["x", "y"]].values
    defenders= frame[frame["team_id"] == team_B_id][["x", "y"]].values

    offense_rating = OffenseRating(config.FIELD_LENGTH)
    #offense_rating.plot_passing_probabilities(receivers, passer, defenders)
    pov_dict = offense_rating.calc_pov(frame) 
    offense_rating.plot_pov(home_team_id=team_A_id,
                            away_team_id=team_B_id,
                            frame=frame,
                            pov_dict=pov_dict)"""
    
    offense_rating = OffenseRating(config.FIELD_LENGTH)
    ocr_df = offense_rating.get_ocr(tracking_df)
    print(ocr_df)
    