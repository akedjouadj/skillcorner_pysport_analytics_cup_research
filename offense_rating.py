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
        self.xt_grids = {
            "left": XTGrid(config.FIELD_LENGTH, config.FIELD_WIDTH, attacking_direction="left"),
            "right": XTGrid(config.FIELD_LENGTH, config.FIELD_WIDTH, attacking_direction="right"),
        }

    
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
            if dist_to_line < config.OD:  # Only consider defenders close to pass line
                obstruction = 1.0 - np.exp(-config.ALPHA * angle / config.BETA)
                defender_factor *= obstruction
        
        probability = distance_factor * defender_factor
        
        return np.clip(probability, 0.0, 1.0)
    
    # Plotting
    def plot_passing_probabilities(self, receivers, passer, defenders):
        """
        Visualize passing probabilities to multiple receivers from a passer in a frame
        """
        
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
        pov_dict = {pid: 0.0 for pid in frame_data["player_id"].values}
        
        # Get possession team
        possession_team_id = frame_data['possession_team_id'].iloc[0]
        
        # Get ball carrier
        ball_carrier = frame_data[frame_data['is_ball_carrier'] == True]
        
        if len(ball_carrier) == 0:
            # No clear ball carrier, return zeros
            return pov_dict
        
        ball_carrier = ball_carrier.iloc[0]
        passer_pos = (ball_carrier['x'], ball_carrier['y'])

         # Attacking direction (once)
        attacking_direction = ball_carrier['attacking_direction']

        # Cached xT grid
        xt_grid = self.xt_grids[attacking_direction]
        
        # Get teammates (excluding ball carrier)
        teammates = frame_data[
            (frame_data['team_id'] == possession_team_id) &
            (frame_data['is_ball_carrier'] == False)
        ]
        
        # Get opponents
        opponents = frame_data[frame_data['team_id'] != possession_team_id]
        defender_positions = opponents[["x", "y"]].to_numpy()
        
        # Loop only over receivers
        for row in teammates.itertuples(index=False):
            receiver_pos = (row.x, row.y)

            # Offside check
            if utils.is_offside(
                receiver_pos,
                passer_pos,
                defender_positions,
                attacking_direction
            ):
                continue

            # Passing probability
            pass_prob = self.calc_passing_probability(
                receiver_pos,
                passer_pos,
                defender_positions
            )

            if pass_prob <= 0.0:
                continue

            # xT value
            xt_value = xt_grid.get_xt_value(row.x, row.y)

            pov_dict[row.player_id] = math.sqrt(pass_prob * xt_value)

        return pov_dict

    def plot_pov(self, home_team_id, away_team_id, frame, pov_dict):
        """
        Visualize Passing Option Values (POV) for all receivers in a frame
        """
        
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
        ax.set_title("Passing Option Values (POV) for Team A")
        plt.tight_layout()
        plt.show()
    
    def get_ocr(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Option Creation Rating (OCR) for all players across entire match
        
        Args:
            tracking_df: Full tracking DataFrame for the match
            
        Returns:
            DataFrame with columns: player_id, team_id, ocr
        """
        # Get unique frames sorted by time
        frames = sorted(tracking_df['frame_id'].unique())

        # downsample to T-second intervals if needed
        period = config.OCR_FREQ
        fps = config.FPS
        frame_interval = int(period * fps)
        frames = frames[::frame_interval]
        
        frame_groups = dict(tuple(tracking_df.groupby("frame_id")))

        player_team_map = (
            tracking_df[["player_id", "team_id"]]
            .drop_duplicates()
            .set_index("player_id")["team_id"]
            .to_dict()
        )
        
        # Initialize accumulator for each player
        ocr_sum = {}
        ocr_count = {}
        ocr_seq = {}
        
        # Process consecutive frames
        print("Calculating Option Creation Rating (OCR) for all players at a 2-second interval...")
        for frame_id in tqdm(frames):
            
            frame_data = frame_groups[frame_id]
            
            # Calculate pov dict for the frame
            pov_dict = self.calc_pov(frame_data)
            
            for player_id, pov in pov_dict.items():

                if player_id not in ocr_sum:
                    ocr_sum[player_id] = 0.0
                    ocr_count[player_id] = 0
                    ocr_seq[player_id] = []

                ocr_sum[player_id] += pov
                ocr_count[player_id] += 1
                ocr_seq[player_id].append(pov)
        
         # --- Build output ---
        results = []
        results_seq = {}

        for player_id in ocr_sum:
            avg_ocr = ocr_sum[player_id] / max(1, ocr_count[player_id])

            results.append({
                "player_id": player_id,
                "team_id": player_team_map[player_id],
                "avg_option_creation_rating": avg_ocr,
            })

            results_seq[player_id] = {
                "seq_option_creation_rating": ocr_seq[player_id]
            }

        return pd.DataFrame(results), results_seq
    
if __name__ == "__main__":
    
    # Example usage: plot passing option values for a single frame

    frame = pd.read_csv("data/match_1886347/frame.csv")

    team_A_id = frame[frame["is_ball_carrier"] == True]["team_id"].values[0]
    team_B_id = frame[frame["team_id"] != team_A_id]["team_id"].values[0]
    
    offense_rating = OffenseRating(config.FIELD_LENGTH)
    pov_dict = offense_rating.calc_pov(frame) 
    offense_rating.plot_pov(home_team_id=team_A_id,
                            away_team_id=team_B_id,
                            frame=frame,
                            pov_dict=pov_dict)
    