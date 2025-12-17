"""
Defensive rating calculation: Space Control Rating (SCR)
Measures defensive positioning, zone coverage, and pass line blocking
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import config
import utils
from xt_grid import XTGrid


class DefenseRating:
    """
    Calculate Space Control Rating (SCR) for defensive positioning
    """
    
    def __init__(self, field_length: float, field_width: float):
        """
        Initialize defense rating calculator
        
        Args:
            xt_grid: XTGrid object for spatial valuation
            field_length: Length of the field in meters
            field_width: Width of the field in meters
        """
        self.field_length = field_length
        self.field_width = field_width
        
        # Create grid of cells for zone coverage calculation
        self.grid_step = config.R
        self.n_cells_x = int(np.ceil(field_length / self.grid_step))
        self.n_cells_y = int(np.ceil(field_width / self.grid_step))
    
    def get_cell_coordinates(self, cell_idx_x: int, cell_idx_y: int) -> Tuple[float, float]:
        """
        Get center coordinates of a grid cell
        
        Args:
            cell_idx_x: X index of cell
            cell_idx_y: Y index of cell
            
        Returns:
            Center position (x, y) of the cell
        """
        x = (cell_idx_x + 0.5) * self.grid_step
        y = (cell_idx_y + 0.5) * self.grid_step
        return (x, y)
    
    def identify_danger_zones(self, frame_data: pd.DataFrame):
        """
        Identify danger zones centered on attackers
        """
        possession_team_id = frame_data['possession_team_id'].iloc[0]

        attackers = frame_data[frame_data['team_id'] == possession_team_id]
        
        # Ball carrier (for attacking direction)
        ball_carrier = frame_data[frame_data['is_ball_carrier']].iloc[0]
        attacking_direction = (
            'right' if ball_carrier['x'] < self.field_length / 2 else 'left'
        )

        xt_grid = XTGrid(
            self.field_length,
            self.field_width,
            attacking_direction=attacking_direction
        )

        danger_zones = []

        for _, attacker in attackers.iterrows():
            attacker_pos = (attacker['x'], attacker['y'])

            xt_value = xt_grid.get_xt_value(*attacker_pos)
            if xt_value < config.DANGER_ZONE_XT_THRESHOLD:
                continue

            danger_zones.append({
                "attacker_id": attacker['player_id'],
                "center": attacker_pos,
                "xt_value": xt_value
            })

        return danger_zones
    
    def assign_danger_zone_coverage(self, frame_data):
        """
        Assign defensive coverage to each danger zone
        """
        possession_team_id = frame_data['possession_team_id'].iloc[0]
        defenders = frame_data[frame_data['team_id'] != possession_team_id]

        covered_zones = []
        
        danger_zones = self.identify_danger_zones(frame_data)

        for zone in danger_zones:
            zone_center = zone["center"]

            min_dist = float("inf")
            closest_defender_id = None
            closest_defender_pos = None

            for _, defender in defenders.iterrows():
                d = utils.distance(
                    zone_center,
                    (defender['x'], defender['y'])
                )
                if d < min_dist:
                    min_dist = d
                    closest_defender_id = defender['player_id']
                    closest_defender_pos = (defender['x'], defender['y'])

            covered = min_dist < self.grid_step

            covered_zones.append({
                **zone,
                "covered": covered,
                "defender_id": closest_defender_id if covered else None,
                "defender_pos": closest_defender_pos if covered else None,
                "defender_distance": min_dist
            })

        return covered_zones
    
    def calc_zone_coverage(self, frame_data):
        """
        Aggregate zone coverage per defender
        """

        possession_team_id = frame_data['possession_team_id'].iloc[0]
        defenders = frame_data[frame_data['team_id'] != possession_team_id]

        coverage = {pid: 0.0 for pid in defenders['player_id']}

        covered_zones = self.assign_danger_zone_coverage(frame_data)
    
        if len(covered_zones) == 0:
            return coverage

        for zone in covered_zones:
            if zone["covered"]:
                coverage[zone["defender_id"]] += 1

        total_zones = len(covered_zones)
        for pid in coverage:
            coverage[pid] /= total_zones

        return coverage

    def plot_danger_zone_coverage(
        self,
        frame_data: pd.DataFrame
    ):
        """
        Plot danger zones and defensive coverage for a single frame
        (player-centric, refactored version)
        """

        possession_team_id = frame_data['possession_team_id'].iloc[0]

        attackers = frame_data[frame_data['team_id'] == possession_team_id]
        defenders = frame_data[frame_data['team_id'] != possession_team_id]

        # Ball carrier
        ball_carrier = frame_data[frame_data['is_ball_carrier']].iloc[0]
        ball_pos = (ball_carrier['x'], ball_carrier['y'])

        # --- Compute danger zones (refactored pipeline) ---
        covered_zones = self.assign_danger_zone_coverage(frame_data)
        coverage_ratio = self.calc_zone_coverage(frame_data)

        # --- Plot setup ---
        fig, ax = plt.subplots(figsize=(12, 7))
        utils.draw_pitch(ax)

        # --- Players ---
        # Attackers
        ax.scatter(
            attackers['x'],
            attackers['y'],
            c='darkgreen',
            s=70,
            label='Receivers'
        )

        # Defenders
        ax.scatter(
            defenders['x'],
            defenders['y'],
            c='darkblue',
            s=70,
            label='Defenders'
        )

        # Ball carrier
        ax.scatter(
            ball_pos[0],
            ball_pos[1],
            c='darkgreen',
            s=140,
            marker='s',
            label='Ball carrier'
        )

        # print coverage ratio per defender
        for _, defender in defenders.iterrows():
            pid = defender['player_id']
            ratio = coverage_ratio.get(pid, 0.0)

            ax.text(
                defender['x'],
                defender['y'] + 1.2,
                f"{ratio:.2f}",
                color='red',
                fontsize=15,
                ha='center',
                va='bottom',
                zorder=5
            )

        # --- Danger zones ---
        for zone in covered_zones:
            center = zone["center"]
            covered = zone["covered"]

            # Zone color
            zone_color = 'blue' if covered else 'green'

            # Danger zone circle (centered on attacker)
            ax.add_patch(
                Circle(
                    center,
                    radius=self.grid_step,
                    color=zone_color,
                    alpha=0.25,
                    zorder=1
                )
            )

            # --- Pass line (ball carrier → attacker) ---
            ax.plot(
                [ball_pos[0], center[0]],
                [ball_pos[1], center[1]],
                linestyle='--',
                color='gray',
                alpha=0.5,
                zorder=2
            )

            # --- Coverage line (attacker → defender) ---
            if covered:
                defender_pos = zone["defender_pos"]

                ax.plot(
                    [center[0], defender_pos[0]],
                    [center[1], defender_pos[1]],
                    linestyle='--',
                    color='blue',
                    alpha=0.6,
                    zorder=3
                )

        # --- Legend ---
        ax.add_patch(Circle((0, 0), 1, color='green', alpha=0.25, label='Uncovered danger zone'))
        ax.add_patch(Circle((0, 0), 1, color='blue', alpha=0.25, label='Covered danger zone'))

        ax.legend(loc='upper right')

        ax.set_xlim(0, self.field_length)
        ax.set_ylim(0, self.field_width)
        ax.set_aspect('equal')
        ax.set_title("Danger Zones & Defensive Coverage")

        plt.tight_layout()
        plt.show()
    
    def identify_pass_lines(self, frame_data: pd.DataFrame):
        """
        Identify potential pass lines from the ball carrier to teammates
        """
        possession_team_id = frame_data['possession_team_id'].iloc[0]

        attackers = frame_data[
            (frame_data['team_id'] == possession_team_id) &
            (frame_data['is_ball_carrier'] == False)
        ]

        ball_carrier = frame_data[frame_data['is_ball_carrier']].iloc[0]
        ball_pos = (ball_carrier['x'], ball_carrier['y'])

        pass_lines = []

        for _, att in attackers.iterrows():
            receiver_pos = (att['x'], att['y'])

            if utils.distance(ball_pos, receiver_pos) < config.D_MAX:
                pass_lines.append({
                    "start": ball_pos,
                    "end": receiver_pos,
                    "receiver_id": att['player_id']
                })

        return pass_lines
    
    def assign_pass_line_blocking(self, frame_data):
        """
        Assign defenders blocking each pass line
        """
        possession_team_id = frame_data['possession_team_id'].iloc[0]
        defenders = frame_data[frame_data['team_id'] != possession_team_id]

        blocked_lines = []

        pass_lines = self.identify_pass_lines(frame_data)
        
        for line in pass_lines:
            blockers = []

            for _, defender in defenders.iterrows():
                d_pos = (defender['x'], defender['y'])

                dist = utils.point_to_line_distance(
                    d_pos,
                    line["start"],
                    line["end"]
                )

                if dist < config.S:
                    blockers.append({
                        "defender_id": defender['player_id'],
                        "defender_pos": d_pos,
                        "distance": dist
                    })

            blocked_lines.append({
                **line,
                "blocked_by": blockers
            })

        return blocked_lines

    def calc_pass_line_blocking(self, frame_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate pass line blocking ratio for each defender
        """
        possession_team_id = frame_data['possession_team_id'].iloc[0]
        defenders = frame_data[frame_data['team_id'] != possession_team_id]

        blocked_lines = self.assign_pass_line_blocking(frame_data)

        blocking_count = {pid: 0 for pid in defenders['player_id']}

        if len(blocked_lines) == 0:
            return blocking_count

        for line in blocked_lines:
            for blocker in line["blocked_by"]:
                blocking_count[blocker["defender_id"]] += 1

        total_lines = len(blocked_lines)
        return {
            pid: blocking_count[pid] / total_lines
            for pid in blocking_count
        }
    
    def plot_pass_line_blocking(self, frame_data: pd.DataFrame):
        """
        Plot pass line blocking by defenders (refactored)
        """

        possession_team_id = frame_data['possession_team_id'].iloc[0]

        attackers = frame_data[
            (frame_data['team_id'] == possession_team_id) &
            (frame_data['is_ball_carrier'] == False)
        ]
        defenders = frame_data[frame_data['team_id'] != possession_team_id]

        ball_carrier = frame_data[frame_data['is_ball_carrier']].iloc[0]
        ball_pos = (ball_carrier['x'], ball_carrier['y'])

        blocked_lines = self.assign_pass_line_blocking(frame_data)
        blocking_ratio = self.calc_pass_line_blocking(frame_data)

        fig, ax = plt.subplots(figsize=(12, 7))
        utils.draw_pitch(ax)

        # Players
        ax.scatter(attackers['x'], attackers['y'], c='darkgreen', s=70, label='Receivers')
        ax.scatter(defenders['x'], defenders['y'], c='darkblue', s=70, label='Defenders')
        ax.scatter(ball_pos[0], ball_pos[1], c='darkgreen', s=140, marker='s', label='Ball carrier')

         # --- Blocking ratio labels ---
        for _, defender in defenders.iterrows():
            pid = defender['player_id']
            ratio = blocking_ratio.get(pid, 0.0)

            ax.text(
                defender['x'],
                defender['y'] + 1.2,
                f"{ratio:.2f}",
                color='red',
                fontsize=15,
                ha='center',
                va='bottom',
                zorder=5
            )
            
        # Pass lines
        for line in blocked_lines:
            p1, p2 = line["start"], line["end"]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                linestyle='--',
                color='lightgray',
                alpha=0.6,
                zorder=1
            )

            if len(line["blocked_by"]) > 0:
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color='blue',
                    linewidth=2.0,
                    alpha=0.8,
                    zorder=3
                )

                for blk in line["blocked_by"]:
                    ax.add_patch(
                        Circle(
                            blk["defender_pos"],
                            radius=config.S,
                            color='blue',
                            alpha=0.08,
                            zorder=2
                        )
                    )

        # --- Legend ---
        ax.plot([], [], linestyle='--', color='lightgray', label='Potential pass line')
        ax.plot([], [], color='blue', linewidth=2, label='Blocked pass line')
        ax.add_patch(Circle((0, 0), 1, color='blue', alpha=0.08, label='Defender influence zone'))
        
        ax.legend(loc='upper right')
        ax.set_xlim(0, self.field_length)
        ax.set_ylim(0, self.field_width)
        ax.set_aspect('equal')
        ax.set_title("Pass Line Blocking by Defenders")

        plt.tight_layout()
        plt.show()

    def get_scr(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Space Control Rating (SCR) for all players across entire match
        
        Args:
            tracking_df: Full tracking DataFrame for the match
            
        Returns:
            DataFrame with columns: player_id, team_id, scr, playing_time
        """
        frames = sorted(tracking_df['frame_id'].unique())
        
        # Initialize accumulators
        coverage_accumulator = {}
        blocking_accumulator = {}
        playing_time = {}
        team_mapping = {}

        # downsample to T-second intervals if needed
        period = 1 # 1 second intervals
        fps = config.FPS
        frame_interval = int(period * fps)
        frames = frames[::frame_interval]
        
        # Process each frame
        print("Calculating Space Control Rating (SCR) for all players at a 1-second interval...")
        for frame_id in tqdm(frames):
            frame_data = tracking_df[tracking_df['frame_id'] == frame_id]
            ball_carrier = frame_data[frame_data['is_ball_carrier']]
            if len(ball_carrier) == 0:
                continue
            
            # Calculate zone coverage
            coverage = self.calc_zone_coverage(frame_data)
            
            # Calculate pass line blocking
            blocking = self.calc_pass_line_blocking(frame_data)
            
            # Accumulate for each player
            for player_id in coverage.keys():
                if player_id not in coverage_accumulator:
                    coverage_accumulator[player_id] = []
                    blocking_accumulator[player_id] = []
                    playing_time[player_id] = 0.0
                    
                    # Store team mapping
                    player_team = frame_data[frame_data['player_id'] == player_id]['team_id'].iloc[0]
                    team_mapping[player_id] = player_team
                
                coverage_accumulator[player_id].append(coverage[player_id])
                blocking_accumulator[player_id].append(blocking[player_id])
                playing_time[player_id] = frame_id/10/60 # minutes played
        
        # Create result DataFrame
        results_df = []
        results_seq = {}
        for player_id in coverage_accumulator.keys():
            time_played = playing_time[player_id]
            
            if time_played > 0:
                # Average coverage and blocking ratios
                avg_coverage = np.mean(coverage_accumulator[player_id]) # avg coverage zones ratio per second
                avg_blocking = np.mean(blocking_accumulator[player_id]) # avg blocking lines ratio per second
                
                # SCR combines both metrics
                scr = (avg_coverage + avg_blocking) / 2.0
                
            else:
                scr = 0.0
            
            results_df.append({
                'player_id': player_id,
                'team_id': team_mapping[player_id],
                'avg_zone_coverage': avg_coverage if time_played > 0 else 0.0,
                'avg_pass_line_blocking': avg_blocking if time_played > 0 else 0.0,
                'avg_space_control_rating': scr,
                'playing_time': time_played
            })

            results_seq[player_id] = {
                'seq_zone_coverage': coverage_accumulator[player_id],
                'seq_pass_line_blocking': blocking_accumulator[player_id],
                'seq_space_control_rating': (np.array(coverage_accumulator[player_id]) + np.array(blocking_accumulator[player_id])) / 2.0,
            }
        
        return pd.DataFrame(results_df), results_seq

if __name__ == "__main__":
    
    
    """frame = pd.read_csv("data/match_1886347/frame.csv")

    # check if a player is in possession of the ball
    if frame["is_ball_carrier"].sum() == 0:
        raise ValueError("No player in possession of the ball in this frame.")

    # translater les coordonnées pour que l'origine soit au coin bas gauche et non au centre
    frame["x"] = frame["x"] + config.FIELD_LENGTH / 2
    frame["y"] = frame["y"] + config.FIELD_WIDTH / 2

    defense_rating = DefenseRating(config.FIELD_LENGTH, config.FIELD_WIDTH)
    #coverage_dict = defense_rating.calc_zone_coverage(frame)
    #print("Zone coverage per defender:\n", coverage_dict)
    defense_rating.plot_danger_zone_coverage(frame)

    #blocking_dict = defense_rating.calc_pass_line_blocking(frame)
    #print("Pass line blocking per defender:\n", blocking_dict)
    #
    #defense_rating.plot_pass_line_blocking(frame)
    """

    tracking_df = pd.read_csv("data/match_1886347/tracking_df.csv")
    
    # translater les coordonnées pour que l'origine soit au coin bas gauche et non au centre
    tracking_df["x"] = tracking_df["x"] + config.FIELD_LENGTH / 2
    tracking_df["y"] = tracking_df["y"] + config.FIELD_WIDTH / 2

    defense_rating = DefenseRating(config.FIELD_LENGTH, config.FIELD_WIDTH)
    scr_df = defense_rating.get_scr(tracking_df)
    print(scr_df)