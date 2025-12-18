"""
Defensive rating calculation: Space Control Rating (SCR)
Measures defensive positioning, zone coverage, and pass line blocking
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Tuple
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
        Optimized identification of danger zones (vectorized xT filtering)
        """
        possession_team_id = frame_data["possession_team_id"].iloc[0]

        attackers = frame_data[frame_data["team_id"] == possession_team_id]

        if attackers.empty:
            return []

        ball_carrier = frame_data[frame_data["is_ball_carrier"]].iloc[0]
        attacking_direction = ball_carrier["attacking_direction"]

        xt_grid = XTGrid(
            self.field_length,
            self.field_width,
            attacking_direction=attacking_direction,
        )

        positions = attackers[["x", "y"]].values
        player_ids = attackers["player_id"].values

        xt_values = np.array([
            xt_grid.get_xt_value(x, y) for x, y in positions
        ])

        mask = xt_values >= config.DANGER_ZONE_XT_THRESHOLD

        danger_zones = [
            {
                "attacker_id": pid,
                "center": (pos[0], pos[1]),
                "xt_value": xt,
            }
            for pid, pos, xt in zip(player_ids[mask], positions[mask], xt_values[mask])
        ]

        return danger_zones

    def assign_danger_zone_coverage(self, frame_data: pd.DataFrame):
        """
        Optimized assignment of defensive coverage to danger zones
        (fully vectorized distance computation)
        """
        possession_team_id = frame_data["possession_team_id"].iloc[0]
        defenders = frame_data[frame_data["team_id"] != possession_team_id]

        if defenders.empty:
            return []

        defender_positions = defenders[["x", "y"]].values
        defender_ids = defenders["player_id"].values

        danger_zones = self.identify_danger_zones(frame_data)

        covered_zones = []

        for zone in danger_zones:
            zone_center = np.array(zone["center"])

            # Vectorized distance to all defenders
            dists = np.linalg.norm(defender_positions - zone_center, axis=1)

            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            covered = min_dist < self.grid_step

            covered_zones.append({
                **zone,
                "covered": covered,
                "defender_id": defender_ids[min_idx] if covered else None,
                "defender_pos": tuple(defender_positions[min_idx]) if covered else None,
                "defender_distance": float(min_dist),
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
        Optimized identification of potential pass lines
        """
        possession_team_id = frame_data["possession_team_id"].iloc[0]

        attackers = frame_data[
            (frame_data["team_id"] == possession_team_id) &
            (~frame_data["is_ball_carrier"])
        ]

        if attackers.empty:
            return []

        ball_carrier = frame_data[frame_data["is_ball_carrier"]].iloc[0]
        ball_pos = np.array([ball_carrier["x"], ball_carrier["y"]])

        attacker_pos = attackers[["x", "y"]].values
        attacker_ids = attackers["player_id"].values

        dists = np.linalg.norm(attacker_pos - ball_pos, axis=1)
        mask = dists < config.D_MAX

        pass_lines = [
            {
                "start": tuple(ball_pos),
                "end": tuple(pos),
                "receiver_id": pid,
            }
            for pid, pos in zip(attacker_ids[mask], attacker_pos[mask])
        ]

        return pass_lines
    
    def assign_pass_line_blocking(self, frame_data: pd.DataFrame):
        """
        Optimized assignment of defenders blocking pass lines
        """
        possession_team_id = frame_data["possession_team_id"].iloc[0]
        defenders = frame_data[frame_data["team_id"] != possession_team_id]

        if defenders.empty:
            return []

        defender_pos = defenders[["x", "y"]].values
        defender_ids = defenders["player_id"].values

        pass_lines = self.identify_pass_lines(frame_data)

        blocked_lines = []

        for line in pass_lines:
            dists = utils.point_to_line_distance_vectorized(
                defender_pos,
                line["start"],
                line["end"],
            )

            mask = dists < config.S

            blockers = [
                {
                    "defender_id": pid,
                    "defender_pos": tuple(pos),
                    "distance": float(dist),
                }
                for pid, pos, dist in zip(
                    defender_ids[mask],
                    defender_pos[mask],
                    dists[mask],
                )
            ]

            blocked_lines.append({
                **line,
                "blocked_by": blockers,
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
            DataFrame with columns: player_id, team_id, scr
        """
        
        coverage_accumulator = {}
        blocking_accumulator = {}
        team_mapping = {}

        period = config.SCR_FREQ
        fps = config.FPS
        frame_interval = int(period * fps)

        frames = sorted(tracking_df["frame_id"].unique())
        frames = frames[::frame_interval]

        # 🔥 Key optimization: group once
        grouped = tracking_df.groupby("frame_id")

        print("Calculating Space Control Rating (SCR) for all players at a 1-second interval...")

        for frame_id in tqdm(frames):
            if frame_id not in grouped.groups:
                continue

            frame_data = grouped.get_group(frame_id)

            if not frame_data["is_ball_carrier"].any():
                continue

            coverage = self.calc_zone_coverage(frame_data)
            blocking = self.calc_pass_line_blocking(frame_data)

            for player_id in coverage.keys():
                if player_id not in coverage_accumulator:
                    coverage_accumulator[player_id] = []
                    blocking_accumulator[player_id] = []

                    team_mapping[player_id] = frame_data.loc[frame_data["player_id"] == player_id, "team_id"].iloc[0]

                coverage_accumulator[player_id].append(coverage[player_id])
                blocking_accumulator[player_id].append(blocking[player_id])

        results_df = []
        results_seq = {}

        for player_id in coverage_accumulator.keys():
            
            avg_cov = np.mean(coverage_accumulator[player_id]) if len(coverage_accumulator[player_id]) > 0 else 0.0
            avg_blk = np.mean(blocking_accumulator[player_id]) if len(blocking_accumulator[player_id]) > 0 else 0.0
            scr = (avg_cov + avg_blk) / 2.0
            
            results_df.append({
                "player_id": player_id,
                "team_id": team_mapping[player_id],
                "avg_zone_coverage": avg_cov,
                "avg_pass_line_blocking": avg_blk,
                "avg_space_control_rating": scr
            })

            results_seq[player_id] = {
                "seq_zone_coverage": coverage_accumulator[player_id],
                "seq_pass_line_blocking": blocking_accumulator[player_id],
                "seq_space_control_rating": (
                    (np.array(coverage_accumulator[player_id]) +
                    np.array(blocking_accumulator[player_id])) / 2.0
                ),
            }

        return pd.DataFrame(results_df), results_seq

if __name__ == "__main__":
    
    # Example usage: plot danger zone coverage for a single frame
    
    frame = pd.read_csv("data/match_1886347/frame.csv")

    defense_rating = DefenseRating(config.FIELD_LENGTH, config.FIELD_WIDTH)
    defense_rating.plot_danger_zone_coverage(frame)