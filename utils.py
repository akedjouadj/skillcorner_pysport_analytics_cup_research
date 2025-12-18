"""
Utility functions for geometric calculations and offside detection
"""

import numpy as np
from typing import Tuple, List
import pandas as pd
import config

from matplotlib.patches import Circle, Rectangle, Arc


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1: Tuple (x, y) for point 1
        p2: Tuple (x, y) for point 2
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_between_three_points(p1: Tuple[float, float], 
                                 p2: Tuple[float, float], 
                                 p3: Tuple[float, float]) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3 in degrees
    
    Args:
        p1: First point (x, y)
        p2: Vertex point (x, y)
        p3: Third point (x, y)
        
    Returns:
        Angle in degrees
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Avoid division by zero
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def is_offside(player_pos: Tuple[float, float], 
               ball_pos: Tuple[float, float],
               opponent_positions: List[Tuple[float, float]],
               attacking_direction: str = 'right') -> bool:
    """
    Check if a player is in an offside position
    
    Args:
        player_pos: Position (x, y) of the attacking player
        ball_pos: Position (x, y) of the ball
        opponent_positions: List of (x, y) positions of all opponent players
        attacking_direction: 'right' if attacking towards positive x, 'left' otherwise
        
    Returns:
        True if player is offside, False otherwise
    """
    player_x = player_pos[0]
    ball_x = ball_pos[0]
    
    if len(opponent_positions) < 2:
        return False
    
    # Extract x coordinates of opponents
    opponent_x_coords = [pos[0] for pos in opponent_positions]
    opponent_x_coords.sort(reverse=(attacking_direction == 'right'))
    
    # Find second-to-last defender position
    if len(opponent_x_coords) >= 2:
        second_last_defender_x = opponent_x_coords[1]
    else:
        second_last_defender_x = opponent_x_coords[0]
    
    # Check offside conditions
    if attacking_direction == 'right':
        # Player is ahead of second-to-last defender and ahead of ball
        is_ahead_of_defender = player_x > second_last_defender_x
        is_ahead_of_ball = player_x > ball_x
    else:
        # Player is ahead of second-to-last defender and ahead of ball (attacking left)
        is_ahead_of_defender = player_x < second_last_defender_x
        is_ahead_of_ball = player_x < ball_x
    
    return is_ahead_of_defender and is_ahead_of_ball


def get_attacking_direction(team_id: int, 
                             team_positions: pd.DataFrame,
                             field_length: float) -> str:
    """
    Determine attacking direction for a team based on average position
    
    Args:
        team_id: Team identifier
        team_positions: DataFrame with player positions for this team
        field_length: Length of the field
        
    Returns:
        'right' if attacking towards positive x, 'left' otherwise
    """
    avg_x = team_positions['x'].mean()
    
    # If team is on the left half on average, they attack right
    if avg_x < field_length / 2:
        return 'right'
    else:
        return 'left'

def point_to_line_distance(point: Tuple[float, float],
                            line_start: Tuple[float, float],
                            line_end: Tuple[float, float]) -> float:
    """
    Calculate perpendicular distance from a point to a line segment
    
    Args:
        point: Point (x, y)
        line_start: Start of line segment (x, y)
        line_end: End of line segment (x, y)
        
    Returns:
        Distance from point to line segment
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1
    
    # Handle degenerate case (line is actually a point)
    if dx == 0 and dy == 0:
        return distance(point, line_start)
    
    # Parameter t represents projection onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return distance(point, (closest_x, closest_y))

def point_to_line_distance_vectorized(points, line_start, line_end):
    """
    Compute distances from multiple points to a line segment
    """
    p = points
    a = np.array(line_start)
    b = np.array(line_end)

    ab = b - a
    ap = p - a

    t = np.clip(
        np.sum(ap * ab, axis=1) / np.dot(ab, ab),
        0.0,
        1.0
    )

    closest = a + t[:, None] * ab
    return np.linalg.norm(p - closest, axis=1)


def midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate midpoint between two points
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Midpoint (x, y)
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def draw_pitch(ax):
    """
    Draw a soccer pitch on given matplotlib axis
    """
    
    # Outer rectangle
    ax.add_patch(Rectangle((0,0), config.FIELD_LENGTH, config.FIELD_WIDTH, fill=False, lw=2))

    # Halfway line
    ax.plot([config.FIELD_LENGTH/2, config.FIELD_LENGTH/2], [0, config.FIELD_WIDTH], color="black")

    # Center circle
    ax.add_patch(Circle((config.FIELD_LENGTH/2, config.FIELD_WIDTH/2), 9.15, fill=False))

    # Penalty areas
    # Left
    ax.add_patch(Rectangle((0, (config.FIELD_WIDTH-40.32)/2), 16.5, 40.32, fill=False))
    ax.add_patch(Rectangle((0, (config.FIELD_WIDTH-18.32)/2), 5.5, 18.32, fill=False))
    ax.add_patch(Rectangle((0, (config.FIELD_WIDTH-7.32)/2), 2.44, 7.32, fill=False))
    # Right
    ax.add_patch(Rectangle((config.FIELD_LENGTH-16.5, (config.FIELD_WIDTH-40.32)/2), 16.5, 40.32, fill=False))
    ax.add_patch(Rectangle((config.FIELD_LENGTH-5.5, (config.FIELD_WIDTH-18.32)/2), 5.5, 18.32, fill=False))
    ax.add_patch(Rectangle((config.FIELD_LENGTH-2.44, (config.FIELD_WIDTH-7.32)/2), 2.44, 7.32, fill=False))
    # Arcs
    ax.add_patch(Arc((11, config.FIELD_WIDTH/2), 18.3, 18.3, angle=0, theta1=310, theta2=50))
    ax.add_patch(Arc((config.FIELD_LENGTH-11, config.FIELD_WIDTH/2), 18.3, 18.3, angle=0, theta1=130, theta2=230))

def find_ball_carrier(frame, match, player_tracking_df_ids):
    """ 
    Identify the ball carrier in a given frame
    """
    
    possession_team_id = frame["ball_owning_team_id"]

    min_dist = float("inf")
    closest_defender_id = None
    ball_x, ball_y = frame["ball_x"], frame["ball_y"]
    
    for player in match["players"]:
        if player["id"] not in player_tracking_df_ids:
            continue

        if player["team_id"] == possession_team_id:
            player_x = frame[f"{player['id']}_x"]
            player_y = frame[f"{player['id']}_y"] 
            d = distance(
                (ball_x, ball_y),
                (player_x, player_y)
            )
            if d < min_dist:
                min_dist = d
                closest_defender_id = player['id']

    is_ball_carrier = closest_defender_id
    
    return is_ball_carrier