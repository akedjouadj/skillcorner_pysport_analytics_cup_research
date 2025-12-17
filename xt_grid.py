"""
Expected Threat (xT) grid management
Provides static spatial value mapping for field positions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple
import config


class XTGrid:
    """
    Expected Threat grid for spatial valuation
    """
    
    def __init__(self, field_length: float, field_width: float, attacking_direction: str = 'right'):
        """
        Initialize xT grid
        
        Args:
            field_length: Length of the field in meters
            field_width: Width of the field in meters
        """
        self.field_length = field_length
        self.field_width = field_width
        self.attacking_direction = attacking_direction
        self.rows = config.XT_GRID_ROWS
        self.cols = config.XT_GRID_COLS
        
        # Create the grid
        self.grid = self._create_xt_grid(self.attacking_direction)
        
        # Create interpolator for smooth value lookup
        x_coords = np.linspace(0, field_length, self.rows)
        y_coords = np.linspace(0, field_width, self.cols)
        self.interpolator = RegularGridInterpolator(
            (x_coords, y_coords), 
            self.grid,
            bounds_error=False,
            fill_value=0.0
        )
    
    def _create_xt_grid(self, attacking_direction: str) -> np.ndarray:
        """
        Create a static xT grid with higher values near opponent's goal
        
        Returns:
            2D numpy array with xT values
        """
        grid = np.zeros((self.rows, self.cols))
        
        # Define zones with different xT values
        
        if attacking_direction == 'right':
            for i in range(self.rows):
                for j in range(self.cols):
                    # Normalize position (0 to 1)
                    x_norm = i / (self.rows - 1)
                    y_norm = j / (self.cols - 1)
                    
                    # Distance from center line (y-axis)
                    y_center_dist = abs(y_norm - 0.5)
                    
                    # Base value increases towards opponent's goal (x=1)
                    base_value = x_norm ** 2
                    
                    # Penalty box area (roughly last 16m, center third)
                    in_penalty_box_x = x_norm > 0.85
                    in_penalty_box_y = y_center_dist < 0.25
                    
                    if in_penalty_box_x and in_penalty_box_y:
                        # High value in penalty box
                        grid[i, j] = 0.15 + 0.10 * (x_norm - 0.85) / 0.15
                    elif x_norm > 0.7:
                        # Attacking third
                        center_bonus = (1 - 2 * y_center_dist) * 0.05
                        grid[i, j] = 0.05 + 0.08 * (x_norm - 0.7) / 0.3 + center_bonus
                    elif x_norm > 0.4:
                        # Middle third
                        grid[i, j] = 0.02 + 0.03 * (x_norm - 0.4) / 0.3
                    else:
                        # Defensive third - low value
                        grid[i, j] = 0.005 + 0.015 * x_norm / 0.4
        
        elif attacking_direction == 'left':
            for i in range(self.rows):
                for j in range(self.cols):
                    # Normalize position (0 to 1)
                    x_norm = i / (self.rows - 1)
                    y_norm = j / (self.cols - 1)
                    
                    # Distance from center line (y-axis)
                    y_center_dist = abs(y_norm - 0.5)
                    
                    # Base value increases towards opponent's goal (x=0)
                    base_value = (1 - x_norm) ** 2
                    
                    # Penalty box area (roughly first 16m, center third)
                    in_penalty_box_x = x_norm < 0.15
                    in_penalty_box_y = y_center_dist < 0.25
                    
                    if in_penalty_box_x and in_penalty_box_y:
                        # High value in penalty box
                        grid[i, j] = 0.15 + 0.10 * (0.15 - x_norm) / 0.15
                    elif x_norm < 0.3:
                        # Attacking third
                        center_bonus = (1 - 2 * y_center_dist) * 0.05
                        grid[i, j] = 0.05 + 0.08 * (0.3 - x_norm) / 0.3 + center_bonus
                    elif x_norm < 0.6:
                        # Middle third
                        grid[i, j] = 0.02 + 0.03 * (0.6 - x_norm) / 0.3
                    else:
                        # Defensive third - low value
                        grid[i, j] = 0.005 + 0.015 * (1 - x_norm) / 0.4
        
        return grid
    
    def get_xt_value(self, x: float, y: float) -> float:
        """
        Get xT value at a specific position using interpolation
        
        Args:
            x: X coordinate on the field
            y: Y coordinate on the field
            
        Returns:
            xT value at position (x, y)
        """
        # Clip coordinates to field bounds
        x = np.clip(x, 0, self.field_length)
        y = np.clip(y, 0, self.field_width)
        
        # Interpolate
        return float(self.interpolator([x, y]))
    
    def get_xt_values_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Get xT values for multiple positions at once
        
        Args:
            positions: Nx2 array of (x, y) positions
            
        Returns:
            Array of xT values
        """
        # Clip all positions
        positions = np.clip(
            positions,
            [0, 0],
            [self.field_length, self.field_width]
        )
        
        return self.interpolator(positions)
    
    def visualize_grid(self) -> np.ndarray:
        """
        Return the full grid for visualization purposes
        
        Returns:
            2D array of xT values
        """
        return self.grid.copy()
    
    def plot_grid(self):
        """
        Plot the xT grid using matplotlib
        """
        
        plt.imshow(
            self.grid.T,
            extent=[0, self.field_length, 0, self.field_width],
            origin='lower',
            cmap='YlOrRd',
            alpha=0.8
        )
        plt.colorbar(label='Expected Threat (xT)')
        plt.title('Expected Threat (xT) Grid')
        plt.xlabel('Field Length (m)')
        plt.ylabel('Field Width (m)')
        plt.show()

if __name__ == "__main__":
    # Example usage and visualization
    
    xt_grid = XTGrid(config.FIELD_LENGTH, config.FIELD_WIDTH, attacking_direction='left')
    xt_grid.plot_grid()