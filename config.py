"""
Configuration file for Off-Ball Rating (OBR) system
Defines all constants and parameters for the analysis
"""

# Temporal parameters
T = 2.0  # Duration of sequence in seconds for discretization

# Spatial parameters
R = 3.0  # Radius for danger zone in meters
S = 1.0  # Threshold for pass line blocking in meters

# Field dimensions (standard, will be overridden by actual data)
FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH = 68.0    # meters

# xT grid resolution
XT_GRID_ROWS = 25     # Number of rows in xT grid
XT_GRID_COLS = 15     # Number of columns in xT grid

# Probability calculation parameters
D_MAX = 30.0  # Maximum effective passing distance in meters
ALPHA = 1.0   # Parameter for defender angle impact
BETA = 45.0   # Parameter for defender angle impact (degrees)

# OBR weights
OCR_WEIGHT = 0.5  # Weight for Offensive Creation Rating
SCR_WEIGHT = 0.5  # Weight for Space Control Rating

# Danger zone xT threshold
DANGER_ZONE_XT_THRESHOLD = 0.03  # Minimum xT value to consider a zone dangerous, opponent midfield

# Frame rate assumption
FPS = 10  # Frames per second if not derivable from data