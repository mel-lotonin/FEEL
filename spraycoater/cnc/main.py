from pygcode import *

from linalg import *
from paths import serp

# Preamble for the g-code script
preamble = [
    GCodeUseMillimeters(),
    GCodeFeedRate(5000),
    GCodeSpindleSpeed(20000),
    GCodeLinearMove(Z=0),
    GCodeLinearMove(X=0, Y=0),
]

gcodes = preamble + serp((36, 14), 10, Point([0, 0])) \
        + serp((36, 14), 7, Point([0, 0]), rotate=True) \
        + serp((36, 14), 10, Point([5, 0])) \
        + serp((36, 14), 7, Point([0, 3.5]), rotate=True) \
        + serp((36, 14), 10, Point([0, 0])) \
        + serp((36, 14), 7, Point([0, 0]), rotate=True) \
        + serp((36, 14), 10, Point([5, 0])) \
        + serp((36, 14), 7, Point([0, 3.5]), rotate=True) \
        + serp((36, 14), 10, Point([0, 0])) \
        + serp((36, 14), 7, Point([0, 0]), rotate=True) \
        + serp((36, 14), 10, Point([5, 0])) \
        + serp((36, 14), 7, Point([0, 3.5]), rotate=True) \
# Print G-Codes

with open("gcode.nc", "w") as file:
    file.write('\n'.join(str(gcode) for gcode in gcodes))