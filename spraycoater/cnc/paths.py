from pygcode import *
from linalg import *


def serp(dims: tuple[int, int], stride: float, offset: Point, rotate: bool = False) -> list[GCode]:
    """
    Create a serpentine path with a given frequency and offset
    :param dims: Dimensions of the serpentine
    :param stride: Stride between lines
    :param offset: Position offset
    :return: Serpentine path
    """
    path = []  # 1x1 serpentine path

    # Scale to correct size
    width, height = dims

    # Create 1x1 path with given stride
    alternate = False  # Alternate side flag
    x = 0
    ds = stride / (width if not rotate else height)
    while x <= 1:
        if alternate:
            path.append(Point([x, 0]))
            path.append(Point([x, 1]))
        else:
            path.append(Point([x, 1]))
            path.append(Point([x, 0]))
        x += ds
        alternate = not alternate

    # Iterate through all points
    for (i, item) in enumerate(path):
        # Skip all gcode items
        if isinstance(item, GCode):
            continue

        # Rotate and scale if needed
        if rotate:
            path[i] = Scale(Point([1, -1, 1])).transform(RotateZ(90).transform(item))

        # Convert to G-Code
        path[i] = Translate(offset).transform(Scale(Point([width, height, 1])).transform(path[i])).fast_to()

    path = path[:1] + [GCodeStartSpindleCW()] + path[1:] + [GCodeStopSpindle()]

    return path
