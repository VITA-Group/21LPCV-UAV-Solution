from enum import Enum

class ObjectCategory(Enum):
    PERSON = 0
    BALL = 1

class DivisionType(Enum):
    CROSSHAIR = 0
    GRID = 1

class BallStatus(Enum):
    FLYING = 0
    CAUGHT = 1