from enum import Enum


class NodeRunningStatus(Enum):
    INACTIVE = 0
    RUNNING = 1
    ERROR = 2


class ConnectableType(Enum):
    BASENODE = 1
    PIPELINE = 2
