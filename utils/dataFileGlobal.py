# Shared global state for tracking violations across modules

myList     = []    # stores ROI bounding boxes
listTrack  = []    # stores tracked vehicle IDs
numTrack   = 0     # count of violations detected
speed_limit = 40.0 # km/h — updated by GUI at runtime
violations = []    # list of violation event dicts

def init():
    global myList, listTrack, numTrack, speed_limit, violations
    myList      = []
    listTrack   = []
    numTrack    = 0
    speed_limit = 40.0
    violations  = []
