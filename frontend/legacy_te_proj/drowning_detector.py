import time

# Store previous hand positions and drowning state for movement analysis
_hand_history = {}
_drowning_state = {}

def is_drowning(person_id, pose_landmarks, frame_time=None):
    """
    Returns (state, confidence) where:
    - state: "Drowning", "Possible Drowning", "Safe"
    - confidence: float between 0 and 1
    """
    if not pose_landmarks or len(pose_landmarks) < 33:
        return "Unknown", 0.0

    # Key indices (mediapipe)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Get y-coordinates
    nose_y = pose_landmarks[NOSE].y
    left_shoulder_y = pose_landmarks[LEFT_SHOULDER].y
    right_shoulder_y = pose_landmarks[RIGHT_SHOULDER].y
    left_hip_y = pose_landmarks[LEFT_HIP].y
    right_hip_y = pose_landmarks[RIGHT_HIP].y
    left_wrist_y = pose_landmarks[LEFT_WRIST].y
    right_wrist_y = pose_landmarks[RIGHT_WRIST].y

    # Get x-coordinates for hand movement
    left_wrist_x = pose_landmarks[LEFT_WRIST].x
    right_wrist_x = pose_landmarks[RIGHT_WRIST].x

    # Heuristics
    head_above_hips = nose_y < min(left_hip_y, right_hip_y)
    wrists_below_shoulders = (left_wrist_y > left_shoulder_y) and (right_wrist_y > right_shoulder_y)
    vertical_posture = head_above_hips and wrists_below_shoulders
    head_above_shoulders = nose_y < min(left_shoulder_y, right_shoulder_y)

    # Hand movement tracking (immobility detection)
    now = frame_time if frame_time else time.time()
    prev = _hand_history.get(person_id)
    movement = 0
    if prev:
        prev_lx, prev_ly, prev_rx, prev_ry, prev_t = prev
        movement = ((left_wrist_x - prev_lx)**2 + (left_wrist_y - prev_ly)**2 +
                    (right_wrist_x - prev_rx)**2 + (right_wrist_y - prev_ry)**2)
        time_diff = now - prev_t
        immobile = movement < 0.0001 and time_diff > 2  # even stricter threshold
    else:
        immobile = False
    _hand_history[person_id] = (left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, now)

    # Drowning state persistence
    prev_state, prev_count = _drowning_state.get(person_id, ("Safe", 0))
    if vertical_posture and immobile and head_above_shoulders:
        count = prev_count + 1
        _drowning_state[person_id] = ("Drowning", count)
        if count >= 2:  # must persist for at least 2 consecutive checks
            return "Drowning", 0.99
        else:
            return "Possible Drowning", 0.7
    elif vertical_posture:
        _drowning_state[person_id] = ("Possible Drowning", 0)
        return "Possible Drowning", 0.7
    else:
        _drowning_state[person_id] = ("Safe", 0)
        return "Safe", 0.2
