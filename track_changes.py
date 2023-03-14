from constants import classes
from classify_latest import classify
from pathlib import Path
from time import time
CHANGE_FILE = '/tmp/changes_in_garage'


def track_changes():
    """Tracks changes in the garage."""
    change_file = Path(CHANGE_FILE)
    state, timestamp = get_saved_state(change_file)
    current_state = get_current_state()
    current_timestamp = time()

    if not state or not timestamp or state != current_state:
        write_new_state(change_file, current_state, current_timestamp)
        return current_state, current_timestamp

    return state, timestamp


def write_new_state(change_file, current_state, current_time):
    change_file.write_text(f"{current_state},{current_time}")


def get_current_state():
    class_index = classify()
    current_state = classes[class_index]
    return current_state


def get_saved_state(change_file):
    state = None
    timestamp = None
    if change_file.exists():
        changes = change_file.read_text()
        state, timestamp = changes.split(',')
    return state, timestamp


if __name__ == '__main__':
    state, timestamp = track_changes()
    print(f"State: {state}")
    print(f"Timestamp: {timestamp}")
