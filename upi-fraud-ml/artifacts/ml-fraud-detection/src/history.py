import json
import os
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), '..', 'predictions.json')


def _load() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)


def _save(data: dict):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def save_prediction(username: str, record: dict):
    data = _load()
    if username not in data:
        data[username] = []
    entry = {
        "id": len(data[username]) + 1,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        **record,
    }
    data[username].insert(0, entry)
    data[username] = data[username][:100]
    _save(data)
    return entry


def get_history(username: str) -> list:
    data = _load()
    return data.get(username, [])


def clear_history(username: str):
    data = _load()
    data[username] = []
    _save(data)
