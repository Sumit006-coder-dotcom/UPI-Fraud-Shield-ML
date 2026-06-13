import json
import hashlib
import os
import re
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), '..', 'users.json')


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)


def _save(users: dict):
    os.makedirs(os.path.dirname(os.path.abspath(USERS_FILE)), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def register(username: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    email = email.strip().lower()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r'^[a-z0-9_]+$', username):
        return False, "Username can only contain letters, numbers and underscores."
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load()
    if username in users:
        return False, "Username already exists. Please choose another."
    for u in users.values():
        if u.get('email') == email:
            return False, "Email already registered. Please login."

    users[username] = {
        'email': email,
        'password': _hash(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None,
    }
    _save(users)
    return True, "Account created successfully!"


def login(username: str, password: str) -> tuple[bool, str, dict]:
    username = username.strip().lower()
    users = _load()

    if username not in users:
        return False, "Username not found. Please register first.", {}
    if users[username]['password'] != _hash(password):
        return False, "Incorrect password. Please try again.", {}

    users[username]['last_login'] = datetime.now().isoformat()
    _save(users)
    return True, "Login successful!", users[username]


def get_user(username: str) -> dict:
    return _load().get(username.lower(), {})
