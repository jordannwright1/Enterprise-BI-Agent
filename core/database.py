import sqlite3
import json
import os

DB_PATH = "tools/navi_skills.db"

def get_db_connection():
    """Helper to ensure the directory exists and return a connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Using 'IF NOT EXISTS' is the standard 'Architect' safety move
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE,
            description TEXT,
            code TEXT,
            packages TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_skill(keyword, task, code, packages):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO skills (keyword, description, code, packages)
            VALUES (?, ?, ?, ?)
        ''', (keyword, task, code, str(packages)))
        conn.commit()
    except Exception as e:
        print(f"Failed to save skill: {e}")
    conn.close()

def get_skill(keyword):
    init_db() 
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT code, packages FROM skills WHERE keyword = ?', (keyword,))
        result = cursor.fetchone()
        return result
    finally:
        conn.close()


def delete_skill(keyword: str):
    """Physically removes a skill from the navi_skills.db database."""
    db_path = "navi_skills.db" 
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM skills WHERE task_keyword = ?", (keyword,))
            conn.commit()
            return cursor.rowcount > 0 # Returns True if a row was actually deleted
    except Exception as e:
        print(f"Database Error: {e}")
        return False
