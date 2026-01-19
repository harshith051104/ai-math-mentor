import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from aimath.config.settings import Settings

class Memory:
    """
    Handles persistence of user sessions, problem states, and HITL data using SQLite.
    """
    def __init__(self, db_path: str = str(Settings.SQLITE_DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for storing conversation/session history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table for storing individual interaction steps
        # This logs: raw input, parsed intent, plan, execution steps, final verification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role TEXT, -- user, assistant, system
                content TEXT, -- JSON string or raw text
                meta_info TEXT, -- JSON string for extra metadata (e.g. confidence scores)
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')
        
        # Table for Human-In-The-Loop feedback
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hitl_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER,
                feedback_type TEXT, -- e.g., 'ocr_correction', 'math_correction'
                original_value TEXT,
                corrected_value TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(interaction_id) REFERENCES interactions(id)
            )
        ''')

        conn.commit()
        conn.close()

    def create_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO sessions (session_id) VALUES (?)', (session_id,))
        conn.commit()
        conn.close()

    def log_interaction(self, session_id: str, role: str, content: Any, meta_info: Dict[str, Any] = None):
        """Log a step in the pipeline."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        
        meta_str = json.dumps(meta_info) if meta_info else "{}"
        
        cursor.execute('''
            INSERT INTO interactions (session_id, role, content, meta_info)
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, meta_str))
        
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, meta_info, timestamp 
            FROM interactions 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        rows = cursor.fetchall()
        history = []
        for row in rows:
            try:
                content = json.loads(row['content'])
            except:
                content = row['content']
                
            history.append({
                'role': row['role'],
                'content': content,
                'meta_info': json.loads(row['meta_info']),
                'timestamp': row['timestamp']
            })
            
        conn.close()
        return history

    def log_feedback(self, interaction_id: int, feedback_type: str, original: str, corrected: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO hitl_feedback (interaction_id, feedback_type, original_value, corrected_value)
            VALUES (?, ?, ?, ?)
        ''', (interaction_id, feedback_type, original, corrected))
        conn.commit()
        conn.close()
