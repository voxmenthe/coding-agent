import os
import sqlite3
from pathlib import Path

def create_test_db():
    """Create a test database with sample data"""
    # Get the current directory
    project_root = Path(os.getcwd())
    db_path = project_root / "data" / "database.db"
    
    # Ensure the directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a test table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        age INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Insert sample data
    users = [
        (1, "Alice Smith", "alice@example.com", 28),
        (2, "Bob Johnson", "bob@example.com", 35),
        (3, "Charlie Brown", "charlie@example.com", 42),
        (4, "Diana Prince", "diana@example.com", 31),
        (5, "Edward Cullen", "edward@example.com", 24)
    ]
    
    cursor.executemany("""
    INSERT OR REPLACE INTO test_users (id, name, email, age) 
    VALUES (?, ?, ?, ?)
    """, users)
    
    # Create another test table for relational data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS test_posts (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        title TEXT NOT NULL,
        content TEXT,
        likes INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES test_users (id)
    )
    """)
    
    # Insert sample posts
    posts = [
        (1, 1, "First Post", "This is Alice's first post", 5),
        (2, 1, "Second Post", "Alice's follow-up thoughts", 3),
        (3, 2, "Hello World", "Bob's introduction post", 7),
        (4, 3, "My Journey", "Charlie's life story", 12),
        (5, 4, "Tech Tips", "Diana's technical advice", 20),
        (6, 5, "Book Review", "Edward's thoughts on literature", 8),
        (7, 3, "Travel Blog", "Charlie's adventures abroad", 15)
    ]
    
    cursor.executemany("""
    INSERT OR REPLACE INTO test_posts (id, user_id, title, content, likes) 
    VALUES (?, ?, ?, ?, ?)
    """, posts)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Test database created at {db_path}")

if __name__ == "__main__":
    create_test_db()