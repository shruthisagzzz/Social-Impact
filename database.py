import sqlite3

def init_db():
    conn = get_db_connection()
    
    # Create users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            company TEXT NOT NULL,
            project_type TEXT NOT NULL,
            description TEXT,
            sustainability_score REAL,
            community_engagement_score REAL,
            ethical_business_score REAL,
            public_engagement_score REAL,
            total_impact_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create file_analysis table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS file_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            domain_type TEXT NOT NULL,
            extracted_text TEXT,
            impact_score REAL,
            grant_recommendation REAL,
            funding_recommendation TEXT,
            gemini_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create grant_proposals table with proper TEXT fields
    conn.execute('''
        CREATE TABLE IF NOT EXISTS grant_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            project_title TEXT NOT NULL,
            domain_type TEXT NOT NULL,
            requested_amount REAL NOT NULL,
            grant_amount REAL,
            impact_score REAL,
            project_description TEXT,
            executive_summary TEXT,
            objectives TEXT,
            methodology TEXT,
            budget_breakdown TEXT,
            timeline TEXT,
            expected_outcomes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!")