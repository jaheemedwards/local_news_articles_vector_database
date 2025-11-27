import os
from dotenv import load_dotenv
import psycopg2

# Load .env variables
load_dotenv(dotenv_path='config.env')
DATABASE_URL = os.getenv("DATABASE_URL")

# Connect to the database
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Enable pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create news_articles table
cur.execute("""
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    category TEXT,
    date_iso TIMESTAMP,
    body TEXT NOT NULL,
    date_from_url TIMESTAMP,
    embedding VECTOR(768)  -- matches nomic-embed-text embedding size
);
""")

# Optional: create index for fast similarity search
cur.execute("""
CREATE INDEX IF NOT EXISTS idx_news_embedding
ON news_articles
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
""")

# Commit changes and close connection
conn.commit()
cur.close()
conn.close()

print("Database connected and table created successfully!")
