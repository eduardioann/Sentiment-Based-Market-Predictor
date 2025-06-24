import os
import psycopg2
from .config import POSTGRES_CONFIG

def init_postgres():
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()

    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
        cursor.execute(schema_sql)

    cursor.close()
    conn.close()

def init_database():
    print("Initializing PostgreSQL...")
    init_postgres()
    print("PostgreSQL initialized successfully!")

if __name__ == "__main__":
    init_database() 