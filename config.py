# config.py
import os

def get_mysql_config():
    # Prefer environment variables (Railway). If absent, fallback to local XAMPP.
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("MYSQL_PORT", 3306))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    db = os.getenv("MYSQL_DATABASE", "softtissue_db")

    return {
        "MYSQL_HOST": host,
        "MYSQL_PORT": port,
        "MYSQL_USER": user,
        "MYSQL_PASSWORD": password,
        "MYSQL_DB": db
    }

def get_other_config():
    return {
        "MODEL_DIR": os.getenv("MODEL_DIR", "softtissue"),
        "SECRET_KEY": os.getenv("SECRET_KEY", "replace-with-secure-key")
    }
