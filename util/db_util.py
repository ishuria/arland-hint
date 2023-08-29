import mysql.connector
from .config_util import DB_HOST

def open_database():
    return mysql.connector.connect(
        # host="192.168.0.102",
        host=DB_HOST,
        user="root",
        password="123456",
        database="ayesha",
        )