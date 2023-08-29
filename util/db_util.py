import mysql.connector

def open_database():
    return mysql.connector.connect(
        # host="192.168.0.102",
        host="127.0.0.1",
        user="root",
        password="123456",
        database="ayesha",
        )