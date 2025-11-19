import os
import pymysql
from dotenv import load_dotenv

load_dotenv()
MYSQL_SERVER = os.getenv('MYSQL_SERVER')
MYSQL_DB = os.getenv('MYSQL_DB')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')

def get_conn():
    return pymysql.connect(
        host=MYSQL_SERVER,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset='utf8'
    )
