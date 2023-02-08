import logging
import os
import sys

from flask import Flask

from .module import Dbs

"""
 Logging configuration
"""

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = '059f86d43fb3ae393745ef1e4485e431'
# app.config.from_object("config")
dbs = Dbs()
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append('/app')
sys.path.append(path)

"""
from sqlalchemy.engine import Engine
from sqlalchemy import event

#Only include this for SQLLite constraints
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    # Will force sqllite contraint foreign keys
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
"""

from . import views
