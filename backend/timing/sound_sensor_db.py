import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pymap3d as pm

dbfile = "sound_sensors.db"
R_Earth = 3959 #Miles
V_Sound = 1 / 4.689 #Miles/second

def init():
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS Sensors (Id INTEGER PRIMARY KEY AUTOINCREMENT, Lat REAL, Lon REAL, Alt REAL, T TIMESTAMP, Amplitude REAL);")
    c.execute("CREATE TABLE IF NOT EXISTS Events (Id INTEGER PRIMARY KEY AUTOINCREMENT, Lat REAL, Lon REAL, T TIMESTAMP);")
    conn.commit()
    return conn

conn = init()

def latlong2ecef(lat, long, alt):
    return pm.geodetic2ecef(lat, long, alt)

def convtime(time):
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")

def latlon_to_xy(s):
    s[:, 0] = R_Earth * np.sin(s[:, 0] / 180.0 * np.pi)
    s[:, 1] = R_Earth * np.multiply(np.sin(s[:, 1] / 180.0 * np.pi), np.cos(s[:, 0] / 180.0 * np.pi))
    return s

def add_reading_to_db(time, amplitude, lat, long, alt=0):
    global conn
    c = conn.cursor()
    #x, y, z = latlong2ecef(lat, long, alt)
    c.execute("INSERT INTO Sensors (Lat, Lon, Alt, T, Amplitude) VALUES (?, ?, ?, ?, ?);", (lat, long, alt, t, amplitude))
    conn.commit()

def add_event_to_db(time, lat, long):
    global conn
    c = conn.cursor()
    c.execute("INSERT INTO Events (Lat, Lon, T) VALUES (?, ?, ?);", (lat, long, t))
    conn.commit()

def clean_db(current_time=datetime.now(), offset=timedelta(seconds=-15)):
    global conn
    c = conn.cursor()
    t = current_time + offset
    c.execute("DELETE FROM Sensors WHERE (T < ?);", (t,))
    c.execute("DELETE FROM Events WHERE (T < ?);", (t,))
    conn.commit()

def dump_for_hotspot():
    global conn
    c = conn.cursor()
    sensors = []
    for row in c.execute("SELECT Lat, Lon, Amplitude FROM Sensors;"):
        sensors.append(row)
    sensors = np.array(sensors)
    sensors = latlon_to_xy(sensors)
    return sensors

def dump_for_delay():
    global conn
    c = conn.cursor()
    sensors = []
    for row in c.execute("SELECT Lat, Lon, T FROM Sensors;"):
        row = (row[0], row[1], (convtime(row[2]) - datetime(2019, 1, 1)) / timedelta(seconds=1) * V_Sound)
        sensors.append(row)
    sensors = np.array(sensors)
    sensors = latlon_to_xy(sensors)
    return sensors
    
