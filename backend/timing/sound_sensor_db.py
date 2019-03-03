import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
import pymap3d as pm

dbfile = "sound_sensors.db"
R_Earth = 3959 #Miles
V_Sound = 1 / 4.689 #Miles/second

def init():
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS Sensors (Id INTEGER PRIMARY KEY AUTOINCREMENT, X REAL, Y REAL, Z REAL, T TIMESTAMP, Amplitude REAL);")
    c.execute("CREATE TABLE IF NOT EXISTS Events (Id INTEGER PRIMARY KEY AUTOINCREMENT, Lat REAL, Lon REAL, T TIMESTAMP);")
    conn.commit()
    return conn

conn = init()

def latlong2ecef(lat, long, alt):
    return pm.geodetic2ecef(lat, long, alt)

def query_to_time(time):
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=time)

def str_to_time(time):
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")

def latlon_to_fake_xy(s):
    s[:, 0] = R_Earth * np.sin(s[:, 0] / 180.0 * np.pi)
    s[:, 1] = R_Earth * np.multiply(np.sin(s[:, 1] / 180.0 * np.pi), np.cos(s[:, 0] / 180.0 * np.pi))
    return s

def add_reading_to_db(time, amplitude, lat, long, alt=0.0):
    global conn
    c = conn.cursor()
    x, y, z = latlon_to_xyz(lat, long, alt)
    c.execute("INSERT INTO Sensors (X, Y, Z, T, Amplitude) VALUES (?, ?, ?, ?, ?);", (x, y, z, query_to_time(time), amplitude))
    conn.commit()

def add_event_to_db(time, z, y, z):
    global conn
    c = conn.cursor()
    lat, lon, _ = pm.ecef2geodetic(x, y, z)
    c.execute("INSERT INTO Events (X, Y, Z, T) VALUES (?, ?, ?);", (lat, lon, t))
    conn.commit()

def clean_db(current_time=datetime.now(tzinfo=timezone.utc), offset=timedelta(seconds=-15)):
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
    for row in c.execute("SELECT X, Y, Z, Amplitude FROM Sensors;"):
        lat, lon, _ = pm.ecef2geodetic(x, y, z)
        row = (x, z, Amplitude)
        sensors.append(row)
    sensors = np.array(sensors)
    sensors = latlon_to_fake_xy(sensors)
    return sensors

def dump_for_delay():
    global conn
    c = conn.cursor()
    sensors = []
    for row in c.execute("SELECT X, Y, Z, T FROM Sensors;"):
        row = (row[0], row[1], row[2], (str_to_time(row[2]) - datetime(2019, 1, 1)) / timedelta(seconds=1) * V_Sound)
        sensors.append(row)
    sensors = np.array(sensors)
    return sensors
