import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
import pymap3d as pm

dbfile = "sound_sensors.db"
V_Sound = 343.2 #m/second

def init():
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS Sensors (Id INTEGER PRIMARY KEY AUTOINCREMENT, X REAL, Y REAL, Z REAL, T Integer, Amplitude REAL);")
    c.execute("CREATE TABLE IF NOT EXISTS Events (Id INTEGER PRIMARY KEY AUTOINCREMENT, Lat REAL, Lon REAL, T Integer);")
    conn.commit()
    return conn

def newconn():
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    conn.commit()
    return conn

conn = init()

def latlong2ecef(lat, long, alt):
    return pm.geodetic2ecef(lat, long, alt)

def query_to_time(time):
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=time)

def str_to_time(time):
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f%z")

def time_to_str(time):
    return datetime.strftime(time, "%Y-%m-%d %H:%M:%S.%f%z")

def latlon_to_fake_xy(s):
    s[:, 0] = R_Earth * np.sin(s[:, 0] / 180.0 * np.pi)
    s[:, 1] = R_Earth * np.multiply(np.sin(s[:, 1] / 180.0 * np.pi), np.cos(s[:, 0] / 180.0 * np.pi))
    return s

def add_reading_to_db(time, amplitude, lat, long, alt=0.0):
    global conn
    c = conn.cursor()
    x, y, z = latlong2ecef(lat, long, alt)
    c.execute("INSERT INTO Sensors (X, Y, Z, T, Amplitude) VALUES (?, ?, ?, ?, ?);", (x, y, z, time, amplitude))
    conn.commit()

def add_event_to_db(x, y, z, time, conn=conn):
    c = conn.cursor()
    lat, lon, _ = pm.ecef2geodetic(x, y, z)
    c.execute("INSERT INTO Events (Lat, Lon, T) VALUES (?, ?, ?);", (lat, lon, time))
    conn.commit()

def clean_db(current_time=datetime.now(timezone.utc), offset_s=timedelta(seconds=-15), offset_e=timedelta(minutes=-2), conn=conn):
    c = conn.cursor()
    t_s = (current_time + offset_s - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(milliseconds=1)
    t_e = (current_time + offset_e - datetime(1970, 1, 1, tzinfo=timezone.utc)) / timedelta(milliseconds=1)
    c.execute("DELETE FROM Sensors WHERE (T < ?);", (t_s,))
    c.execute("DELETE FROM Events WHERE (T < ?);", (t_e,))
    conn.commit()

def dump_for_hotspot(conn=conn):
    c = conn.cursor()
    sensors = []
    for row in c.execute("SELECT X, Y, Z, Amplitude FROM Sensors;"):
        lat, lon, _ = pm.ecef2geodetic(x, y, z)
        row = (x, z, Amplitude)
        sensors.append(row)
    sensors = np.array(sensors)
    sensors = latlon_to_fake_xy(sensors)
    return sensors

def dump_for_delay(conn=conn):
    c = conn.cursor()
    sensors = []
    t_first = None
    for row in c.execute("SELECT X, Y, Z, T FROM Sensors;"):
        row_conv = (row[0], row[1], row[2], row[3] / 1000 * V_Sound)
        if not t_first:
            t_first = row[3]
        sensors.append(row_conv)
    sensors = np.array(sensors)
    return sensors, t_first
