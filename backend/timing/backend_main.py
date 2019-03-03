#!/usr/bin/env python3

import wsgiref.simple_server
from urllib.parse import parse_qs
import sound_sensor_db as sdb
import tdoa_locate as tdoa
from threading import Thread, Event
import numpy as np
from datetime import datetime, timezone

min_acc = 20

def test_tdoa_goodness(conn):
    s, t_first = sdb.dump_for_delay(conn)
    n = s.shape[0]
    if n >= 4:
        loc = s[:, :3]
        dist = s[:, 3]
        print(loc, dist)
        acc = min_acc + 1
        x_s = None
        try:
            x_s, acc = tdoa.shuffle_tdoa_locate(loc, dist, n // 2)
        except np.linalg.LinAlgError:
            print("Oops! Singlular matrix; continuing!")
        print(acc)
        if acc < min_acc:
            print(x_s[0], x_s[1], x_s[2], t_first)
            sdb.add_event_to_db(x_s[0], x_s[1], x_s[2], t_first, conn=conn)

def test_hotspot_goodness():
    return

class DbCleanerThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event
    
    def run(self):
        self.conn = sdb.newconn()
        while not self.stopped.wait(2.0):
            sdb.clean_db(datetime.now(timezone.utc), conn=self.conn)
            test_tdoa_goodness(conn=self.conn)

stop = Event()
t = DbCleanerThread(stop)
t.start()

def app(env, start_response):
    start_response("200 OK", [("Content-type", "text/html; charset=utf-8")])
    q = parse_qs(env["QUERY_STRING"])
    print(q)
    sdb.add_reading_to_db(int(q["t"][0]), float(q["m"][0]), float(q["lat"][0]), float(q["long"][0]), float(q["alt"][0]))
    return []

httpd = wsgiref.simple_server.make_server("localhost", 8080, app)
httpd.serve_forever()
