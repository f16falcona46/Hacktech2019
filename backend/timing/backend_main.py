#!/usr/bin/env python3

import wsgiref.simple_server
from urllib.parse import parse_qs
import sound_sensor_db as sdb
import tdoa_locate as tdoa

min_acc = 20

def test_tdoa_goodness():
    s, t_first = sdb.dump_for_delay()
    n = s.shape[0]
    if n >= 4:
        loc = s[:, :3]
        dist = s[:, 3]
        x_s, acc = shuffle_tdoa_locate(loc, dist, n // 2)
        if acc < min_acc:
            sdb.add_event_to_db(x_s[0], x_s[1], x_s[2], t_first)

def test_hotspot_goodness():
    return

class DbCleanerThread(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event
    
    def run(self):
        while not self.stopped.wait(2.0):
            sdb.clean_db()
            test_tdoa_goodness()

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
