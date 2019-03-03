#!/usr/bin/env python3

import wsgiref.simple_server
from urllib.parse import parse_qs
import sound_sensor_db as sdb

def app(env, start_response):
    start_response("200 OK", [("Content-type", "text/html; charset=utf-8")])
    q = parse_qs(env["QUERY_STRING"])
    print(q)
    sdb.add_reading_to_db(int(q["t"][0]), float(q["m"][0]), float(q["lat"][0]), float(q["long"][0]), float(q["alt"][0])) #FIX DATE CONVERSION!
    return []

httpd = wsgiref.simple_server.make_server("localhost", 8080, app)
httpd.serve_forever()
