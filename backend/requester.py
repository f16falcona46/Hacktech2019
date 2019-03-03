import requests
import time
import random
import numpy as np
import math
from haversine import haversine

URL = "http://maps.googleapis.com/maps/api/geocode/json"
  

location = "delhi technological university"

# sending get request and saving the response as response object

def make_request_url(magnitude,latitude,longitude,time,altitude):
    URL = "https://jasonli.us/cgi-bin/hacktech2019.cgi?";
    URL = URL + "m=" + str(magnitude) + "&lat=" + str(latitude) + "&long=" + str(longitude) + "&t=" + str(time) + "&alt=" + str(altitude);
    return URL
 
def get_lat_long_itude(low,high):
    return random.uniform(low,high)
def get_magnitude(low,high):
    return random.uniform(low,high)

def generate_timestamps():
    timestamp = time.time() * 10**3 + 10**4 + random.randint(0,20)# 10 second offset
    return int(timestamp)

def get_altitude():
    return 0

def send_request(url):    
    r = requests.get(url)
    print(r.text)
    return r.text


num_requests = 5

s_speed = 343.2
lowlat = 33
lowlong = -119
latrange = 0.005
origin = (get_lat_long_itude(lowlat, lowlat + latrange), get_lat_long_itude(lowlong, lowlong + latrange))
origin_time = generate_timestamps()


gunshots = [(get_lat_long_itude(lowlat, lowlat + latrange), get_lat_long_itude(lowlong, lowlong + latrange)) for i in range(num_requests)]

urls = [make_request_url(
    get_magnitude(85,90),
    i[0],
    i[1],
    origin_time + int(haversine(i, origin,"m") * 10**3 / s_speed),
    get_altitude()
    ) for i in gunshots]

for i in urls:
    send_request(i)


time.sleep(3) 
urlreturn ="https://jasonli.us/cgi-bin/hacktech2019_return.cgi?"
predictionlist = send_request(urlreturn).split(',')
# prediction = (float(predictionlist[0]), float(predictionlist[1]))

print("prediction")
print(predictionlist)
print("actual")
print(origin)
print("actual time")
print(origin_time)

print(gunshots)


