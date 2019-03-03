import requests 
import time
import random
import numpy as np

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
    timestamp = time.time() * np.power(10., 3.) 
    return timestamp

def get_altitude():
    return 0


def send_request(url):
    
    r = requests.get(url)
    print(r.text)

num_requests = 10
urls = [make_request_url(
    get_magnitude(85,90),
    get_lat_long_itude(33,35),
    get_lat_long_itude(-119,-117),
    generate_timestamps(),
    get_altitude()
    ) for i in range(num_requests)]

for i in urls:
    send_request(i)
