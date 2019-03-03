import requests 
import time
import random

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

def generate_timestamps(n, tmin, tmax): 
    timestamp = time.time() * np.power(10., 3.) 
    return timestamp

url = make_request_url(0,0,0,0,0)
r = requests.get(url)
print(r.text)
