import requests 
  

URL = "http://maps.googleapis.com/maps/api/geocode/json"
  

location = "delhi technological university"

# sending get request and saving the response as response object

def make_request_url(magnitude,latitude,longitude,time,altitude):
    URL = "https://jasonli.us/cgi-bin/hacktech2019.cgi?";
    URL = URL + "m=" + str(magnitude) + "&lat=" + str(latitude) + "&long=" + str(longitude) + "&t=" + str(time) + "&alt=" + str(altitude);
    return URL

url = make_request_url(0,0,0,0,0)
r = requests.get(url)
print(r.text)
