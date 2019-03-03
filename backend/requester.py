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
    timestamp = time.time() * 10**3 + random.randint(0,20)# 10 second offset
    return int(timestamp)

def get_altitude():
    return 0

def send_request(url):
    r = requests.get(url)
    print(r.text)
    return r.text


num_requests = 5

s_speed = 343.2
lowlat = 34.13
lowlong = -118.125
latrange = 0.5 # 0.005
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



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import io

u = u"""latitude,longitude
42.357778,-71.059444
39.952222,-75.163889
25.787778,-80.224167
30.267222, -97.763889"""
# detections = [(33.00408284015414, -118.99708569185316), (33.000130008413386, -118.99955942632312), (33.00147698094128, -118.9957493364294), (33.00031887563896, -118.9999448520674), (33.00355549092523, -118.99573956534623)]
# origin = (33.00147698094128, -118.99519119827238)


lat = []
lon = []

for tup in gunshots:
    lat.append(tup[0])
    lon.append(tup[1])
lat.append(origin[0])
lon.append(origin[1])
lat.append(float(predictionlist[0]))
lon.append(float(predictionlist[1]))

print('all')
print(lat)
print(lon)

# read in data to use for plotted points
buildingdf = pd.read_csv(io.StringIO(u), delimiter=",")
#lat = buildingdf['latitude'].values
#lon = buildingdf['longitude'].values

# determine range to print based on min, max lat and lon of the data
margin =  max(((max(lat) - min(lat)) / 2), int((max(lon) - min(lon)) / 2)) # buffer to add to the range
lat_min = min(lat) - margin
lat_max = max(lat) + margin
lon_min = min(lon) - margin
lon_max = max(lon) + margin

# create map using BASEMAP
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max-lon_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawcounties(zorder=20)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and lon to map projection coordinates
'''
lon_last = lon[-1]
lat_last = lat[-1]
lons, lats = m(lon[:len(lon)-1], lat[:len(lon)-1])
lonc, latc = m(lon_last, lat_last)
'''
lons, lats = m(lon, lat)
colors = ['b'] * (len(lons) - 2) + ['r'] + ['g']
sizes = [15] * (len(lons) - 2) + [35] + [15]
# plot points as red dots
reports = m.scatter(lons[:len(lons)-2], lats[:len(lats)-2], marker = 'o', color=colors[:len(lons)-2], zorder=5, s=15, label='Reports')
actual = m.scatter(lons[-2], lats[-2], marker = 'o', color=colors[-2], zorder=5, s=35, label='Actual')
predict = m.scatter(lons[-1], lats[-1], marker = 'o', color=colors[-1], zorder=5, s=15, label='Predicted')
plt.legend((reports, actual, predict),
('Reports', 'Actual Location', 'Predicted Location'),
           loc='best',
           ncol=1,
           fontsize=8
           )
plt.title('Location of Gunshots and Reports')
#m.scatter(lonc, latc, marker = 'o', color='r', zorder=5)
plt.show()
