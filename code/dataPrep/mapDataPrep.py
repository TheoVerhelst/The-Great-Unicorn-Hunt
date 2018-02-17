import urllib.request
import json
#t=urllib.request.urlopen('http://router.project-osrm.org/route/v1/driving/13.388860,52.517037;13.397634,52.529407;13.428555,52.523219?overview=false').read()
#t=urllib.request.urlopen('http://router.project-osrm.org/route/v1/driving/40.67999267578125,-73.964202880859375;40.655403137207031,-73.959808349609375;').read()
#print(t)
googleApi="AIzaSyCCfPdcnn1lUJwJHuJ-Xrur4qWRct9wopU"
t=urllib.request.urlopen('https://maps.googleapis.com/maps/api/directions/json?origin=40.67999267578125,-73.964202880859375&destination=40.655403137207031,-73.959808349609375&key={0}'.format(googleApi)).read()
data=json.loads(t)
print(json.dumps(data,indent=2))















