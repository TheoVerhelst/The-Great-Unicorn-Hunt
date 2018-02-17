import urllib.request
import json
import pandas as pd
#t=urllib.request.urlopen('http://router.project-osrm.org/route/v1/driving/13.388860,52.517037;13.397634,52.529407;13.428555,52.523219?overview=false').read()
#t=urllib.request.urlopen('http://router.project-osrm.org/route/v1/driving/40.67999267578125,-73.964202880859375;40.655403137207031,-73.959808349609375;').read()
#print(t)


df = pd.read_csv("../../data/train/train.csv")


googleApi="AIzaSyCCfPdcnn1lUJwJHuJ-Xrur4qWRct9wopU"
t=urllib.request.urlopen('https://maps.googleapis.com/maps/api/directions/json?origin=40.67999267578125,-73.964202880859375&destination=40.655403137207031,-73.959808349609375&key={0}'.format(googleApi)).read()
data=json.loads(t)
#print(json.dumps(data["routes"][0]["legs"][0]["distance"]["value"],indent=2))

#print(data["routes"][0]["legs"][0]["distance"]["value"])

futureDF=dict()
#print(df.head())
futureDF["id"]=[]
futureDF["shortest_distance"]=[]


def getFastestRoute(pickupLong,pickupLat,dropoffLong,dropoffLat):
    url='https://maps.googleapis.com/maps/api/directions/json?origin={0},{1}&destination={2},{3}&key={4}'.format(pickupLat,pickupLong,dropoffLat,dropoffLong,googleApi)
    t=urllib.request.urlopen(url).read()
    data=json.loads(t)
    distance=-1
    try:
        distance=data["routes"][0]["legs"][0]["distance"]["value"]
    except IndexError:
        print(json.dumps(data))
        print(pickupLong,pickupLat,dropoffLong,dropoffLat)

    return distance

for row in df.itertuples():
    id=row[1]
    plo=row[6]
    pla=row[7]
    dolo=row[8]
    dola=row[9]
    futureDF["id"].append(id)
    dist=getFastestRoute(plo,pla,dolo,dola)
    futureDF["shortest_distance"].append(dist)


newDF=pd.DataFrame(futureDF)
newDF.to_csv("test.csv")









