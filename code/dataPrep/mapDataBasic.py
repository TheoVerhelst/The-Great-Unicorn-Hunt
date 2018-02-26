import csv
import urllib.request
import json

def getFastestRoute(pickupLong,pickupLat,dropoffLong,dropoffLat,id):
	url='http://127.0.0.1:5000/route/v1/driving/{0},{1};{2},{3}?steps=true'.format(pickupLong,pickupLat,dropoffLong,dropoffLat)
	#print(url)
	try:
		t=urllib.request.urlopen(url).read().decode('utf-8')
		data=json.loads(t)
	except:
		print("request error")
		print(pickupLong,pickupLat,dropoffLong,dropoffLat)
		print(url)
		print(id)
		return -1
	distance=-1
   #try:
#        distance=data["routes"][0]["legs"][0]["distance"]["value"]
	try:
		distance=data["routes"][0]["distance"]
	except:
		print("json error")
		print(id)
		print(url)
	#print(type(data))
    #except IndexError:
	#print("distance")
	#print(json.dumps(data["routes"][0]["distance"],indent=2))
	#print(json.dumps(data,indent=2))
	#print(pickupLong,pickupLat,dropoffLong,dropoffLat)
	# for elem in data["routes"][0]:
	# 	print(elem)

	return distance

writeData=[["id","distance"]]

test=True
if test:
	idIDX=0
	ploIDX=4
	plaIDX=5
	doloIDX=6
	dolaIDX=7
	originFile='../../data/test/test.csv'
	destFile='test_distances.csv'
else:
	idIDX=0
	ploIDX=4+1
	plaIDX=5+1
	doloIDX=6+1
	dolaIDX=7+1
	originFile='../../data/train/train.csv'
	destFile='train_distances.csv'


with open(originFile, newline='') as csvfile:
	csvFile = csv.reader(csvfile, delimiter=',', quotechar='|')
	i=0
	row=next(csvFile)
	row=next(csvFile)
	try:
		while row != None:# and i<5:
			#print(row[0])
			id=row[idIDX]
			plo=row[ploIDX]
			pla=row[plaIDX]
			dolo=row[doloIDX]
			dola=row[dolaIDX]
			dist=getFastestRoute(plo,pla,dolo,dola,id)
			writeData.append([id,dist])
			#print()
			#print(dist)
			row=next(csvFile)
			i+=1
	except:
		pass

myFile = open(destFile, 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(writeData)