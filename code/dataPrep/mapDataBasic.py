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

with open('../../data/train/train.csv', newline='') as csvfile:
	csvFile = csv.reader(csvfile, delimiter=',', quotechar='|')
	i=0
	row=next(csvFile)
	row=next(csvFile)
	try:
		while row != None :#and i<5:
			#print(row[0])
			id=row[0]
			plo=row[5]
			pla=row[6]
			dolo=row[7]
			dola=row[8]
			dist=getFastestRoute(plo,pla,dolo,dola,id)
			writeData.append([id,dist])
			#print()
			#print(dist)
			row=next(csvFile)
			i+=1
	except:
		pass

myFile = open('example2.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(writeData)