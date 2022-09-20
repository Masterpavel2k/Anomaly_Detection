import pymongo
from bson.objectid import ObjectId

myclient = pymongo.MongoClient('mongodb://masterpavel:5uunt0@192.168.1.125:27017/?authMechanism=DEFAULT')
mydb = myclient['SignalAnalysis']
mycol = mydb['HeartBeats']
# esempio di inserimento di battito
"""

example = {
  "ML2": [
    199,
    500
  ],
  "Class": "Normal",
  "Method Prediction": [
    "Normal",
    "Anomalous",
    "Normal"
  ],
  "Test": False
}
id = mycol.insert_one(example)
"""

# esempio di modifica di battito
"""
myquery = {
    "_id": ObjectId("6313bd7066cb874317982650")
}

newvalues = \
    {'$set':
        {"ML2": [
            434,
            600
            ]
        }
    }

mycol.update_one(myquery, newvalues)
"""

# esempio di lettura valore di battito
"""
for el in mycol.find():
    print(el['ML2'])
"""
