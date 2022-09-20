import pymongo


def get_norm_collection():
    db = get_client()
    col = db['HeartBeats']
    return col


def get_anorm_collection():
    db = get_client()
    col = db['AnomHeartBeats']
    return col


def get_client():
    client = pymongo.MongoClient('mongodb://masterpavel:5uunt0@192.168.1.125:27017/?authMechanism=DEFAULT')
    db = client['SignalAnalysis']
    return db
