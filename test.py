from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Access the database
db = client['image_db']

# Access the collection
collection = db['images']

# Insert a test document
test_doc = {"test": "Hello MongoDB"}
collection.insert_one(test_doc)

# Verify it was inserted
result = collection.find_one({"test": "Hello MongoDB"})
print(result)