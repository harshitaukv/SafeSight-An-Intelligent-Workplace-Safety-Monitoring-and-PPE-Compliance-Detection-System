from pymongo import MongoClient

# ----------------------------------------
# Connect to MongoDB
# ----------------------------------------
client = MongoClient("mongodb://localhost:27017")

# ----------------------------------------
# Database
# ----------------------------------------
db = client["SAFESIGHT"]

# ----------------------------------------
# Collections
# ----------------------------------------
detections = db["detection_history"]

# New collection for uploaded manuals/documents
documents = db["documents"]

# Collection for login accounts (normal users + admins)
users = db["users"]

# Collection for construction sites (first-class entity, used for
# filtering/reporting and assigned at upload time)
sites = db["sites"]

print("MongoDB Connected Successfully")