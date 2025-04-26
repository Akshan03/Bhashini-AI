from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import Depends
from typing import AsyncGenerator

MONGO_DETAILS = "mongodb://localhost:27017"  # MongoDB connection string
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.chatbot_db

async def get_database() -> AsyncGenerator:
    try:
        yield database
    finally:
        pass