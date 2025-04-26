from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from fastapi import Depends
from typing import AsyncGenerator
from app.core.config import settings

# Create a MongoDB client using settings
client = AsyncIOMotorClient(settings.MONGO_DETAILS)
database = client[settings.DATABASE_NAME]

async def get_database() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    try:
        yield database
    finally:
        pass