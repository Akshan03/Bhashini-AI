from fastapi.security import APIKeyHeader
from fastapi import HTTPException, Security

API_KEY_NAME = "x-api-key"
API_KEY = "your-secure-api-key"  # Replace with your secure key or environment variable

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
