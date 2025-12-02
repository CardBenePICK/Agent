# 파일: agent/app/schemas/user_preference.py

from pydantic import BaseModel
from typing import List, Optional

class UserPreferenceCreate(BaseModel):
    user_id: Optional[str] = None
    cluster_id: int
    preferred_categories: List[str]
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "test_user_id",
                "cluster_id": 0,
                "preferred_categories": ["CAFE", "OTT"],
                "timestamp": "2024-12-01T12:00:00Z"
            }
        }