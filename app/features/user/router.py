from fastapi import APIRouter, HTTPException, Form, Depends
from features.user.service import login_user, register_user
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from session.dependencies import get_db
from session.database import User, Admin, History  
from session.auth import oauth2_scheme, Token, TokenData, create_access_token, get_current_user
from datetime import datetime, timedelta
import sys
from pydantic import BaseModel
from typing import List



router = APIRouter()



@router.post("/login")
async def register(
    username: str = Form(...), 
    password: str = Form(...), 
    db: Session = Depends(get_db)
):
    return login_user(username=username, password=password,db=db)


@router.post("/register")
async def register(
    username: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...), 
    role: str = Form(...), 
    db: Session = Depends(get_db)
):

    return register_user(username=username, email=email, password=password, role=role, db=db)


# Pydantic model for the response
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str

    class Config:
        orm_mode = True  # Enable ORM compatibility


class HistoryRespond(BaseModel):
    id: int
    feature: str
    input_data: str
    output_data: str
    created_at: datetime

    class Config:
        orm_mode = True  # Enable ORM compatibility, necessary for ORM queries

@router.get("/history", response_model=List[HistoryRespond])
async def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get the current user's username
    username = current_user.username

    # Fetch the user from the database
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"error": "User not found"}

    # Query the user's history records
    histories = db.query(History).filter(History.user_id == user.id).all()

    # Return the history as a list of HistoryRespond models
    return histories
    
    


@router.get("/admin/users", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="You are not authorized to access this resource")

    users = db.query(User).offset(skip).limit(limit).all()
    return users