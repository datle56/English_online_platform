from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from session.database import User, Admin
from session.dependencies import get_db  # Import the get_db dependency

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

SECRET_KEY = "123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 90

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    print(expire)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)  # Inject the database session using dependency
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role not in ["user", "admin"]:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception

    user = None
    if role == "user":
        user = db.query(User).filter(User.username == token_data.username).first()
    elif role == "admin":
        user = db.query(Admin).filter(Admin.username == token_data.username).first()

    if user is None:
        raise credentials_exception
    return user

from fastapi import WebSocket
async def get_websocket_user(websocket: WebSocket, db: Session = Depends(get_db)):
    # Lấy token từ query parameter
    token = websocket.query_params.get("token")
    print(token)
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    
    # Giải mã token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None or role not in ["user", "admin"]:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise HTTPException(status_code=401, detail="Token decoding failed")
    
    # Tìm người dùng trong database
    user = db.query(User).filter(User.username == token_data.username).first()
    
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user