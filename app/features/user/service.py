from fastapi import APIRouter, HTTPException
# from sqlalchemy.orm import Session
from passlib.context import CryptContext
from session.dependencies import get_db
from session.database import User, Admin
from session.auth import oauth2_scheme, Token, TokenData, create_access_token, get_current_user
from datetime import datetime, timedelta


pwd_context  = CryptContext(schemes=["bcrypt"], deprecated="auto")

def login_user(username: str, password: str, db):
    # Kiểm tra xem user có tồn tại không
    user = db.query(User).filter(User.username == username).first()
    admin = db.query(Admin).filter(Admin.username == username).first()

    if not user and not admin:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # Kiểm tra mật khẩu và xác định vai trò
    if user and pwd_context.verify(password, user.password):
        role = "user"
    elif admin and pwd_context.verify(password, admin.password):
        role = "admin"
    else:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # Tạo access token
    access_token_expires = timedelta(minutes=90)
    access_token = create_access_token(
        data={"sub": username, "role": role, "username": username},
        expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer", "role": role, "username": username}


def register_user(username: str, email: str, password: str, role: str, db):
    # Kiểm tra xem username đã tồn tại chưa
    user = db.query(User).filter(User.username == username).first()
    admin = db.query(Admin).filter(Admin.username == username).first()
    if user or admin:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Mã hóa mật khẩu
    hashed_password = pwd_context.hash(password)

    # Tạo user hoặc admin tùy thuộc vào role
    if role == "user":
        new_user = User(username=username, email=email, password=hashed_password)
        print(new_user)   
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    elif role == "admin":
        new_admin = Admin(username=username, email=email, password=hashed_password)
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)

    return {"username": username, "email": email, "role": role, "message": "Registration successful"}