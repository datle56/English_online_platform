from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
# Define the base class for declarative models
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String, default="user") 

# Define the Admin model
class Admin(Base):
    __tablename__ = "admin"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String, default="admin") 

# Define the History model
class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # Foreign key to User table
    feature = Column(String, nullable=False)  # Feature name (Conversation, Grammar, Pronunciation)
    input_data = Column(Text, nullable=False)  # Input data (audio or text)
    output_data = Column(Text, nullable=False)  # Processed result (transcript, corrected sentence, score)
    created_at = Column(DateTime, default=datetime.utcnow)  # Timestamp for the record

    # Relationship to the User table
    user = relationship("User", back_populates="histories")

# Add a relationship in the User model to link with History
User.histories = relationship("History", back_populates="user", cascade="all, delete-orphan")
