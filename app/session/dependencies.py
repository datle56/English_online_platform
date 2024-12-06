from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from session.database import Base

# Create the SQLite database engine
engine = create_engine('sqlite:///./users.db', connect_args={"check_same_thread": False})  # check_same_thread=False is needed for SQLite in multi-threaded contexts

# Configure a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables in the database
Base.metadata.create_all(bind=engine)
