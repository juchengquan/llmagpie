import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .datatype import Base

engine = create_engine(f'sqlite:///{os.getenv("SQLITE_DB_DIR")}/sql_app.db', echo=False, connect_args={'timeout': 10})
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
