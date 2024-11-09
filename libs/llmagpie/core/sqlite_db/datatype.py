from sqlalchemy import String, LargeBinary, Column, Double, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped


class Base(DeclarativeBase):
    pass


class SessionBase(Base):
    __tablename__ = "Sessions"
    id: Mapped[str] = Column(String, primary_key=True, index=True)
    time_started: Mapped[Double] = Column(Double, index=False)
    api_request: Mapped[LargeBinary] = Column(LargeBinary, index=False)
    prompt_template: Mapped[LargeBinary] = Column(LargeBinary, index=False)


class AppStateBase(Base):
    __tablename__ = "AppStates"
    id: Mapped[int] = Column(Integer, primary_key=True, index=True)
    key: Mapped[str] = Column(String, unique=True, index=True)
    value: Mapped[str] = Column(String, unique=True, index=True)
