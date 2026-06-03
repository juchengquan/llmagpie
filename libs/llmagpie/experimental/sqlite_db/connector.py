"""SQLAlchemy engine/session factory for the experimental sqlite store.

Importing this module used to create the engine and run
``Base.metadata.create_all`` at module load — which crashed unless
``SQLITE_DB_DIR`` was set and silently created tables otherwise.
Use :func:`get_session_factory` (or :func:`get_engine`) instead so
the side effects only happen when a caller asks for them."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .datatype import Base


def get_engine(db_dir: str | None = None) -> Engine:
    """Build the SQLite engine. ``db_dir`` defaults to the
    ``SQLITE_DB_DIR`` env var; raises ``ValueError`` if neither is set."""
    db_dir = db_dir if db_dir is not None else os.getenv("SQLITE_DB_DIR")
    if not db_dir:
        raise ValueError(
            "SQLite DB directory not configured. Pass `db_dir=...` or set "
            "the SQLITE_DB_DIR environment variable."
        )
    engine = create_engine(
        f"sqlite:///{db_dir}/sql_app.db", echo=False, connect_args={"timeout": 10}
    )
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(db_dir: str | None = None):
    """Return a ``sessionmaker`` bound to the configured engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine(db_dir))
