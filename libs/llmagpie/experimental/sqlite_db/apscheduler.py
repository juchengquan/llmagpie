from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .connector import SessionLocal
from .datatype import SessionBase


def get_apscheduler(session_timeout: int = 600):
    """Get apscheduler."""

    def clean_by_time(session_timeout: int = session_timeout):
        """Clean by time."""
        with SessionLocal() as sql_session:
            _q = sql_session.query(SessionBase)
            for session in _q.all():
                timestamp_now = datetime.now().timestamp()
                if timestamp_now - session.time_started > session_timeout:  # in seconds TODO
                    _q.filter(SessionBase.id == session.id).delete()
                    # logger.info(f'Deleting record: {session.id}')
                    sql_session.commit()

    sgtTZObject = timezone(timedelta(hours=8), name="SGT")
    _apscheduler = BackgroundScheduler(timezone="Asia/Singapore")

    _apscheduler.add_job(
        id="clean_by_time",
        func=clean_by_time,
        trigger=CronTrigger.from_crontab("*/1 * * * *", timezone=sgtTZObject),
        kwargs={"session_timeout": session_timeout},
    )

    # _apscheduler.start()
    return _apscheduler
