
import json
import base64
from fastapi import (
    APIRouter,
    FastAPI as FastAPIOriginal,
    Request,
    HTTPException
)
from contextlib import asynccontextmanager

from llmagpie.core.sqlite_db.connector import SessionLocal
from llmagpie.core.sqlite_db.datatype import SessionBase, AppStateBase
from llmagpie.core.logging import get_or_create_logger
# typing
from typing import Dict


class FastAPIHandler(FastAPIOriginal):
    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        app_state: dict = kwargs.pop("app_state", {})

        super().__init__(
            *args,
            lifespan=lifespan,
            **kwargs
        )
        with SessionLocal() as sql_session:
            for _key, _value in app_state.items():
                _value_b64 = base64.b64encode(json.dumps(_value).encode('utf-8'))
                sql_session.add(
                    AppStateBase(
                        key=_key,
                        value=_value_b64,
                    )
                )
                sql_session.commit()

        # bind health function
        self.add_api_route("/", lambda: {"status": "OK"}, methods=["GET", "POST"])
        self.include_router(ManagementRouter(), tags=["Management"], prefix="/manage")

    def bind_router(
        self,
        router: APIRouter,
        cb_router: APIRouter,
        prefix: str = "/api",
        cb_prefix: str = "/__callback",
    ):
        self.include_router(router, tags=["API"], prefix=prefix,)
        self.include_router(cb_router, tags=["Callback"], prefix=cb_prefix)


class ManagementRouter(APIRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_api_route(path="/", endpoint=lambda: {"status": "OK"}, methods=["GET", "POST"])
        self.add_api_route(path="/app_state/{asset_name}/{action}", endpoint=self._api_app_state_action, methods=["GET", "POST"])
        self.add_api_route(path="/object_store/{action}/{session_id}", endpoint=self._api_object_store_flush, methods=["GET", "POST"])
    
        self.sql_session = SessionLocal()
        self.logger = get_or_create_logger(self.__class__.__name__)

    async def _api_app_state_action(self, request: Request, asset_name: str, action: str):
        try:
            _q = self.sql_session.query(AppStateBase).filter(AppStateBase.key == asset_name)
            assert _q.count() == 1
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'{repr(exc)}: {_q.count()}')
        
        try:
            _app_state_dict: dict = json.loads(base64.b64decode(_q.one().value).decode('utf-8'))
            assert isinstance(_app_state_dict, dict)
            
            if action == "list":
                return _app_state_dict
            elif action == "update":
                content = await request.json()
                for key, value in content.items():
                    if key not in _app_state_dict.keys():
                        raise HTTPException(status_code=400, detail=f"Key:{key} not valid.")
                    _app_state_dict[key] = value

                _app_state_base64 = base64.b64encode(json.dumps(_app_state_dict).encode('utf-8'))
                _r = _q.update({
                    "key": asset_name,
                    "value": _app_state_base64,
                })
                self.sql_session.commit()

                _q = self.sql_session.query(AppStateBase).filter(AppStateBase.key == asset_name)
                assert _q.count() == 1
                _app_state_dict: dict = json.loads(base64.b64decode(_q.one().value).decode('utf-8'))
                return _app_state_dict
            else:
                raise Exception("Action is wrong.")

        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'{repr(exc)}: {_q.count()}')

    async def _api_object_store_list(self, request: Request):
        result = {}
        for sessions in self.sql_session.query(SessionBase).all():
            result[sessions.id] = sessions.__dict__
            result[sessions.id]["api_request"] = json.loads(base64.b64decode(result[sessions.id]["api_request"]).decode('utf-8'))
            result[sessions.id]["prompt_template"] = json.loads(base64.b64decode(result[sessions.id]["prompt_template"]).decode('utf-8'))
        return result

    async def _api_object_store_flush(self, request: Request, action: str, session_id: str, asset_name: str = "object_store"):
        _q = self.sql_session.query(SessionBase)

        if action == "list" and session_id == "all":
            result = {}
            for sessions in self.sql_session.query(SessionBase).all():
                result[sessions.id] = sessions.__dict__
                result[sessions.id]["api_request"] = json.loads(base64.b64decode(result[sessions.id]["api_request"]).decode('utf-8'))
                result[sessions.id]["prompt_template"] = json.loads(base64.b64decode(result[sessions.id]["prompt_template"]).decode('utf-8'))
            return result

        if action == "flush":
            if session_id == "all":
                res = _q.delete()
            else:
                res = _q.filter(SessionBase.id == session_id).delete()
            
            self.sql_session.commit()
            self.sql_session.rollback()
            
            result = {}
            for sessions in self.sql_session.query(SessionBase).all():
                result[sessions.id] = sessions.__dict__
                result[sessions.id]["api_request"] = json.loads(base64.b64decode(result[sessions.id]["api_request"]).decode('utf-8'))
                result[sessions.id]["prompt_template"] = json.loads(base64.b64decode(result[sessions.id]["prompt_template"]).decode('utf-8'))
            return result

        else:
            raise HTTPException(status_code=400, detail=f"Action:{action};Asset:{asset_name} not valid")


@asynccontextmanager
async def lifespan(
    application: FastAPIHandler,
    **kwargs,
):
    """Lifespan function for handler.
    """
    for k, v in kwargs.items():
        setattr(application.state, k, v)

    yield {}
    for k, _ in kwargs.items():
        setattr(application.state, k, None)
