from pydantic import Field
from fastapi import APIRouter
from llmagpie.core.fastapi import FastAPIHandler
from llmagpie.core.logging import fastapi_wrapper
from llmagpie.core.sqlite_db.apscheduler import get_apscheduler
from llmagpie.core.utilities.prometheus import make_metrics_app

from ._base import BasePipelineMixin
# typing
from typing import Dict
from apscheduler.schedulers.background import BackgroundScheduler


class QueryPipelineWithFastAPI(BasePipelineMixin):
    fastapi_app: FastAPIHandler = None  # to be instantiated later
    api_router: APIRouter = Field(default_factory=APIRouter)
    cb_api_router: APIRouter = Field(default_factory=APIRouter)
    app_state: Dict = {}

    apscheduler: BackgroundScheduler = Field(default_factory=get_apscheduler)

    def compile(self) -> "type.Self":
        """compile pipeline.
        """
        self.graph.validate()
        self._instantiate_fastapi()
        self._instantiate_apscheduler()
        return self

    def _instantiate_apscheduler(self):
        if not self.apscheduler.running:
            self.apscheduler.start()
        
    def _instantiate_fastapi(self):
        try:
            self.fastapi_app = FastAPIHandler(
                app_state=self.app_state
            )

            for n, degree in self.graph.in_degree():
                if degree == 0:  # head
                    node = self.graph.nodes[n]["_obj"]
                    assert hasattr(node, "api")
                    assert hasattr(node, "api_callback")

                    self.api_router.add_api_route(
                        path=node.api_route,
                        endpoint=fastapi_wrapper(node.api),
                        methods=["POST"]
                    )
                    self.cb_api_router.add_api_route(
                        path=node.api_route,
                        endpoint=fastapi_wrapper(node.api_callback),
                        methods=["POST"]
                    )

            self.fastapi_app.bind_router(
                self.api_router,
                self.cb_api_router,
            )
            self.fastapi_app.mount(path="/metrics", app=make_metrics_app())
            return self
        except AssertionError as err:
            raise err
        except Exception as err:
            raise err
