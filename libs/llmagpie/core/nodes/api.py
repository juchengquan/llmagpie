import os
from abc import abstractmethod
# from hydra.utils.common import check_cml_env
from pydantic import field_validator, model_validator


from ._base import BaseNode
# typing
from fastapi import Request
from typing import Optional
import sys
if sys.version_info.minor >= 11:
    from typing import Self
else:
    from typing_extensions import Self

class BaseServiceRetriever(BaseNode):
    service_endpoint: Optional[str] = None
    service_endpoint_name: Optional[str] = None

    @model_validator(mode='after')
    def check_endpoints(self) -> Self:
        assert self.service_endpoint is not None or self.service_endpoint_name is not None, '"service_endpoint" and "service_endpoint_name" cannot be None at same time.'
        return self


class BaseFastAPIService(BaseNode):
    api_route: str

    @field_validator("is_start")
    @classmethod
    def key_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError("is_start must be True.")
        return v

    @abstractmethod
    async def api(self, request: Request, *args, **kwargs):
        """"""


class BaseFastAPIServiceWithCallback(BaseServiceRetriever):
    api_route: str

    @field_validator("is_start")
    @classmethod
    def key_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError("is_start must be True.")
        return v

    # @computed_field(return_type=str)
    @property
    def callback_url(self) -> str:
        """callback_url"""
        # TODO
        # return f'https://public-{os.environ["CDSW_ENGINE_ID"]}.{os.environ["CDSW_DOMAIN"]}/__callback' + self.api_route
        return f'https://{os.environ["APP_NAME"]}.{os.environ["CDSW_DOMAIN"]}/__callback' + self.api_route

    @abstractmethod
    async def api(self, request: Request, *args, **kwargs):
        """"""

    @abstractmethod
    async def api_callback(self, request: Request, *args, **kwargs):
        """"""
