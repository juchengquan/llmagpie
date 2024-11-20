import os
import wrapt
import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
# 
from inspect import BoundArguments, Parameter, signature, iscoroutinefunction
from opentelemetry.trace.status import Status, StatusCode
from functools import partial
# typing
from typing import Union, Callable, Dict, Any, Optional, cast
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass


def _initialize_default_remote_tracer(
    attributes: Dict = {
            "openinference.project.name": "OpenTelemeTry Project Name",  # Optional
            "service.name": "OpenTelemeTry Project Name"
        }
    ):
    # Project based resource
    resource = Resource.create(
        attributes=attributes
    )
    remote_tracer_provider = TracerProvider(resource=resource)

    span_exporter = OTLPSpanExporter(endpoint=os.environ["OTEL_COLLECTOR_ENDPOINT"])
    service_span_processor = BatchSpanProcessor(span_exporter=span_exporter)

    remote_tracer_provider.add_span_processor(service_span_processor)  # can add multiple 
    trace.set_tracer_provider(remote_tracer_provider)  # IMPORTANT: for default, set only once

def _get_default_tracer(tracer_provider: Optional[TracerProvider] = None):
    # get default tracer
    tracer = trace.get_tracer("my.tracer", tracer_provider=tracer_provider)
    # from opentelemetry.trace.status import Status, StatusCode
    # from openinference.instrumentation import OITracer, TraceConfig

    # tracer = OITracer(
    #     trace.get_tracer("my.tracer", tracer_provider=remote_tracer_provider),
    #     config=TraceConfig(),
    # )
    return tracer

def _get_bound_arguments(function: Callable[..., Any], *args: Any, **kwargs: Any) -> BoundArguments:
    """
    Safely returns bound arguments from the current context.
    """
    sig = signature(function)
    accepts_arbitrary_kwargs = any(
        param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_arbitrary_kwargs or key in sig.parameters
    }
    return sig.bind(*args, **valid_kwargs)


class WrapDecorator:
    def __init__(self, tracer):
        self._tracer = tracer
    
    @classmethod
    def _get_instance_info(cls, instance):
        if hasattr(instance, "name"):
            return instance.name
        instance_name = instance.__class__.__name__
        instance_id = instance._id
        name = f"{instance_name} {instance_id}"
        return name
        
    def __call__(self, func=None):
        if func is None:
            return partial(self.__call__)

        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            name = self._get_instance_info(instance)
            
            func_bound_args = _get_bound_arguments(func, *args, **kwargs)
            func_arguments = func_bound_args.kwargs
            # span = self._tracer.start_span(name=name)
            with self._tracer.start_as_current_span(name=name) as span:
                span.set_attributes({
                    "input.value": json.dumps(func_arguments, default=str, ensure_ascii=False),
                })
                try:
                    response = func(*args, **kwargs)
                    # TODO cqju
                    if isinstance(response, Union[ModelMetaclass, BaseModel]):
                        response = cast(BaseModel, response).model_dump()
                    span.set_attributes({
                        "output.value": json.dumps(response, default=str, ensure_ascii=False),  
                    })
                    span.set_status(StatusCode.OK)
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise exc

            # span.end()
            return response

        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            name = self._get_instance_info(instance)
            
            func_bound_args = _get_bound_arguments(func, *args, **kwargs)
            func_arguments = func_bound_args.kwargs

            # span = self._tracer.start_span(name=name)
            with self._tracer.start_as_current_span(name=name) as span:
                # TODO: cqju remove for opentelemetry
                span.set_attributes({
                    "openinference.span.kind": instance.node_type if hasattr(instance, "node_type") else "LLM",
                })
                span.set_attributes({
                    "input.value": json.dumps(func_arguments, default=str, ensure_ascii=False),
                })
                try:
                    response = await func(*args, **kwargs)
                    # TODO cqju
                    if isinstance(response, Union[ModelMetaclass, BaseModel]):
                        response = cast(BaseModel, response).model_dump()
                    span.set_attributes({
                        "output.value": json.dumps(response, default=str, ensure_ascii=False), 
                    })
                    span.set_status(StatusCode.OK)
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise exc
            # span.end()
            return response
        
        if iscoroutinefunction(func):
            return async_wrapper(func)  # type: ignore
        return wrapper(func)  # type: ignore


class EmptyWrapDecorator:
    _tracer: Any = None
    
    def __call__(self, func=None):
        if func is None:
            return partial(self.__call__)

        @wrapt.decorator
        def wrapper(func, instance, args, kwargs):
            try:
                response = func(*args, **kwargs)
                return response
            except Exception as exc:
                raise exc
        
        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            try:
                response = await func(*args, **kwargs)
                return response
            except Exception as exc:
                raise exc

        if iscoroutinefunction(func):
            return async_wrapper(func)  # type: ignore
        return wrapper(func)  # type: ignore

OTEL_ENABLED: bool = False
if os.getenv("OTEL_COLLECTOR_ENDPOINT"):
    _initialize_default_remote_tracer()
    opentelemetry_tracer = WrapDecorator(tracer=_get_default_tracer())
    OTEL_ENABLED = True
else:
    opentelemetry_tracer = EmptyWrapDecorator()

if __name__ == "__main__":
    from abc import abstractmethod
    import uuid
    import asyncio

    class A:
        _id = uuid.uuid4().hex
        
        @opentelemetry_tracer
        async def execute(self, *args, **kwargs):
            response = await self._run(*args, **kwargs)
            return response
            # return self._run(*args, **kwargs)

        @abstractmethod
        async def _run(self, *args, **kwargs):
            ...

    class B(A):
        async def _run(self, *args, **kwargs):
            return {**kwargs}
        
    class BB(A):
        async def _run(self, *args, **kwargs):
            return {**kwargs} 

    class C(A):
        async def _run(self, clses, *args, **kwargs):
            task_list = []
            for cls in clses:
                task_list.append(
                    asyncio.create_task(cls.execute(*args, **kwargs))
                )
            # for coro in asyncio.as_completed(task_list):
            #     _ = await coro
            return {**kwargs}


    asyncio.run(C().execute([B(), BB(), B()], k=1))