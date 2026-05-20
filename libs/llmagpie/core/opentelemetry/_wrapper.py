import json
import os
import warnings
from asyncio import (
    create_task,
)
from asyncio import (
    run as asyncio_run,
)
from collections.abc import Callable
from functools import partial

#
from inspect import BoundArguments, Parameter, iscoroutinefunction, signature

# typing
from typing import Any, Optional, cast

import wrapt
from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass

try:
    from opentelemetry import context, trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.status import Status, StatusCode

    OTEL_ENABLED: bool = True
except ImportError:
    warnings.warn("opentelemetry-python is not installed", stacklevel=2)
    OTEL_ENABLED: bool = False
    trace, context = None, None


_DEFAULT_TRACER_ATTRIBUTES: dict = {
    "openinference.project.name": "OpenTelemeTry Project Name",  # Optional
    "service.name": "OpenTelemeTry Project Name",
}


def _initialize_default_remote_tracer(
    attributes: dict | None = None,
):
    if attributes is None:
        attributes = dict(_DEFAULT_TRACER_ATTRIBUTES)
    # Project based resource
    resource = Resource.create(attributes=attributes)
    remote_tracer_provider = TracerProvider(resource=resource)

    span_exporter = OTLPSpanExporter(endpoint=os.environ["OTEL_COLLECTOR_ENDPOINT"])
    service_span_processor = BatchSpanProcessor(span_exporter=span_exporter)

    remote_tracer_provider.add_span_processor(service_span_processor)  # can add multiple
    trace.set_tracer_provider(remote_tracer_provider)  # IMPORTANT: for default, set only once


def _get_default_tracer(tracer_provider: Optional["TracerProvider"] = None):
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
    """
    This class is a decorator that wraps a function and adds OpenTelemetry tracing.

    If opentelemetry is enabled, it creates a span for the function call, adds input and output
    attributes to the span, and sets the span status to OK or ERROR based on whether the function
    executes successfully.  If opentelemetry is not enabled, it does nothing.  The decorator
    handles both synchronous and asynchronous functions.  It also handles Pydantic models as
    function outputs, converting them to dictionaries before adding them to the span attributes.
    """

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

            with self._tracer.start_as_current_span(name=name) as span:
                span.set_attributes(
                    {
                        "input.value": json.dumps(func_arguments, default=str, ensure_ascii=False),
                    }
                )
                try:
                    response = func(*args, **kwargs)
                    if isinstance(response, (ModelMetaclass, BaseModel)):
                        response = cast(BaseModel, response).model_dump()
                    else:
                        assert isinstance(response, dict), (
                            f"response type is wrong: {type(response)}"
                        )
                    span.set_attributes(
                        {
                            "output.value": json.dumps(response, default=str, ensure_ascii=False),
                        }
                    )
                    span.set_status(StatusCode.OK)
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

            return response

        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            name = self._get_instance_info(instance)

            func_bound_args = _get_bound_arguments(func, *args, **kwargs)
            func_arguments = func_bound_args.kwargs

            with self._tracer.start_as_current_span(name=name) as span:
                span.set_attributes(
                    {
                        "openinference.span.kind": instance.connectable_type
                        if hasattr(instance, "connectable_type")
                        else "LLM",
                        "input.value": json.dumps(func_arguments, default=str, ensure_ascii=False),
                    }
                )
                try:
                    response = await func(*args, **kwargs)
                    if isinstance(response, (ModelMetaclass, BaseModel)):
                        response = cast(BaseModel, response).model_dump()
                    else:
                        assert isinstance(response, dict), "response type is wrong"
                    span.set_attributes(
                        {
                            "output.value": json.dumps(response, default=str, ensure_ascii=False),
                        }
                    )
                    span.set_status(StatusCode.OK)
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
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
            return func(*args, **kwargs)

        @wrapt.decorator
        async def async_wrapper(func, instance, args, kwargs):
            return await func(*args, **kwargs)

        if iscoroutinefunction(func):
            return async_wrapper(func)  # type: ignore
        return wrapper(func)  # type: ignore


if os.getenv("OTEL_COLLECTOR_ENDPOINT"):
    _initialize_default_remote_tracer()
    opentelemetry_tracer = WrapDecorator(tracer=_get_default_tracer())
    # OTEL_ENABLED stays True (set in the import-try block above)
else:
    opentelemetry_tracer = EmptyWrapDecorator()
    OTEL_ENABLED = False

if __name__ == "__main__":
    import uuid
    from abc import abstractmethod

    class A:
        _id = uuid.uuid4().hex

        @opentelemetry_tracer
        async def execute(self, *args, **kwargs):
            response = await self._run(*args, **kwargs)
            return response
            # return self._run(*args, **kwargs)

        @abstractmethod
        async def _run(self, *args, **kwargs): ...

    class B(A):
        async def _run(self, *args, **kwargs):
            return {**kwargs}

    class BB(A):
        async def _run(self, *args, **kwargs):
            return {**kwargs}

    class C(A):
        async def _run(self, clses, *args, **kwargs):
            task_list = []
            for c in clses:
                task_list.append(create_task(c.execute(*args, **kwargs)))
            # for coro in as_completed(task_list):
            #     _ = await coro
            return {**kwargs}

    asyncio_run(C().execute([B(), BB(), B()], k=1))
