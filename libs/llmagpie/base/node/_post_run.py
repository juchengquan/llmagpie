from collections.abc import AsyncGenerator, Generator

from pydantic import BaseModel


def _marshal_iterable(res_iterable: Generator, model: type[BaseModel]) -> Generator:
    for _res in res_iterable:
        if isinstance(_res, dict):
            yield model(**_res if _res else {}).model_dump(exclude_none=True)
        elif isinstance(_res, tuple):
            yield model(
                **{k: v for k, v in zip(model.model_fields.keys(), _res, strict=False)}
                if _res
                else {}
            ).model_dump(exclude_none=True)
        else:
            yield model(
                **{k: v for k, v in zip(model.model_fields.keys(), [_res], strict=False)}
                if _res
                else {}
            ).model_dump(exclude_none=True)


async def _async_marshal_iterable(
    async_res_iterable: AsyncGenerator, model: type[BaseModel]
) -> AsyncGenerator:
    async for _res in async_res_iterable:
        if isinstance(_res, dict):
            yield model(**_res if _res else {}).model_dump(exclude_none=True)
        elif isinstance(_res, tuple):
            yield model(
                **{k: v for k, v in zip(model.model_fields.keys(), _res, strict=False)}
                if _res
                else {}
            ).model_dump(exclude_none=True)
        else:
            yield model(
                **{k: v for k, v in zip(model.model_fields.keys(), [_res], strict=False)}
                if _res
                else {}
            ).model_dump(exclude_none=True)


def post_run(
    res: tuple | dict | Generator | AsyncGenerator, model: type[BaseModel]
) -> dict | Generator | AsyncGenerator:
    if isinstance(res, Generator):
        return _marshal_iterable(res, model)
    elif isinstance(res, AsyncGenerator):
        return _async_marshal_iterable(res, model)
    elif isinstance(res, dict):
        return model(**res if res else {}).model_dump(exclude_none=True)
    elif isinstance(res, tuple):
        return model(
            **{k: v for k, v in zip(model.model_fields.keys(), res, strict=False)} if res else {}
        ).model_dump(exclude_none=True)
    try:
        return model(
            **{k: v for k, v in zip(model.model_fields.keys(), [res], strict=False)} if res else {}
        ).model_dump(exclude_none=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Result type is wrong: {type(res).__name__}.") from exc
