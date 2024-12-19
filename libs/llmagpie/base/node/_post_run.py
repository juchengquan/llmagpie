from pydantic import BaseModel
from typing import Dict, Tuple, Union, Generator, AsyncGenerator

def _marshal_iterable(res_iterable: Generator, model: type[BaseModel]) -> Generator:
    for _res in res_iterable:
        if isinstance(_res, Dict):
            yield model(**_res if _res else {}).model_dump(exclude_none=True)
        elif isinstance(_res, Tuple):
            yield model(**{k:v for k, v in zip(model.model_fields.keys(), _res)} if _res else {}).model_dump(exclude_none=True)
        else:
            yield model(**{k:v for k, v in zip(model.model_fields.keys(), [_res])} if _res else {}).model_dump(exclude_none=True)

async def _async_marshal_iterable(async_res_iterable: AsyncGenerator, model: type[BaseModel]) -> AsyncGenerator:
    async for _res in async_res_iterable:
        if isinstance(_res, Dict):
            yield model(**_res if _res else {}).model_dump(exclude_none=True)
        elif isinstance(_res, Tuple):
            yield model(**{k:v for k, v in zip(model.model_fields.keys(), _res)} if _res else {}).model_dump(exclude_none=True)
        else:
            yield model(**{k:v for k, v in zip(model.model_fields.keys(), [_res])} if _res else {}).model_dump(exclude_none=True)


def post_run(res: Union[Tuple, Dict, Generator, AsyncGenerator], model: type[BaseModel]) \
    -> Union[Dict, Generator, AsyncGenerator]:
    if isinstance(res, Generator):
        return _marshal_iterable(res, model)
    elif isinstance(res, AsyncGenerator):
        return _async_marshal_iterable(res, model)
    elif isinstance(res, Dict):
        return model(**res if res else {}).model_dump(exclude_none=True)
    elif isinstance(res, Tuple):
        return model(**{k:v for k, v in zip(model.model_fields.keys(), res)} if res else {}).model_dump(exclude_none=True)
    try:
        return model(**{k:v for k, v in zip(model.model_fields.keys(), [res])} if res else {}).model_dump(exclude_none=True)
    except:
        raise TypeError("Result type is wrong.")
