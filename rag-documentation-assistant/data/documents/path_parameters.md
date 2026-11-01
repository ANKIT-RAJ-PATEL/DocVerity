# Path Parameters and Query Parameters

Source: https://fastapi.tiangolo.com/tutorial/path-params/
Also see: https://fastapi.tiangolo.com/tutorial/query-params-str-validations/

## Path parameters

Path parameters are declared in the path using curly braces. FastAPI passes
their values to your function as arguments with automatic type conversion
and validation.

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

If you call `/items/42`, `item_id` will be the integer `42`. If you call
`/items/foo`, FastAPI returns a 422 validation error because `foo` is not
a valid integer.

### Path parameter with type hints

You can use standard Python types: `int`, `float`, `str`, `bool`, and also
Pydantic-validated types. FastAPI validates the value before your code runs.

### Order matters

When you have overlapping paths, define the more specific path first:

```python
@app.get("/users/me")
def read_current_user():
    return {"user": "current"}


@app.get("/users/{user_id}")
def read_user(user_id: str):
    return {"user_id": user_id}
```

If `/users/{user_id}` came first, a request to `/users/me` would match it
with `user_id="me"`.

### Enum path parameters

You can restrict a path parameter to a fixed set of values with an Enum:

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"


@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

## Query parameters

Parameters that are not part of the path are treated as query parameters.

```python
@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

A request to `/items/?skip=20&limit=5` gives `skip=20, limit=5`.
Query parameters are optional when they have default values.

### Required vs optional query parameters

```python
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None, short: bool = False):
    result = {"item_id": item_id}
    if q:
        result["q"] = q
    if not short:
        result["description"] = "A longer description"
    return result
```

Here `q` is optional (defaults to `None`) and `short` defaults to `False`.

### Query parameter validation

Use `Query` from FastAPI to add validation and metadata:

```python
from fastapi import Query

@app.get("/items/")
def read_items(q: str | None = Query(default=None, min_length=3, max_length=50)):
    return {"q": q}
```

You can also declare a query parameter as a list to accept repeated keys:

```python
@app.get("/items/")
def read_items(q: list[str] | None = Query(default=None)):
    return {"q": q}
```

A request to `/items/?q=a&q=b` gives `q=["a", "b"]`.
