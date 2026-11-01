# Error Handling and HTTP Exceptions

Source: https://fastapi.tiangolo.com/tutorial/handling-errors/

## Raising HTTP exceptions

The standard way to return an HTTP error from an endpoint is to raise
`HTTPException`:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}


@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```

When you raise `HTTPException`, FastAPI stops the endpoint and returns an
error response like:

```json
{
  "detail": "Item not found"
}
```

The `status_code` can be any valid HTTP status code, and `detail` can be
any JSON-serializable value (a string, dict, or list).

## Using status codes from Starlette

Instead of hardcoding numbers, import the status constants:

```python
from fastapi import status

raise HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="Item not found",
    headers={"X-Error": "missing-item"},
)
```

You can also attach custom `headers` to the error response, which is
useful for things like `WWW-Authenticate` on 401 responses.

## Custom exception handlers

To handle a specific exception type across the whole app, register a
handler with `@app.exception_handler`:

```python
from fastapi import Request
from fastapi.responses import JSONResponse


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(UnicornException)
def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )
```

Any endpoint that raises `UnicornException` now returns your custom JSON
response instead of the default 500 error.

## Overriding the default handlers

You can also override how FastAPI handles its own exceptions:

- `HTTPException` via `@app.exception_handler(HTTPException)` — for
  example to return plain-text errors instead of JSON.
- `RequestValidationError` via
  `@app.exception_handler(RequestValidationError)` — to customize the
  422 validation-error response body.

```python
from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation failed", "errors": exc.errors()},
    )
```

## When to use which

- `HTTPException` for expected client errors (404 not found, 403
  forbidden, 401 unauthorized) raised directly in endpoints or
  dependencies.
- Custom exception handlers for domain-specific errors you want to
  translate into consistent API responses.
- Overriding `RequestValidationError` only when you need a different
  error format than FastAPI's default — most APIs should keep the
  default, since clients and docs already understand it.
