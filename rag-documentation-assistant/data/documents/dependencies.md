# Dependency Injection in FastAPI

Source: https://fastapi.tiangolo.com/tutorial/dependencies/

## What is dependency injection?

Dependency injection is a way to declare things your endpoint needs —
"dependencies" — and let FastAPI provide them automatically. A dependency
is just a callable (usually a function) that FastAPI runs before your
endpoint, passing its return value into the endpoint.

This is useful for:

- Shared logic (authentication, database sessions).
- Enforcing requirements (only allow a request if some condition holds).
- Reducing code duplication across endpoints.

## First example

```python
from fastapi import Depends, FastAPI

app = FastAPI()


def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items/")
def read_items(commons: dict = Depends(common_parameters)):
    return commons


@app.get("/users/")
def read_users(commons: dict = Depends(common_parameters)):
    return commons
```

Both endpoints share the same query parameters without repeating code.
`Depends(common_parameters)` tells FastAPI to call `common_parameters`
first and inject its result.

## Dependencies with yield (setup and teardown)

If your dependency uses `yield` instead of `return`, FastAPI treats the
code before `yield` as setup and the code after `yield` as teardown. This
is the standard pattern for database sessions:

```python
from collections.abc import Generator


def get_db() -> Generator:
    db = create_session()
    try:
        yield db
    finally:
        db.close()


@app.get("/items/")
def read_items(db=Depends(get_db)):
    return db.query(Item).all()
```

The session is created, passed to the endpoint, and always closed after
the request finishes — even if the endpoint raises an exception.

## Classes as dependencies

You can also use a class instance as a dependency, which is handy when
the dependency itself takes parameters:

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items/")
def read_items(commons: CommonQueryParams = Depends()):
    return commons
```

FastAPI inspects the class `__init__` signature the same way it inspects
a function signature.

## Sub-dependencies

Dependencies can themselves depend on other dependencies, forming a chain.
FastAPI resolves the whole chain and caches each dependency's result per
request, so a shared sub-dependency runs only once per request.

```python
def get_current_user(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user


def get_active_user(current_user=Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
```

## Global dependencies

You can apply a dependency to an entire application, router, or group of
routes instead of repeating `Depends()` on every endpoint:

```python
app = FastAPI(dependencies=[Depends(verify_api_key)])
```

This is commonly used for API-key or authentication checks that must run
for every request.
