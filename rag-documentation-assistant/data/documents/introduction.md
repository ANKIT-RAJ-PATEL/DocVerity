# Introduction to FastAPI

Source: https://fastapi.tiangolo.com/

FastAPI is a modern, fast (high-performance) web framework for building APIs
with Python 3.8+ based on standard Python type hints.

## Key features

- **Fast**: Very high performance, on par with NodeJS and Go, thanks to
  Starlette and Pydantic.
- **Fast to code**: Increases development speed by about 200% to 300%.
- **Fewer bugs**: Reduces human-induced errors by about 40%.
- **Intuitive**: Great editor support with autocompletion everywhere.
- **Easy**: Designed to be easy to use and learn, with excellent documentation.
- **Short**: Minimizes code duplication with sensible defaults and type hints.
- **Robust**: Production-ready code with automatic interactive documentation.
- **Standards-based**: Fully compatible with OpenAPI and JSON Schema.

## How it works

FastAPI uses Python type hints to validate, serialize, and document your API
automatically. You declare request and response shapes with standard Python
types or Pydantic models, and FastAPI handles the rest.

A minimal FastAPI application looks like this:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

Run it with Uvicorn:

```bash
uvicorn main:app --reload
```

## Interactive documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI at `/docs`
- ReDoc at `/redoc`
- OpenAPI JSON schema at `/openapi.json`

The documentation is generated from your type hints, path operations, and
Pydantic models, so it always stays in sync with your code.

## When to use FastAPI

FastAPI is a good choice for REST APIs, microservices, ML model serving,
webhooks, and any backend where request validation, serialization, and
automatic documentation matter. It is built on Starlette for the web parts
and Pydantic for the data parts.
