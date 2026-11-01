# Request Bodies and Pydantic Models

Source: https://fastapi.tiangolo.com/tutorial/body/

## Request body

When you need to send data from a client to your API (for example creating
an item), you send it as a request body. In FastAPI you declare the shape
of the body with a Pydantic model.

```python
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


app = FastAPI()


@app.post("/items/")
def create_item(item: Item):
    return item
```

With this model:

- `name` is required and must be a string.
- `description` is optional.
- `price` is required and must be a float.
- `tax` is optional.

FastAPI automatically validates the JSON body, converts types, and returns
clear 422 errors when validation fails. It also generates the OpenAPI
schema and documentation from the model.

## Using the model inside the endpoint

The `item` parameter is already a validated `Item` instance, not a raw
dict. You can access attributes like `item.name` and `item.price`, and
return it directly — FastAPI serializes it to JSON.

```python
@app.post("/items/")
def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict["price_with_tax"] = price_with_tax
    return item_dict
```

## Request body + path + query parameters

You can mix all three in one endpoint. FastAPI figures out which is which
from the signature:

```python
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item, q: str | None = None):
    result = {"item_id": item_id, **item.model_dump()}
    if q:
        result["q"] = q
    return result
```

- `item_id` is a path parameter (it appears in the path).
- `item` is the request body (it is a Pydantic model).
- `q` is a query parameter (it is a simple type with a default).

## Field validation with Pydantic

Pydantic fields support extra validation:

```python
from pydantic import BaseModel, Field


class Item(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price must be greater than zero")
```

FastAPI surfaces these constraints in the automatic documentation, and
rejects invalid input with a descriptive error before your code runs.
