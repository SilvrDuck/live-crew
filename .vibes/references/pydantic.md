# Pydantic V2 Reference

## Key Concepts

### ConfigDict
Modern Pydantic V2 way to configure model behavior. Replaces the old `Config` class.

```python
from pydantic import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,      # Make immutable
        extra="forbid",   # Reject unknown fields
        str_strip_whitespace=True,  # Auto-strip strings
        validate_assignment=True,   # Validate on assignment
    )
```

### Field Validators
Use `@field_validator` decorator for custom validation:

```python
from pydantic import field_validator

@field_validator('field_name')
@classmethod
def validate_field(cls, value):
    # validation logic
    return value
```

### Discriminated Unions
For type-safe polymorphism:

```python
from typing import Annotated, Union, Literal
from pydantic import Field

class TypeA(BaseModel):
    type: Literal['a']
    # fields...

class TypeB(BaseModel):
    type: Literal['b']
    # fields...

UnionType = Annotated[Union[TypeA, TypeB], Field(discriminator='type')]
```

### Generic Models
For type-safe payload handling:

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar('T')

class Container(BaseModel, Generic[T]):
    payload: T
```

### Performance Considerations
- Use `ConfigDict(frozen=True)` for immutable models
- Use `extra="forbid"` to catch typos early
- Pre-compile regex patterns for validation
- Consider `validate_assignment=False` for performance if not needed

### Built-in String Validation
Pydantic V2 provides powerful built-in string validation through Field constraints:

```python
from pydantic import Field
import re

class MyModel(BaseModel):
    # Pattern validation (faster than custom regex)
    identifier: str = Field(pattern=r'^[a-zA-Z0-9_]+$')

    # Length constraints
    name: str = Field(min_length=1, max_length=50)

    # Combined constraints
    stream_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$'
    )
```

### Built-in Validator Types
Pydantic includes many specialized types:

```python
from pydantic import EmailStr, HttpUrl, UUID4, PositiveInt
from pydantic.types import constr, conint

# Constrained string with pattern
KindStr = constr(pattern=r'^[a-zA-Z0-9_]+$', min_length=1, max_length=50)

# Constrained integer
TTLInt = conint(gt=0, le=300_000)

class MyModel(BaseModel):
    kind: KindStr
    ttl_ms: TTLInt
```

### Performance Benefits
- Built-in `pattern` validation uses Rust-based regex engine (much faster)
- No need for custom validator functions for simple pattern matching
- Automatic error messages with field context
- Better JSON schema generation

### Best Practices
- Always use ConfigDict instead of Config class
- Prefer built-in Field constraints over custom validators for simple cases
- Use `pattern` Field constraint instead of custom regex validation
- Use field validators for complex validation logic only
- Leverage discriminated unions for polymorphism
- Use Generic types for reusable models
- Keep validation logic in pure functions when possible
