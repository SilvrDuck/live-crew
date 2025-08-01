# Python Typing Reference

## Key Concepts

### TypeVar and Generics
For type-safe generic classes:

```python
from typing import TypeVar, Generic

T = TypeVar('T')
# Covariant: when you only return T, never accept it
T_co = TypeVar('T_co', covariant=True)
# Contravariant: when you only accept T, never return it
T_contra = TypeVar('T_contra', contravariant=True)

class Container(Generic[T]):
    def __init__(self, item: T) -> None:
        self.item = item

    def get(self) -> T:
        return self.item
```

### Discriminated Unions with Literal
For type-safe polymorphism:

```python
from typing import Union, Literal, Annotated
from pydantic import Field

class TypeA(BaseModel):
    type: Literal['a']
    value: int

class TypeB(BaseModel):
    type: Literal['b']
    value: str

# Discriminated union
MyUnion = Annotated[Union[TypeA, TypeB], Field(discriminator='type')]
```

### Final for Constants
Mark values as constants:

```python
from typing import Final

MAX_SIZE: Final = 100
PATTERN: Final = re.compile(r'[a-z]+')
```

### When to Use Covariance
- Use covariant (`T_co`) when you only **output** the type (return it)
- Use contravariant (`T_contra`) when you only **input** the type (accept it)
- Use invariant (default) when you both input and output the type

### Best Practices
- Use `Final` for constants
- Use `Literal` types for discriminated unions
- Be careful with covariance - invariant is usually safer
- Use `Generic[T]` for reusable container types
