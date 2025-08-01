# pytest Reference

## Key Concepts

### Fixtures
Reusable test setup and teardown:

```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_something(sample_data):
    assert sample_data["key"] == "value"
```

### Parametrized Tests
Run same test with different inputs:

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_upper(input, expected):
    assert input.upper() == expected
```

### Exception Testing
Test that exceptions are raised:

```python
def test_exception():
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("test error")
    assert "test error" in str(exc_info.value)
```

### Test Classes
Group related tests:

```python
class TestUser:
    def test_creation(self):
        # test code
        pass

    def test_validation(self):
        # test code
        pass
```

### Best Practices
- Use descriptive test names
- Follow DAMP principle (Descriptive And Meaningful Phrases) over DRY in tests
- Group tests logically with classes
- Use fixtures for common setup
- Test both happy path and edge cases
