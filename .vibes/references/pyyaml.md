# PyYAML Reference

## Key Concepts

### Safe Loading
Always use safe_load for untrusted input:

```python
import yaml

# Safe loading (recommended)
with open('config.yaml', 'r') as f:
    data = yaml.safe_load(f)

# Unsafe loading (avoid)
# data = yaml.load(f, Loader=yaml.FullLoader)  # Don't use this
```

### Dumping Data
Convert Python objects to YAML:

```python
data = {"key": "value", "list": [1, 2, 3]}

# To string
yaml_str = yaml.dump(data)

# To file
with open('output.yaml', 'w') as f:
    yaml.dump(data, f)
```

### Error Handling
Handle YAML parsing errors:

```python
try:
    data = yaml.safe_load(yaml_content)
except yaml.YAMLError as e:
    print(f"YAML parsing error: {e}")
```

### Common Patterns
- Configuration files
- Data serialization
- Test fixtures

### Security
- Never use `yaml.load()` without specifying a Loader
- Always use `yaml.safe_load()` for untrusted input
- `safe_load` only loads basic Python objects (dict, list, str, int, float, bool)
