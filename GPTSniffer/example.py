from collections import OrderedDict

class LRUCache:
"""
Implementation of an LRU (Least Recently Used) Cache.

```
The cache supports O(1) get and put operations by using
OrderedDict to maintain insertion order.
"""

def __init__(self, capacity: int):
    if capacity <= 0:
        raise ValueError("Capacity must be greater than 0")
    self.capacity = capacity
    self.cache = OrderedDict()

def get(self, key: int) -> int:
    """
    Retrieve value associated with key.
    Moves the key to the end to mark it as recently used.
    """
    if key not in self.cache:
        return -1

    self.cache.move_to_end(key)
    return self.cache[key]

def put(self, key: int, value: int) -> None:
    """
    Insert or update the value of the key.
    Evicts least recently used item if capacity exceeded.
    """
    if key in self.cache:
        self.cache.move_to_end(key)

    self.cache[key] = value

    if len(self.cache) > self.capacity:
        evicted_key, evicted_val = self.cache.popitem(last=False)
        print(f"Evicted: {evicted_key} -> {evicted_val}")
```

if **name** == "**main**":
lru = LRUCache(2)

```
lru.put(1, 10)
lru.put(2, 20)
print(lru.get(1))  # Expected 10

lru.put(3, 30)    # Evicts key 2
print(lru.get(2)) # Expected -1

lru.put(4, 40)    # Evicts key 1
print(lru.get(1)) # Expected -1
print(lru.get(3)) # Expected 30
print(lru.get(4)) # Expected 40
```
