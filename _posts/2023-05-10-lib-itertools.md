---
layout: post
title: Python | itertools library
subtitle: 
categories: Python-Library
tags: [itertools]
---


## 1. prodcut

- Equivalent to a nested for-loop
- Lexicographic ordering according to the order of the input iterable

### ADT
```python
def product(*iterables, repeat=1):
    return iterable_object
```

### Examples
```python
from itertools import product

x = range(1, 3)
y = 'abc'

list(product(x, repeat=3))
"""[(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2),
    (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)]"""

list(product(x, y))
"""[(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (2, 'c')]"""

list(product(product(x, y), repeat=2)) 
"""[((1, 'a'), (1, 'a')),
 ((1, 'a'), (1, 'b')),
 ...
 ((2, 'c'), (2, 'b')),
 ((2, 'c'), (2, 'c'))]""" # 3 dimensions, 36 elements

product(x, y, x, y) == product(x, y, repeat=2) # False, different object
list(product(x, y, x, y)) == list(product(x, y, repeat=2)) # True, same content
"""[(1, 'a', 1, 'a'),
 (1, 'a', 1, 'b'),
 ...
 (2, 'c', 2, 'b'),
 (2, 'c', 2, 'c')]""" # 2 dimensions, 36 elements
```

## 2. permutations
- Lexicographic ordering according to the order of the input iterable

### ADT
```python
def permutations(iterable, r=None):
    # r = length of return object
    return iterable_object
```

### Examples
```python
from itertools import permutations

y = 'abc'

list(permutations(y))
"""[('a', 'b', 'c'),
 ('a', 'c', 'b'),
 ('b', 'a', 'c'),
 ('b', 'c', 'a'),
 ('c', 'a', 'b'),
 ('c', 'b', 'a')]"""

list(permutations(y, 2))
"""[('a', 'b'), ('a', 'c'), ('b', 'a'),
('b', 'c'), ('c', 'a'), ('c', 'b')]"""
```

## 3. combinations
- Lexicographic ordering according to the order of the input iterable

### ADT
```python
def combinations(iterable, r):
    return iterable_object
```

### Examples
```python
from itertools import combinations

y = 'abc'

list(combinations(y, 2))
"""[('a', 'b'), ('a', 'c'), ('b', 'c')]"""
```

## 4. combinations_with_replacement
- Lexicographic ordering according to the order of the input iterable

### ADT
```python
def combinations_with_replacement(iterable, r):
    return iterable_object
```

### Examples
```python
from itertools import combinations_with_replacement

y = 'abc'

list(combinations_with_replacement(y, 2))
"""[('a', 'a'), ('a', 'b'), ('a', 'c'),
('b', 'b'), ('b', 'c'), ('c', 'c')]"""
```

## 5. zip_longest
- Lexicographic ordering according to the order of the input iterable

### ADT
```python
def zip_longest(*iterables, fillvalue=None):
    return iterable_object
```

### Examples
```python
from itertools import zip_longest

x = 'ab'
y = [3, 4, 5]
 
list(zip(x, y)) # [('a', 3), ('b', 4)]
list(zip(x, y, strict=True)) # ValueError
list(zip_longest(x, y)) # [('a', 3), ('b', 4), (None, 5)]
list(zip_longest(x, y, fillvalue='k'))  # [('a', 3), ('b', 4), ('k', 5)]
```


