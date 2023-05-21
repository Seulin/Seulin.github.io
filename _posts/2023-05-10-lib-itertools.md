---
layout: post
title: Python | itertools library
subtitle: 
categories: Python-Library
tags: [itertools]
---

```python
from itertools import product, permutations, combinations, combinations_with_replacement

x = [1, 2]
y = 'abc'


list(product(x, repeat=2)) # equivalent to a nested for-loop
"""[(1, 1), (1, 2), (2, 1), (2, 2)]"""

list(product(x, y)) # 'repeat=1' is omitted
"""[(1, 'a'), (1, 'b'), (1, 'c'), (2, 'a'), (2, 'b'), (2, 'c')]"""

list(product(product(x, y), repeat=2)) 
"""[((1, 'a'), (1, 'a')),
 ((1, 'a'), (1, 'b')),
 ...
 ((2, 'c'), (2, 'b')),
 ((2, 'c'), (2, 'c'))]""" # 36 elements

product(x, y, x, y) == product(x, y, repeat=2) # False, different object
list(product(x, y, x, y)) == list(product(x, y, repeat=2)) # True, same content
"""[(1, 'a', 1, 'a'),
 (1, 'a', 1, 'b'),
 ...
 (2, 'c', 2, 'b'),
 (2, 'c', 2, 'c')]""" # 36 elements

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

list(combinations(y, 2)) # second parameter is mandatory
"""[('a', 'b'), ('a', 'c'), ('b', 'c')]"""

list(combinations_with_replacement(y, 2))
"""[('a', 'a'), ('a', 'b'), ('a', 'c'),
('b', 'b'), ('b', 'c'), ('c', 'c')]"""
```