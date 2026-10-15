# C++ filtering example

```bash
cmake -S cpp -B build
cmake --build build
ctest --test-dir build
```

Standalone C++17 semantic filter. The tested SLAM implementation is Python; this example does not duplicate registration or graph optimization.
