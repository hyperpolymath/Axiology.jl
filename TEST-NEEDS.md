# TEST-NEEDS.md — Axiology.jl

## CRG Grade: C — ACHIEVED 2026-04-04

## Current Test State

| Category | Count | Notes |
|----------|-------|-------|
| Test directories | 2 | Location(s): /test, /tests |
| CI workflows | 19 | Running tests on GitHub Actions |
| Unit tests | Built-in | Julia test framework |

## What's Covered

- [x] Julia unit tests (Pkg.test())
- [x] Module doctests

## Still Missing (for CRG B+)

- [ ] Code coverage reports (codecov integration)
- [ ] Detailed test documentation in CONTRIBUTING.md
- [ ] Integration tests beyond unit tests
- [ ] Performance benchmarking suite

## Run Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```
