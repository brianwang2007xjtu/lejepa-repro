.PHONY: test test-fast

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -q -m "not slow"
