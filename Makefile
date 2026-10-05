.PHONY: install test lint run clean

install:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts

run:
	python -m action_vision assets/videos/video_1.mp4 assets/videos/video_2.mp4

clean:
	python scripts/clean_outputs.py

