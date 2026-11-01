.PHONY: setup pilot first corpus test dry-run

setup:
	./setup.sh

pilot:
	./run_short_pilot.sh

first:
	./run_first_run.sh

corpus:
	./run_corpus_pass.sh

test:
	.venv/bin/pytest -q

dry-run:
	.venv/bin/python -m recipe_dapt.pipeline --config configs/short_pilot.yaml --dry-run
