.PHONY: fmt codestyle test
ADAPTIVE=main_glue_adaptive.py utils/adaptive

codestyle:
	ruff check $(ADAPTIVE)
	ruff format --diff $(ADAPTIVE) || (echo "Formatting issues detected. See diff above." && exit 1)
fmt:
	ruff format $(ADAPTIVE)
	ruff check --fix $(ADAPTIVE)

test: codestyle
	echo "No unit tests yet"
