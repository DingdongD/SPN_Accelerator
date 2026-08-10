PYTHON ?= python

.PHONY: setup setup-core third-party third-party-core check-third-party check-third-party-core test calibrate calibrate-strict

setup:
	$(PYTHON) -m pip install -e '.[validation]'
	bash third_party/bootstrap.sh

setup-core:
	$(PYTHON) -m pip install -e '.[validation]'
	bash third_party/bootstrap.sh --core

third-party:
	bash third_party/bootstrap.sh

third-party-core:
	bash third_party/bootstrap.sh --core

check-third-party:
	$(PYTHON) third_party/check.py

check-third-party-core:
	$(PYTHON) third_party/check.py --core

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m compileall -q spn_accel_cmodel validation tests examples third_party

calibrate:
	$(PYTHON) third_party/check.py --pins-only
	PYTHONPATH=. $(PYTHON) validation/run_calibration_suite.py

calibrate-strict:
	$(PYTHON) third_party/check.py --pins-only
	PYTHONPATH=. $(PYTHON) validation/run_calibration_suite.py --strict
