PYTHON ?= python

.PHONY: setup third-party check-third-party test calibrate calibrate-strict

setup:
	$(PYTHON) -m pip install -e '.[validation]'
	bash third_party/bootstrap.sh

third-party:
	bash third_party/bootstrap.sh

check-third-party:
	$(PYTHON) third_party/check.py

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m compileall -q spn_accel_cmodel validation tests examples third_party

calibrate:
	PYTHONPATH=. $(PYTHON) validation/run_calibration_suite.py

calibrate-strict:
	PYTHONPATH=. $(PYTHON) validation/run_calibration_suite.py --strict
