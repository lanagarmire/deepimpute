PYTHON_VERSION = ''

reformat:
	# Requires black: pip3 install black
	black deepimpute
	black tests
	black setup.py

test:
	python${PYTHON_VERSION} ./tests/normalizer_test.py
	python${PYTHON_VERSION} ./tests/util_test.py
	python${PYTHON_VERSION} ./tests/maskedArrays_test.py
	python${PYTHON_VERSION} ./tests/net_test.py
	python${PYTHON_VERSION} ./tests/multinet_test.py
	python${PYTHON_VERSION} ./tests/deepImpute_test.py

coverage:
	coverage${PYTHON_VERSION} run -p --source ./deepimpute/ ./tests/deepImpute_test.py
	coverage${PYTHON_VERSION} run -p --source ./deepimpute/ ./tests/util_test.py
	coverage${PYTHON_VERSION} run -p --source ./deepimpute/ ./tests/maskedArrays_test.py
	coverage${PYTHON_VERSION} run -p --source ./deepimpute/ ./tests/normalizer_test.py
	coverage${PYTHON_VERSION} run -p --source ./deepimpute/ ./tests/multinet_test.py
	coverage${PYTHON_VERSION} run -p --source ./deepimpute/ ./tests/net_test.py
	coverage${PYTHON_VERSION} combine
	coverage${PYTHON_VERSION} html
	echo "Results in: file://${CURDIR}/htmlcov/index.html"

test-mp-hang-bug:
	python${PYTHON_VERSION} -m unittest discover -s deepimpute -p '*_test.py'
