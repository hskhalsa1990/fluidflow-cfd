.PHONY: 3d backwater jump pump open-channel blasius network install test lint convection cavity taylor-green shear channel pipe convergence diagrams clean
install:
	python -m pip install -r requirements-dev.txt
test:
	python -m pytest -q
lint:
	ruff check main.py test_main.py make_diagrams.py make_3d.py --select E9,F,I
cavity:
	python main.py cavity --re 100 --n 65
taylor-green:
	python main.py taylor-green
shear:
	python main.py shear
channel:
	python main.py channel
pipe:
	python main.py pipe --diameter 0.05 --length 100 --velocity 2 --minor-k 2.5 --efficiency 0.7
convection:
	python main.py convection --ra 1e4 --n 41
backwater:
	python main.py backwater --flow 5 --width 3 --slope 0.001 --y-control 2 --length 3000
jump:
	python main.py jump --flow 5 --width 3 --y1 0.4
pump:
	python main.py pump --shutoff-head 40 --design-flow 0.01 --design-head 30 --diameter 0.1 --length 200 --static-head 10
open-channel:
	python main.py open-channel --flow 5 --width 3 --slope 0.001
blasius:
	python main.py blasius --u-inf 5 --nu 1.5e-5 --x 0.5 --length 1
network:
	python main.py network
convergence:
	python main.py convergence --re 100 --grids 17 33 65
diagrams:
	python make_diagrams.py
3d:
	python make_3d.py
clean:
	rm -rf outputs ci_out __pycache__ .pytest_cache .ruff_cache
