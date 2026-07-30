.PHONY: test synthetic clean-synthetic

test:
	python -m unittest discover -s tests -v

synthetic:
	python src/generate_synthetic.py --output-dir runs/synthetic_demo --overwrite
	python src/run_pipeline.py --run-dir runs/synthetic_demo --overwrite
	python src/evaluate_synthetic.py --run-dir runs/synthetic_demo

clean-synthetic:
	python -c "import shutil; shutil.rmtree('runs/synthetic_demo', ignore_errors=True)"
