.PHONY: test synthetic experiment-synthetic clean-synthetic

PYTHON ?= python3
test:
	$(PYTHON) -m unittest discover -s tests -v

synthetic:
	$(PYTHON) src/generate_synthetic.py --output-dir runs/synthetic_demo --overwrite
	$(PYTHON) src/run_pipeline.py --run-dir runs/synthetic_demo --overwrite
	$(PYTHON) src/evaluate_synthetic.py --run-dir runs/synthetic_demo

experiment-synthetic: synthetic
	$(PYTHON) src/compare_experiments.py \
		--run-dir runs/synthetic_demo \
		--configs configs/baseline_v1.json configs/hard_filter_v1_soft_ranking_v1.json \
		--labels runs/synthetic_demo/ground_truth.csv

clean-synthetic:
	$(PYTHON) -c "import shutil; shutil.rmtree('runs/synthetic_demo', ignore_errors=True)"
