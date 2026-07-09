.PHONY: install check generate gold evaluate analyze mitigate reproduce test dashboard

install:
	pip install -r requirements.txt

check:  ## validate the config without spending any tokens
	python -m src.config config/default.yaml

generate:  ## retrieve + generate answers from all models
	python -m src.generate --config config/default.yaml

gold:  ## build a blind human-labeling template
	python -m scripts.gold template --config config/default.yaml --n 60

evaluate:  ## reference metrics + grounding + pointwise & pairwise judging
	python -m src.evaluate --config config/default.yaml --pairwise-examples 25

analyze:  ## correlation table + judge-bias report
	python -m src.analyze --config config/default.yaml

mitigate:  ## balanced-position + jury, with before/after deltas
	python -m src.mitigate --config config/default.yaml

reproduce:  ## reproduce a published position-bias finding, then extend
	@echo "Point --benchmark at MT-Bench judgments and --domain at your pairwise file:"
	@echo "  python -m experiments.reproduce_position_bias --benchmark <mtbench.jsonl> --domain data/raw/asqa_meta_eval_v1__pairwise.jsonl --target 0.65"

test:  ## run the eval suite
	pytest tests/ -v

dashboard:
	streamlit run dashboard/app.py
