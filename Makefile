test: 
	python3 -m unittest discover
clean:
	find . -name ".ipynb_checkpoints" -type d  -exec rm -rf {} +
	find . -name "__pycache__" -type d  -exec rm -rf {} +
	find . -name "extracted" -type d  -exec rm -rf {} +
	find . -name "log" -type d  -exec rm -rf {} +
