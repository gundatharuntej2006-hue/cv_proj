docker-build:
	docker build -t cv_proj:latest .

docker-run:
	docker run -p 8501:8501 cv_proj:latest
