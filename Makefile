PROJECT_NAME=assignment1
DOCKER_IMAGE_NAME=$(PROJECT_NAME)_image
install:
	pip install -r requirements.txt

build:
	docker build -t $(DOCKER_IMAGE_NAME) .