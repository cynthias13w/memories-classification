.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y memories || :
	@pip install -e .

install_requirements:
	@pip install -r requirements.txt

#################### BUILD CLOUD API ###################

##Test local api
run_api:
	uvicorn memories.api.fast:app --reload

## Build docker image
build_docker:
	docker build --no-cache --tag=${GCR_IMAGE} .
#
#If ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE
#		 run: sudo pip install --no-cache-dir flask
# or run: pip install tensorflow==2.10.0 --no-cache-dir (or any package)
# or run: pip install tensorflow==2.10.0 --no-cache-dir
#docker images: displays all the images

run_docker:
	docker run -e PORT=8000 -p 8000:8000 ${GCR_IMAGE}

## Image in Google Run Platform

buid_cloud:
	docker build -t ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod .

run_cloud:
	docker run -e PORT=8080 -p 8080:8080 --env-file .env ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

push_cloud:
	docker push ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

deploy_cloud:
	gcloud run deploy --image ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod --memory ${GCR_MEMORY} --region ${GCP_REGION} --env-vars-file .env.yaml

######################## RUN FE ########################
streamlit:
	-@streamlit run app.py
