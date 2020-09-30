#!/bin/bash
gcloud auth revoke --all

while [[ -z "$(gcloud config get-value core/account)" ]]; 
do echo "waiting login" && sleep 2; 
done

while [[ -z "$(gcloud config get-value project)" ]]; 
do echo "waiting project" && sleep 2; 
done

export IMAGE_FAMILY="tf-1-14-cpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="tf-tensorboard-1"
export INSTANCE_TYPE="n1-standard-4"
gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${ZONE}" \
        --image-family="${IMAGE_FAMILY}" \
        --image-project=deeplearning-platform-release \
        --machine-type="${INSTANCE_TYPE}" \
        --boot-disk-size=200GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="proxy-mode=project_editors"

export QWIKLABS_PROJECT_ID=$(gcloud config list project --format "value(core.project)")

gsutil mb gs://$QWIKLABS_PROJECT_ID
gsutil cp model.pkl gs://$QWIKLABS_PROJECT_ID/original/
gsutil cp dist/custom_transforms-0.1.tar.gz gs://$QWIKLABS_PROJECT_ID/

gcloud ai-platform models create census_income_classifier --regions us-central1

export MODEL_NAME="census_income_classifier"
export VERSION_NAME="original"
export MODEL_DIR="gs://$QWIKLABS_PROJECT_ID/original/"
export CUSTOM_CODE_PATH="gs://$QWIKLABS_PROJECT_ID/custom_transforms-0.1.tar.gz"


gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.15 \
  --python-version 3.7 \
  --origin $MODEL_DIR \
  --package-uris $CUSTOM_CODE_PATH \
  --prediction-class predictor.MyPredictor

gcloud ai-platform predict --model=census_income_classifier --json-instances=predictions.json --version=original

gsutil cp model2/model.pkl gs://$QWIKLABS_PROJECT_ID/balanced/

export MODEL_NAME="census_income_classifier"
export VERSION_NAME="balanced"
export MODEL_DIR="gs://$QWIKLABS_PROJECT_ID/balanced/"
export CUSTOM_CODE_PATH="gs://$QWIKLABS_PROJECT_ID/custom_transforms-0.1.tar.gz"

gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.15 \
  --python-version 3.7 \
  --origin $MODEL_DIR \
  --package-uris $CUSTOM_CODE_PATH \
  --prediction-class predictor.MyPredictor




gcloud beta compute ssh --zone "us-west1-b" "tf-tensorboard-1" --quiet --command="sudo mkdir /home/jupyter/training-data-analyst/"

# gcloud beta compute ssh --zone "us-west1-b" "tf-tensorboard-1" --quiet --command="pip3 install xgboost==0.82 --user && pip3 install scikit-learn==0.20.4 --user"
# gcloud beta compute ssh --zone "us-west1-b" "tf-tensorboard-1" --quiet --command="git clone https://github.com/GoogleCloudPlatform/training-data-analyst"
# gcloud beta compute ssh --zone "us-west1-b" "tf-tensorboard-1" --quiet --command="jupyter nbconvert --to notebook --execute training-data-analyst/quests/dei/census/income_xgboost.ipynb"




