git clone https://github.com/GoogleCloudPlatform/training-data-analyst

pip3 install xgboost==0.82 --user
pip3 install scikit-learn==0.20.4 --user



import datetime
import pickle
import os

import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from witwidget.notebook.visualization import WitWidget, WitConfigBuilder

import custom_transforms

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

os.environ['QWIKLABS_PROJECT_ID'] = 'qwiklabs-gcp-02-2f58f22fcb83'

train_csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

COLUMNS = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occ|upation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income-level'
)

raw_train_data = pd.read_csv(train_csv_path, names=COLUMNS, skipinitialspace=True)
raw_train_data = shuffle(raw_train_data, random_state=4)

test_csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
raw_test_data = pd.read_csv(test_csv_path, names=COLUMNS, skipinitialspace=True, skiprows=1)

raw_train_features = raw_train_data.drop('income-level', axis=1).values
raw_test_features = raw_test_data.drop('income-level', axis=1).values

# Create training labels list
train_labels = (raw_train_data['income-level'] == '>50K').values.astype(int)
test_labels = (raw_test_data['income-level'] == '>50K.').values.astype(int)

numerical_indices = [0, 12]  
categorical_indices = [1, 3, 5, 7]  

p1 = make_pipeline(
    custom_transforms.PositionalSelector(categorical_indices),
    custom_transforms.StripString(),
    custom_transforms.SimpleOneHotEncoder()
)
p2 = make_pipeline(
    custom_transforms.PositionalSelector(numerical_indices),
    StandardScaler()
)
p3 = FeatureUnion([
    ('numericals', p1),
    ('categoricals', p2),
])

pipeline = make_pipeline(
    p3,
    xgb.sklearn.XGBClassifier(max_depth=4)
)
pipeline.fit(raw_train_features, train_labels)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)



python setup.py sdist --formats=gztar

gsutil cp model.pkl gs://$QWIKLABS_PROJECT_ID/original/
gsutil cp dist/custom_transforms-0.1.tar.gz gs://$QWIKLABS_PROJECT_ID/

!gcloud ai-platform models create census_income_classifier --regions us-central1

MODEL_NAME="census_income_classifier"
VERSION_NAME="original"
MODEL_DIR="gs://$QWIKLABS_PROJECT_ID/original/"
CUSTOM_CODE_PATH="gs://$QWIKLABS_PROJECT_ID/custom_transforms-0.1.tar.gz"

gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.15 \
  --python-version 3.7 \
  --origin $MODEL_DIR \
  --package-uris $CUSTOM_CODE_PATH \
  --prediction-class predictor.MyPredictor

%%writefile predictions.json
[25, "Private", 226802, "11th", 7, "Never-married", "Machine-op-inspct", "Own-child", "Black", "Male", 0, 0, 40, "United-States"]

gcloud ai-platform predict --model=census_income_classifier --json-instances=predictions.json --version=original

num_datapoints = 2000  

test_examples = np.hstack(
    (raw_test_features[:num_datapoints], 
     test_labels[:num_datapoints].reshape(-1,1)
    )
)

config_builder = (
    WitConfigBuilder(test_examples.tolist(), COLUMNS)
    .set_ai_platform_model(os.environ['QWIKLABS_PROJECT_ID'], 'census_income_classifier', 'original')
    .set_target_feature('income-level')
    .set_model_type('classification')
    .set_label_vocab(['Under 50K', 'Over 50K'])
)

WitWidget(config_builder, height=800)


bal_data_path = 'https://storage.googleapis.com/cloud-training/dei/balanced_census_data.csv' 
bal_data = pd.read_csv(bal_data_path, names=COLUMNS, skiprows=1)

gsutil cp model.pkl gs://$QWIKLABS_PROJECT_ID/balanced/
    
MODEL_NAME="census_income_classifier"
VERSION_NAME="balanced"
MODEL_DIR="gs://$QWIKLABS_PROJECT_ID/balanced/"
CUSTOM_CODE_PATH="gs://$QWIKLABS_PROJECT_ID/custom_transforms-0.1.tar.gz"

gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.15 \
  --python-version 3.7 \
  --origin $MODEL_DIR \
  --package-uris $CUSTOM_CODE_PATH \
  --prediction-class predictor.MyPredictor

