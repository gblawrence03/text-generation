import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from naive_feedforward import naive_test
from naive_feedforward import naive_train

#naive_test.run(10, model_file='models/test_model_2.keras')
naive_train.train(model_file='models/test_model_2.keras')