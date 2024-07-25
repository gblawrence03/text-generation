import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from naive_feedforward import naive_test, naive_train
from char_embed_feedforward import char_embed_train, char_embed_test

#naive_test.run(10, model_file='models/test_model_2.keras')
#naive_train.train(model_file='models/test_model_2.keras')
char_embed_train.train(model_file='models/test_model_3.keras')

#char_embed_test.run(10, model_file='models/test_model_3.keras')