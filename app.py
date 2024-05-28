import tensorflow
from flask import Flask, render_template, request
import nltk
import numpy as np
import pandas as pd

nltk.download('punkt')

app = Flask(__name__)

data = pd.read_csv('postagging-aahar-3.csv', encoding='utf-8')
data = data.fillna(method="ffill")  # Deal with N/A

tags = list(set(data["Pos"].values))  # Read POS value
words = list(set(data["Word"].values))

# Load word to id mapping
word2id = {w: i for i, w in enumerate(words)}  # Load your word to id mapping here
tag2id = {t: i for i, t in enumerate(tags)}

config={'name': 'model',
 'trainable': True,
 'layers': [{'module': 'keras.layers',
   'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, None),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'registered_name': None,
   'name': 'input_1',
   'inbound_nodes': []},
  {'module': 'keras.layers',
   'class_name': 'Embedding',
   'config': {'name': 'embedding',
    'trainable': True,
    'dtype': 'float32',
    'batch_input_shape': (None, None),
    'input_dim': 1963,
    'output_dim': 50,
    'embeddings_initializer': {'module': 'keras.initializers',
     'class_name': 'RandomUniform',
     'config': {'minval': -0.05, 'maxval': 0.05, 'seed': None},
     'registered_name': None},
    'embeddings_regularizer': None,
    'activity_regularizer': None,
    'embeddings_constraint': None,
    'mask_zero': False,
    'input_length': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, None)},
   'name': 'embedding',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Dropout',
   'config': {'name': 'dropout',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.1,
    'noise_shape': None,
    'seed': None},
   'registered_name': None,
   'build_config': {'input_shape': (None, None, 50)},
   'name': 'dropout',
   'inbound_nodes': [[['embedding', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'Bidirectional',
   'config': {'name': 'bidirectional',
    'trainable': True,
    'dtype': 'float32',
    'layer': {'module': 'keras.layers',
     'class_name': 'LSTM',
     'config': {'name': 'lstm',
      'trainable': True,
      'dtype': 'float32',
      'return_sequences': True,
      'return_state': False,
      'go_backwards': False,
      'stateful': False,
      'unroll': False,
      'time_major': False,
      'units': 100,
      'activation': 'tanh',
      'recurrent_activation': 'sigmoid',
      'use_bias': True,
      'kernel_initializer': {'module': 'keras.initializers',
       'class_name': 'GlorotUniform',
       'config': {'seed': None},
       'registered_name': None},
      'recurrent_initializer': {'module': 'keras.initializers',
       'class_name': 'Orthogonal',
       'config': {'gain': 1.0, 'seed': None},
       'registered_name': None},
      'bias_initializer': {'module': 'keras.initializers',
       'class_name': 'Zeros',
       'config': {},
       'registered_name': None},
      'unit_forget_bias': True,
      'kernel_regularizer': None,
      'recurrent_regularizer': None,
      'bias_regularizer': None,
      'activity_regularizer': None,
      'kernel_constraint': None,
      'recurrent_constraint': None,
      'bias_constraint': None,
      'dropout': 0.0,
      'recurrent_dropout': 0.1,
      'implementation': 1},
     'registered_name': None},
    'merge_mode': 'concat'},
   'registered_name': None,
   'build_config': {'input_shape': (None, None, 50)},
   'name': 'bidirectional',
   'inbound_nodes': [[['dropout', 0, 0, {}]]]},
  {'module': 'keras.layers',
   'class_name': 'TimeDistributed',
   'config': {'name': 'time_distributed',
    'trainable': True,
    'dtype': 'float32',
    'layer': {'module': 'keras.layers',
     'class_name': 'Dense',
     'config': {'name': 'dense',
      'trainable': True,
      'dtype': 'float32',
      'units': 14,
      'activation': 'softmax',
      'use_bias': True,
      'kernel_initializer': {'module': 'keras.initializers',
       'class_name': 'GlorotUniform',
       'config': {'seed': None},
       'registered_name': None},
      'bias_initializer': {'module': 'keras.initializers',
       'class_name': 'Zeros',
       'config': {},
       'registered_name': None},
      'kernel_regularizer': None,
      'bias_regularizer': None,
      'activity_regularizer': None,
      'kernel_constraint': None,
      'bias_constraint': None},
     'registered_name': None}},
   'registered_name': None,
   'build_config': {'input_shape': (None, None, 200)},
   'name': 'time_distributed',
   'inbound_nodes': [[['bidirectional', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['time_distributed', 0, 0]]}
model = tensorflow.keras.models.Model.from_config(config)
# Load the model weights
model.load_weights("your_model_file.h5")

@app.route('/', methods=['GET', 'POST'])
def home():
    tagged_sentence = None
    sentence = ""
    if request.method == 'POST':
        user_input = request.form['sentence']
        sentence = user_input
        tokenized_sentence = nltk.word_tokenize(user_input)
        X_Samp = [[word2id.get(word, len(words) - 1) for word in tokenized_sentence]]  # No padding
        p = model.predict(np.array(X_Samp))  # Predict on it
        p = np.argmax(p, axis=-1)  # Map softmax back to a POS index
        tagged_sentence = [(word, tags[pred]) for word, pred in zip(tokenized_sentence, p[0])]
    
    return render_template('index.html', sentence=sentence, tagged_sentence=tagged_sentence)
