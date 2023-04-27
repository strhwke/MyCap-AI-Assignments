import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# define the model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, vocab_size), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# train the model
model.fit(X, y, batch_size=128, epochs=50)

# generate text
def generate_text(model, start_seed, num_chars):
    """
    Generates text based on a trained model and a starting seed.
    """
    # convert the starting seed to a one-hot encoded array
    seed = [char_to_idx[c] for c in start_seed]
    x = np.zeros((1, len(seed), vocab_size))
    for i, idx in enumerate(seed):
        x[0, i, idx] = 1

    # generate the text
    text = start_seed
    for i in range(num_chars):
        # make a prediction based on the current input
        preds = model.predict(x)[0]
        # sample the next character using the predicted probabilities
        next_idx = np.random.choice(len(preds), p=preds)
        # add the next character to the text
        next_char = idx_to_char[next_idx]
        text += next_char
        # update the input with the new character
        x = np.roll(x, -1, axis=1)
        x[0, -1, next_idx] = 1

    return text