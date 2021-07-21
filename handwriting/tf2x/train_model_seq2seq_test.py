# Header
from functions.read_data import decode_target_hex
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np
import time
import sys
import random
import pickle
import os
import logging
import glob
import datetime
import configparser
import argparse
import cv2
import pickle
import traceback

notebook_name = 'train_model_seq2seq_test'


# Logs
know_time = datetime.datetime.now()
log_name = notebook_name+'-'+str(know_time.year)+"_"+str(know_time.month).zfill(2)+"_"+str(know_time.day).zfill(2)\
    + "_"+str(know_time.hour).zfill(2)+"_" + \
    str(know_time.minute).zfill(2)+".log"
if not os.path.exists("logs"):
    os.makedirs("logs")


# logging
logger = logging.getLogger(notebook_name)
logging.basicConfig()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr = logging.FileHandler("logs/"+log_name)
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(20)
logger.setLevel(logging.DEBUG)


def parse_arguments() -> argparse.Namespace:
    """
    Command line arguments definition
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help="Config file", dest="config", required=True, default='config/remote.cfg')
    parser.add_argument('-e', '--experiment', type=str,
                        help="Experiment relative path to models dir", dest="experiment", required=True, default='test/default')
    parser.add_argument('-d', '--data_path', type=str,
                        help="Data path", dest="data_path", required=False, 
                        default='/home/ubuntu/data/tesis/handwriting/databases/IAM/normalized_original_size_sample')
                        
    parser.add_argument('-x_size', '--x_size', type=int,
                        help="X size", dest="x_size", required=False, default=512)
    parser.add_argument('-y_size', '--y_size', type=int,
                        help="Y size", dest="y_size", required=False, default=64)

    return parser.parse_args()





def read_data(data_path):
    """[summary]

    Args:
        data_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    filenames_list = glob.glob(os.path.join(data_path, '*'))

    images_list = []
    targets_list = []
    for filename in filenames_list:
        try:
            im = Image.open(filename).convert('L')  # read and convert to grayscale
            images_list += [np.array(im, dtype=np.float32)/255]

            # decode target from filename
            f = os.path.basename(filename)
            file_name = '.'.join(f.split('.')[:-1])
            hex_name = file_name.split('_')[-1]  # extract hexadecimal part
            targets_list += [decode_target_hex(hex_name)]
        except Exception as e:
            logger.info(f'error reading file {filename}')
            logger.info(f'Exception: {e}')
            logger.info(traceback.format_exc())
    logger.info(f'Readed {len(filenames_list)} images')

    return images_list, targets_list



def encode_targets(targets_list, target_len=None, encoder_dict=None):
    """[summary]

    Args:
        targets_list ([type]): [description]
        target_len ([type], optional): [description]. Defaults to None.
        encoder_dict ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if encoder_dict == None:
        # get all characters
        char_list = sorted(set(''.join(targets_list)))

        encoder_dict = {}
        decoder_dict = {0: '<PAD>', 1: '<START>', 2: '<END>'}
        # code 0 reserved for padding
        for i, c in enumerate(char_list):
            encoder_dict[c] = i+3
            decoder_dict[i+3] = c
    else:
        decoder_dict = None

    # if target_len is None --> Train.  use the max len in data
    if target_len == None:
        target_len = max([len(w) for w in targets_list])

    targets_encoded_list = []
    for w in targets_list:
        w = w[:target_len]
        # Add 1 -> <START> at beggining and 2 -> <END> at the end
        w_encoded = [1] + [encoder_dict[c]
                           for c in w if c in encoder_dict.keys()] + [2]

        # padding whit 0 until target_len
        w_encoded = w_encoded + [0]*(max(0, target_len + 2 - len(w_encoded)))

        targets_encoded_list += [w_encoded]

    return np.array(targets_encoded_list, dtype=np.uint8), encoder_dict, decoder_dict




class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        """[summary]

        Args:
            enc_units ([type]): [description]
            batch_sz ([type]): [description]
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        # pending to add convolutions
        # self.convolution = tf.keras.layers.Convolution(params)
        self.c1 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same')
        self.c2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same')
        self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))

        self.c3 = tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', padding='same')
        self.c4 = tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', padding='same')
        self.mp2 = tf.keras.layers.MaxPooling2D((2, 2))

        self.c5 = tf.keras.layers.Conv2D(
            256, (3, 3), activation='relu', padding='same')
        self.c6 = tf.keras.layers.Conv2D(
            256, (3, 3), activation='relu', padding='same')

        self.dense  = tf.keras.layers.Dense(enc_units, activation='relu')

        # RNN
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=False,
                dropout=0.5,
                recurrent_initializer='glorot_uniform'
            )
        )

        self.gru2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                dropout=0.5,
                recurrent_initializer='glorot_uniform'
            )
        )


    def call(self, x, hidden, training):
        """[summary]

        Args:
            x ([type]): [description]
            hidden ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Convolutional layers
        # out x shape: num_features x size_feature
        x = tf.expand_dims(x, -1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.mp1(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.mp2(x)
        x = self.c5(x)
        x = self.c6(x)
        x = tf.keras.layers.Permute((2, 1, 3))(x)
        s = x.get_shape()
        x = tf.keras.layers.Reshape((s[1], -1))(x)

        x = self.dense(x)

        # RNN layers for encoder
        x = self.gru(x, training=training)
        output, state_f, state_b = self.gru2(x, training=training, initial_state=hidden)

        return output, tf.concat([state_f, state_b], axis=-1)

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))] # add *2 to use in bidirectional layer



class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()

        self.batch_sz = batch_sz
        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            dropout=0.5,
            recurrent_initializer='glorot_uniform'
        )

        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)


    def call(self, x, hidden, enc_output, training):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, training=training)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights





def decode_text(char_id_list, decoder_dict):
    """[summary]

    Args:
        char_id_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    word = ''
    for i in char_id_list:
        if i > 2:
            word += decoder_dict[i]
    return word



def evaluate(inputs, encoder, decoder, decoder_dict):
    """[summary]

    Args:
        inputs ([type]): [description]

    Returns:
        [type]: [description]
    """
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    size_rnn_encoder = encoder.gru.get_config()['units']
    #hidden = [tf.zeros((1, size_rnn_encoder)), tf.zeros((1, size_rnn_encoder))]
    hidden = [tf.zeros((1, size_rnn_encoder)), tf.zeros((1, size_rnn_encoder))]
    enc_out, enc_hidden = encoder(inputs, hidden, training=False)

    dec_hidden = enc_hidden
    
    dec_input = tf.expand_dims([0], 0)
    
    max_word_len = decoder.fc.get_config()['units']
    for t in range(max_word_len):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out, training=False)
        predicted_id = tf.argmax(predictions[0]).numpy()
        
        if predicted_id in decoder_dict.keys():
            if decoder_dict[predicted_id] == '<END>':
                return result
            if decoder_dict[predicted_id] != '<PAD>':
                result += decoder_dict[predicted_id]

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
                
    return result




# Normalize size of each word
def normalize_shape(img_list, x_size=192, y_size=48, plot=False):
    
    img_normalized_list = []

    for img in img_list:
        # ajuste de altura
        y, x = img.shape
        if y_size is not(None):
            img = cv2.resize(img, (max(2,int(x*(y_size/y))), y_size))

        # Recorte derecha e izquierda
        true_points = np.argwhere(img)
        if len(true_points)>0:
            # take the smallest points and use them as the top left of your crop
            top_left = true_points.min(axis=0)
            # take the largest points and use them as the bottom right of your crop
            bottom_right = true_points.max(axis=0)
            if bottom_right[1] - top_left[1] > 2:
                img = img[:, top_left[1]:bottom_right[1]+1]


        # Ajuste de anchura
        y, x = img.shape
        if x < x_size:
            img = np.concatenate([img, np.zeros([y, x_size - x])], axis=1)
        else:
            img = cv2.resize(img, (x_size, y_size))

        img_normalized_list += [img-0.5]
    
    if plot:
        fig = plt.figure()
        n = 1
        for img, img_norm in zip(img_list, img_normalized_list): 
            a = fig.add_subplot(len(img_list), 2, n)
            a.set_title('Original')
            fig.tight_layout()
            plt.imshow(255-img, cmap='gray')
            n += 1
            
            a = fig.add_subplot(len(img_list), 2, n)
            a.set_title('Normalized')
            fig.tight_layout()
            plt.imshow(255-img_norm, cmap='gray')
            n += 1
        
    return img_normalized_list





def main(arguments: argparse.Namespace) -> None:

    #print arguments
    logger.info(f'Arguments: {arguments}')

    # Load config
    config = configparser.ConfigParser()
    config.read(arguments.config)

    # Print config
    logger.info(f'Config content: {arguments.config}')
    for section in config.sections():
        logger.info(f'section: {dict(config[section])}')

    logger.info(f'Tensorflow: {tf.__version__}')


    # Create experiments dir
    experiments_path = os.path.join(config['LOCAL_PATHS']['experiments_path'], arguments.experiment)
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)




    # Read train data
    logger.info('Reading train data...')
    train_data_path = os.path.join(arguments.data_path, 'trn')
    images_train_array, targets_train_list = read_data(train_data_path)

    # Encode targets of train data
    targets_encoded_array, encoder_dict, decoder_dict = encode_targets(
        targets_train_list)
    logger.info('Reading train data... Done!')
    
    
    # Model parameters
    # Length of the target in train
    MAX_WORD_LEN = targets_encoded_array.shape[1]
    NUM_CHARACTERS_DICTIONARY = max(decoder_dict.keys()) + 1
    logger.info(f'MAX_WORD_LEN: {MAX_WORD_LEN}')
    logger.info(f'NUM_CHARACTERS_DICTIONARY: {NUM_CHARACTERS_DICTIONARY}')

    # Save parameters in the experiment folder
    # decoder_dict and MAX_WORD_LEN
    pickle.dump(decoder_dict, open(os.path.join(experiments_path, 'decoder_dict.pkl'), 'wb' ))
    pickle.dump(encoder_dict, open(os.path.join(experiments_path, 'encoder_dict.pkl'), 'wb' ))
    with open(os.path.join(experiments_path,'MAX_WORD_LEN.txt'), 'w') as f:
        f.write(str(MAX_WORD_LEN))



    # Create a tf.data dataset for train data
    def generate_images_trn():
        for img, t in zip(images_train_array, targets_encoded_array):
            #img_augmented_list = get_img_augmented([img], augment=True)
            #img_normalized = normalize_shape(img_augmented_list, x_size=192, y_size=48, plot=False)    
            img_normalized = normalize_shape([img], x_size=arguments.x_size, y_size=arguments.y_size, plot=False)    
            yield (img_normalized[0], t)
    

    BUFFER_SIZE = len(images_train_array)
    BATCH_SIZE = 32
    steps_per_epoch = len(images_train_array)//BATCH_SIZE
    dataset_trn = tf.data.Dataset.from_generator(
        generate_images_trn,
        (tf.float32, tf.int32)
    )
    dataset_for_train = dataset_trn.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)




    # Read validation data
    logger.info('Reading val data...')
    val_data_path = os.path.join(arguments.data_path, 'val')
    images_val_array, targets_val_list = read_data(val_data_path)
    logger.info('Reading val data... Done!')

    # to encode validation and test use:
    targets_val_encoded_array, _, _ = encode_targets(
        targets_val_list,
        target_len=MAX_WORD_LEN,
        encoder_dict=encoder_dict
    )

    def generate_images_val():
        for img, t in zip(images_val_array, targets_val_encoded_array):
            img_normalized = normalize_shape([img], x_size=arguments.x_size, y_size=arguments.y_size, plot=False)
            yield (img_normalized[0], t)

    dataset_val = tf.data.Dataset.from_generator(
        generate_images_val, (tf.float32, tf.int32)
    )

    # Sample to evaluation in each epoch
    img_eval_list = []
    target_eval_list = []
    n=0
    for img, t in dataset_val.shuffle(len(images_val_array)).as_numpy_iterator():
        img_eval_list += [img]
        target_eval_list += [t]
        n += 1
        if n>20:
            break

    # Prepare dataset for full score
    dataset_val = dataset_val.batch(BATCH_SIZE)
    steps_per_epoch_val = len(images_val_array)//BATCH_SIZE



    # Read test data
    logger.info('Reading test data...')
    tst_data_path = os.path.join(arguments.data_path, 'tst')
    images_tst_array, targets_tst_list = read_data(tst_data_path)
    logger.info('Reading test data... Done!')

    # to encode validation and test use:
    targets_tst_encoded_array, _, _ = encode_targets(
        targets_tst_list,
        target_len=MAX_WORD_LEN,
        encoder_dict=encoder_dict
    )

    def generate_images_tst():
        for img, t in zip(images_tst_array, targets_tst_encoded_array):
            img_normalized = normalize_shape([img], x_size=arguments.x_size, y_size=arguments.y_size, plot=False)
            yield (img_normalized[0], t)

    dataset_tst = tf.data.Dataset.from_generator(
        generate_images_tst, (tf.float32, tf.int32)
    ).batch(BATCH_SIZE)
    steps_per_epoch_tst = len(images_tst_array)//BATCH_SIZE




    # model parameters
    SIZE_RNN_ENCODER = 128
    embedding_dim = 64
    SIZE_RNN_DECODER = 128

    # Model
    encoder = Encoder(SIZE_RNN_ENCODER, BATCH_SIZE)
    decoder = Decoder(NUM_CHARACTERS_DICTIONARY, embedding_dim, SIZE_RNN_DECODER, BATCH_SIZE)


    # Check Model. Sample imput
    def check_model():
        example_input_batch, example_target_batch = next(iter(dataset_for_train))

        sample_hidden = encoder.initialize_hidden_state()
        sample_output, sample_hidden = encoder(example_input_batch, sample_hidden, training=False)
        logger.info('Encoder output shape: (batch size, sequence length, units) {}'.format(
            sample_output.shape))
        logger.info('Encoder Hidden state shape: (batch size, units) {}'.format(
            sample_hidden.shape))

        attention_layer = BahdanauAttention(10)
        attention_result, attention_weights = attention_layer(
            sample_hidden, sample_output)
        logger.info("Attention result shape: (batch size, units) {}".format(
            attention_result.shape))
        logger.info("Attention weights shape: (batch_size, sequence_length, 1) {}".format(
            attention_weights.shape))

        sample_decoder_output, _, _ = decoder(
            tf.ones((BATCH_SIZE, 1)),
            sample_hidden,
            sample_output,
            training=False
        )
        logger.info('Decoder output shape: (batch_size, vocab size) {}'.format(
            sample_decoder_output.shape))

    check_model()
    logger.info(encoder.summary())
    logger.info(decoder.summary())



    # Define the optimizer and the loss function
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=3000,
        decay_rate=0.95,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    epsilon_smoothing = 0.4
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing = epsilon_smoothing/NUM_CHARACTERS_DICTIONARY,
        reduction=tf.keras.losses.Reduction.NONE
    )
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(tf.one_hot(real, NUM_CHARACTERS_DICTIONARY), pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)


    # Define the optimizer and the loss function for evaluation. No label smoothing
    loss_object_eval = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    def loss_function_eval(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object_eval(tf.one_hot(real, NUM_CHARACTERS_DICTIONARY), pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)






    # Checkpoints (Object-based saving)
    checkpoint_dir = os.path.join(experiments_path, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder
    )



    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden, training=True)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(targ[:,0], 1)
            # Feeding the prediction of previous as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, training=True)
                loss += loss_function(targ[:, t], predictions)
                # using predictions
                dec_input = tf.expand_dims(tf.argmax(predictions, axis=1), 1)
                # using teacher forcing
                #dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        #print(variables,'\n')
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss



    @tf.function
    def score_step(inp, targ, enc_hidden):
        loss = 0
        predict_out = []

        enc_output, enc_hidden = encoder(inp, enc_hidden, training=False)
        dec_hidden = enc_hidden
        dec_input = np.ones((BATCH_SIZE, 1), dtype=np.float32) # 1 --> START
        # Feeding the prediction of previous as the next input
        for t in range(1, MAX_WORD_LEN):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, training=False)
            loss += loss_function_eval(targ[:, t], predictions)
            predict_out += [tf.argmax(predictions, axis=1)]
            # using predictions as input of the next position
            dec_input = tf.expand_dims(tf.argmax(predictions, axis=1), 1)
        batch_loss = (loss / int(targ.shape[1]))

        return tf.stack(predict_out, axis=1), batch_loss



    def evaluate_dataset(dataset, steps_per_epoch, print_sample=False):

        total_loss = 0
        real_list = []
        prediction_list = []

        enc_hidden = encoder.initialize_hidden_state()

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            predictions, batch_loss = score_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            for p, r in zip(predictions, targ):
                real_list += [decode_text(r.numpy(), decoder_dict)]
                prediction_list += [decode_text(p.numpy(), decoder_dict)]

        validation_loss = total_loss / steps_per_epoch
        validation_accuracy = np.mean([r==p for r, p in zip(real_list, prediction_list)])

        if print_sample:
            for p, r in zip(predictions, targ):
                word = decode_text(r.numpy(), decoder_dict)
                p_word = decode_text(p.numpy(), decoder_dict)
                logger.info(f'{word} -> {p_word}')


        return validation_loss, validation_accuracy
        




    # Recupera progreso anterior
    try:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        logger.info(f'Restore checkpoint from {checkpoint_dir}')
    except Exception as e:
        logger.info(f'Problems recovering last chekpoint from {checkpoint_dir}')
        logger.info(f'Exception: {e}')
        logger.info(traceback.format_exc())



    # Tensorboard path
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(checkpoint_dir, 'train')
    valid_log_dir = os.path.join(checkpoint_dir, 'valid')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    # Train
    EPOCHS = 100

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset_for_train.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                logger.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        
        
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        logger.info('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
        logger.info(f'Time taken for 1 epoch {time.time() - start} sec\n')
        logger.info(f"LR: {optimizer._decayed_lr('float32').numpy()}")

        # Evaluate train data each epoch
        train_loss, train_accuracy = evaluate_dataset(dataset_trn.batch(BATCH_SIZE), steps_per_epoch)
        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Train Accuracy: {train_accuracy:.4f}')
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('accuracy', train_accuracy, step=epoch)
            tf.summary.scalar('lr', optimizer._decayed_lr('float32').numpy(), step=epoch)


        # Evaluate validation data each epoch
        validation_loss, validation_accuracy = evaluate_dataset(dataset_val, steps_per_epoch_val, print_sample=True)
        logger.info(f'Validation Loss: {validation_loss:.4f}')
        logger.info(f'Validation Accuracy: {validation_accuracy:.4f}')
        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', validation_loss, step=epoch)
            tf.summary.scalar('accuracy', validation_accuracy, step=epoch)


        # Evaluate test data each epoch
        test_loss, test_accuracy = evaluate_dataset(dataset_tst, steps_per_epoch_tst, print_sample=True)
        logger.info(f'Test Loss: {test_loss:.4f}')
        logger.info(f'Test Accuracy: {test_accuracy:.4f}')





if __name__ == '__main__':
    """
    Process main initialization
    """

    args = parse_arguments()
    
    main(args)
