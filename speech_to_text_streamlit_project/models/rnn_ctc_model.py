import tensorflow as tf
from tensorflow.keras import layers, Model

# Charset: lowercase letters + space + apostrophe
CHARSET = "abcdefghijklmnopqrstuvwxyz '"
VOCAB_SIZE = len(CHARSET) + 1  # +1 for CTC blank

def text_to_int_sequence(text):
    return [CHARSET.index(c) for c in text if c in CHARSET]

def int_sequence_to_text(seq):
    return "".join(CHARSET[i] for i in seq if i < len(CHARSET))

def build_rnn_ctc_model(n_mfcc=13, rnn_units=128):
    """Build a small BiLSTM + CTC model for demo purposes."""
    input_features = layers.Input(shape=(None, n_mfcc), name="input_features")
    input_labels = layers.Input(shape=(None,), dtype="int32", name="input_labels")
    input_feature_lengths = layers.Input(shape=(1,), dtype="int32", name="input_feature_lengths")
    input_label_lengths = layers.Input(shape=(1,), dtype="int32", name="input_label_lengths")

    x = layers.Masking(mask_value=0.0)(input_features)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Dense(VOCAB_SIZE, activation="linear")(x)  # logits

    def ctc_loss_lambda(args):
        y_pred, labels, input_lengths, label_lengths = args
        # (batch, time, classes) -> (time, batch, classes)
        y_pred = tf.transpose(y_pred, [1, 0, 2])
        return tf.nn.ctc_loss(
            labels=labels,
            logits=y_pred,
            label_length=tf.cast(tf.squeeze(label_lengths, axis=-1), tf.int32),
            logit_length=tf.cast(tf.squeeze(input_lengths, axis=-1), tf.int32),
            logits_time_major=True,
            blank_index=VOCAB_SIZE - 1
        )

    loss_out = layers.Lambda(ctc_loss_lambda, name="ctc_loss")(
        [x, input_labels, input_feature_lengths, input_label_lengths]
    )

    model = Model(
        inputs=[input_features, input_labels, input_feature_lengths, input_label_lengths],
        outputs=loss_out
    )

    # Custom training: loss is output of Lambda, so pass through
    model.compile(optimizer="adam", loss={"ctc_loss": lambda y_true, y_pred: y_pred})

    # Inference model: only features -> logits
    inference_model = Model(inputs=input_features, outputs=x)
    return model, inference_model
