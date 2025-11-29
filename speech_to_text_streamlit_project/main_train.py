import os
import numpy as np
from utils.audio_processing import load_dataset
from models.rnn_ctc_model import build_rnn_ctc_model, text_to_int_sequence, VOCAB_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def prepare_ctc_batches(X, y, batch_size=2):
    """Prepare small CTC batches from (features, transcripts)."""
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_feats = [X[i] for i in batch_idx]
        batch_texts = [y[i] for i in batch_idx]

        # Feature padding
        max_timesteps = max(feat.shape[0] for feat in batch_feats)
        n_mfcc = batch_feats[0].shape[1]
        feat_padded = np.zeros((len(batch_feats), max_timesteps, n_mfcc), dtype="float32")
        feat_lengths = np.zeros((len(batch_feats), 1), dtype="int32")
        for i, feat in enumerate(batch_feats):
            t = feat.shape[0]
            feat_padded[i, :t, :] = feat
            feat_lengths[i, 0] = t

        # Label sequences
        label_seqs = [text_to_int_sequence(t) for t in batch_texts]
        max_label_len = max(len(seq) for seq in label_seqs)
        labels_padded = np.ones((len(batch_feats), max_label_len), dtype="int32") * (VOCAB_SIZE - 1)
        label_lengths = np.zeros((len(batch_feats), 1), dtype="int32")
        for i, seq in enumerate(label_seqs):
            l = len(seq)
            labels_padded[i, :l] = seq
            label_lengths[i, 0] = l

        dummy_y = np.zeros((len(batch_feats), 1), dtype="float32")
        yield (
            {
                "input_features": feat_padded,
                "input_labels": labels_padded,
                "input_feature_lengths": feat_lengths,
                "input_label_lengths": label_lengths,
            },
            {"ctc_loss": dummy_y},
        )

def main():
    print("Loading dataset...")
    X, y, paths = load_dataset(DATA_DIR)
    print(f"Loaded {len(X)} samples.")
    if len(X) == 0:
        print("No data found. Check data/ folder.")
        return

    print("Building model...")
    model, inference_model = build_rnn_ctc_model()
    model.summary()

    epochs = 3
    steps_per_epoch = max(1, len(X) // 2)

    print("Starting demo training...")
    batch_gen = prepare_ctc_batches(X, y, batch_size=2)
    model.fit(batch_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, "rnn_ctc_train_model.h5"))
    inference_model.save(os.path.join(models_dir, "rnn_ctc_inference_model.h5"))
    print("Models saved in models/ directory.")

if __name__ == "__main__":
    main()
