"""
SecShield
=========

Reproducibility implementation for:

SecShield: A Privacy-Preserved Federated Learning Model
to Detect Zero-Day Malware Attack

ICCCN 2026

This script consolidates recovered SecShield experimental code
for federated autoencoder-based IoT malware detection.

Expected files:
    benignData.npy
    attackData.npy

Each file is expected to contain 300 power-trace features
followed by a label column.
"""

# ============================================================
# IMPORTS
# ============================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_federated as tff

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

BENIGN_FILE = "benignData.npy"
ATTACK_FILE = "attackData.npy"

CLIENTS = 10
SAMPLES_PER_CLIENT = 1000
ROUNDS = 25

BATCH_SIZE = 16
CLIENT_LR = 0.001
SERVER_LR = 1.0

# Main experiment
RUN_MAIN = True

# C=10, D=500/1000/2000, rounds=5...30
RUN_SAMPLE_EXPERIMENT = False


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SECSHIELD")
print("Federated IoT Malware Detection")
print("=" * 70)

benign_raw = np.load(BENIGN_FILE)
attack_raw = np.load(ATTACK_FILE)

print("\nRaw benign shape:", benign_raw.shape)
print("Raw attack shape:", attack_raw.shape)


# Last column is label
benign_data = benign_raw[:, :-1].astype(np.float32)
attack_data = attack_raw[:, :-1].astype(np.float32)

input_dim = benign_data.shape[1]

print("Input dimension:", input_dim)


# ============================================================
# NORMALIZATION
# ============================================================

# Use benign statistics so malware information does not
# influence preprocessing.

min_val = np.min(benign_data)
max_val = np.max(benign_data)

benign_data = (
    (benign_data - min_val) /
    (max_val - min_val + 1e-8)
)

attack_data = (
    (attack_data - min_val) /
    (max_val - min_val + 1e-8)
)


# ============================================================
# DATA SPLIT
# ============================================================

# Recovered SecShield code used 20,000 benign training samples
# and 1,000 benign test samples where sufficient data exists.

if len(benign_data) >= 21000:

    benign_train, benign_remaining = train_test_split(
        benign_data,
        train_size=20000,
        random_state=42
    )

    # Use 1000 held-out benign samples
    benign_test = benign_remaining[:1000]

else:

    # Reproducible fallback for different dataset versions
    benign_train, benign_test = train_test_split(
        benign_data,
        test_size=0.20,
        random_state=42
    )


# Separate validation benign samples from training samples.
# These are used only to establish the anomaly threshold.

validation_size = min(
    4000,
    len(benign_train) // 5
)

benign_validation = benign_train[
    :validation_size
]

federated_train = benign_train[
    validation_size:
]


# Split malware so threshold/evaluation do not use
# the same malware samples.

attack_validation, attack_test = train_test_split(
    attack_data,
    test_size=0.50,
    random_state=42
)


print("\nDataset Split")
print("-" * 40)

print(
    "Federated benign train:",
    federated_train.shape
)

print(
    "Benign validation:",
    benign_validation.shape
)

print(
    "Benign test:",
    benign_test.shape
)

print(
    "Malware validation:",
    attack_validation.shape
)

print(
    "Malware test:",
    attack_test.shape
)


# ============================================================
# AUTOENCODER
# ============================================================

def create_autoencoder():

    """
    Architecture recovered from the original SecShield
    autoencoder experiment:

        300
         |
        64
         |
        32
         |
        16
         |
        32
         |
        64
         |
        300

    """

    return tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(input_dim,)
        ),

        # Encoder
        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        # Decoder
        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            input_dim,
            activation="sigmoid"
        )
    ])


# ============================================================
# TENSORFLOW FEDERATED MODEL
# ============================================================

def model_fn():

    keras_model = create_autoencoder()

    return tff.learning.models.from_keras_model(

        keras_model,

        input_spec=(

            tf.TensorSpec(
                [None, input_dim],
                tf.float32
            ),

            tf.TensorSpec(
                [None, input_dim],
                tf.float32
            )
        ),

        # Recovered SecShield implementation uses MAE.
        loss=tf.keras.losses.MeanAbsoluteError()
    )


# ============================================================
# FEDERATED DATA
# ============================================================

def make_federated_data(
    data,
    client_count,
    samples_per_client
):

    total = (
        client_count *
        samples_per_client
    )

    if total > len(data):

        raise ValueError(
            f"Requested {total} samples, "
            f"but only {len(data)} available."
        )

    indices = np.random.choice(
        len(data),
        total,
        replace=False
    )

    sampled = data[indices]

    clients = np.array_split(
        sampled,
        client_count
    )

    federated_data = []

    for client_data in clients:

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (client_data, client_data)
            )
            .batch(BATCH_SIZE)
        )

        federated_data.append(
            dataset
        )

    return federated_data


# ============================================================
# FEDERATED TRAINING
# ============================================================

def train_federated(
    client_count,
    samples_per_client,
    rounds
):

    print("\n" + "=" * 70)

    print(
        f"Training: "
        f"Clients={client_count}, "
        f"Samples/client={samples_per_client}, "
        f"Rounds={rounds}"
    )

    print("=" * 70)

    fed_data = make_federated_data(
        federated_train,
        client_count,
        samples_per_client
    )

    trainer = (
        tff.learning.algorithms
        .build_weighted_fed_avg(

            model_fn=model_fn,

            client_optimizer_fn=lambda:
                tf.keras.optimizers.Adam(
                    CLIENT_LR
                ),

            server_optimizer_fn=lambda:
                tf.keras.optimizers.SGD(
                    SERVER_LR
                )
        )
    )

    state = trainer.initialize()

    for round_number in range(
        1,
        rounds + 1
    ):

        result = trainer.next(
            state,
            fed_data
        )

        # Compatibility with different TFF releases
        if hasattr(result, "state"):
            state = result.state
        else:
            state = result[0]

        print(
            f"Round "
            f"{round_number:02d}/{rounds}"
        )

    # Transfer federated global weights
    # to normal Keras model.

    model = create_autoencoder()

    trainer.get_model_weights(
        state
    ).assign_weights_to(
        model
    )

    return model


# ============================================================
# RECONSTRUCTION LOSS
# ============================================================

def get_reconstruction_loss(
    model,
    data
):

    reconstruction = model.predict(
        data,
        verbose=0
    )

    loss = tf.keras.losses.mae(
        data,
        reconstruction
    ).numpy()

    return loss


# ============================================================
# THRESHOLD
# ============================================================

def calculate_threshold(
    model
):

    """
    Threshold approach recovered from the original
    SecShield autoencoder code:

        threshold =
            mean(reconstruction loss)
            +
            std(reconstruction loss)
    """

    validation_loss = (
        get_reconstruction_loss(
            model,
            benign_validation
        )
    )

    threshold = (
        np.mean(validation_loss) +
        np.std(validation_loss)
    )

    print("\nThreshold:", threshold)

    return threshold


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    threshold,
    make_plots=True
):

    # Match sample counts where possible.
    n = min(
        4000,
        len(benign_test),
        len(attack_test)
    )

    benign_eval = benign_test[:n]
    malware_eval = attack_test[:n]

    benign_loss = (
        get_reconstruction_loss(
            model,
            benign_eval
        )
    )

    malware_loss = (
        get_reconstruction_loss(
            model,
            malware_eval
        )
    )


    # --------------------------------------------------------
    # Labels
    #
    # 0 = Benign
    # 1 = Malware
    # --------------------------------------------------------

    losses = np.concatenate([
        benign_loss,
        malware_loss
    ])

    labels = np.concatenate([

        np.zeros(
            len(benign_loss),
            dtype=int
        ),

        np.ones(
            len(malware_loss),
            dtype=int
        )
    ])

    predictions = (
        losses > threshold
    ).astype(int)


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions
    )


    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall   : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)


    # ========================================================
    # FIGURES
    # ========================================================

    if make_plots:

        # ----------------------------------------------------
        # Figure 3 style:
        # Benign reconstruction error
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        plt.hist(
            benign_loss,
            bins=50,
            alpha=0.8
        )

        plt.axvline(
            threshold,
            linestyle="--",
            label=f"Threshold = {threshold:.4f}"
        )

        plt.xlabel(
            "Train/Test Loss"
        )

        plt.ylabel(
            "No. of Examples"
        )

        plt.title(
            "Benign Reconstruction Loss"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "figure3_benign_loss.png",
            dpi=300
        )

        plt.show()


        # ----------------------------------------------------
        # Figure 4 style:
        # Malware reconstruction error
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 5)
        )

        plt.hist(
            malware_loss,
            bins=50,
            alpha=0.8
        )

        plt.axvline(
            threshold,
            linestyle="--",
            label=f"Threshold = {threshold:.4f}"
        )

        plt.xlabel(
            "Test Loss"
        )

        plt.ylabel(
            "No. of Examples"
        )

        plt.title(
            "Malware Reconstruction Loss"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "figure4_malware_loss.png",
            dpi=300
        )

        plt.show()


        # ----------------------------------------------------
        # Figure 5 style:
        # Confusion matrix
        # ----------------------------------------------------

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Benign",
                "Malware"
            ]
        )

        disp.plot()

        plt.title(
            "Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            "figure5_confusion_matrix.png",
            dpi=300
        )

        plt.show()


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold
    }


# ============================================================
# MAIN SECSHIELD EXPERIMENT
# ============================================================

def run_main():

    model = train_federated(

        client_count=CLIENTS,

        samples_per_client=
            SAMPLES_PER_CLIENT,

        rounds=ROUNDS
    )

    threshold = (
        calculate_threshold(
            model
        )
    )

    results = evaluate(
        model,
        threshold,
        make_plots=True
    )

    print("\nPublished SecShield reference:")
    print("Accuracy : 95.16%")
    print("Precision: 95.63%")
    print("Recall   : 94.19%")

    return results


# ============================================================
# FIGURE 6 EXPERIMENT
# ============================================================

def run_sample_experiment():

    configurations = [

        (samples, rounds)

        for samples in [
            500,
            1000,
            2000
        ]

        for rounds in [
            5,
            10,
            15,
            20,
            25,
            30
        ]
    ]

    rows = []

    for samples, rounds in configurations:

        try:

            model = train_federated(

                client_count=10,

                samples_per_client=
                    samples,

                rounds=rounds
            )

            threshold = (
                calculate_threshold(
                    model
                )
            )

            metrics = evaluate(
                model,
                threshold,
                make_plots=False
            )

            rows.append({

                "Clients":
                    10,

                "Samples_per_Client":
                    samples,

                "Rounds":
                    rounds,

                "Accuracy":
                    metrics["accuracy"],

                "Precision":
                    metrics["precision"],

                "Recall":
                    metrics["recall"],

                "F1":
                    metrics["f1"],

                "Threshold":
                    threshold
            })


        except ValueError as error:

            print(
                "Skipping:",
                error
            )


    results = pd.DataFrame(
        rows
    )

    print("\n")
    print("=" * 70)
    print("FIGURE 6 EXPERIMENT RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    results.to_csv(
        "secshield_figure6_results.csv",
        index=False
    )

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    if RUN_MAIN:
        run_main()

    if RUN_SAMPLE_EXPERIMENT:
        run_sample_experiment()

    print("\nSecShield experiment complete.")