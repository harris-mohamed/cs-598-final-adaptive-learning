# Standard libraries
import os

# Third party libraries
# import keras.src
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
from fastdtw import fastdtw
from scipy.spatial.distance import cityblock
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from torch.utils.data import DataLoader, TensorDataset
try:
    import tensorflow_addons as tfa
except ImportError:
    tfa = None
    print("Warning: tensorflow_addons is not installed. Some models may not work.")

def load_dsa(data_root):
    segments, labels, metadata = [], [], []
    for activity in sorted(os.listdir(data_root)):
        for person in sorted(os.listdir(os.path.join(data_root, activity))):
            for session_file in sorted(os.listdir(os.path.join(data_root, activity, person))):
                if session_file.endswith('.txt'):
                    path = os.path.join(data_root, activity, person, session_file)
                    raw = np.loadtxt(path, delimiter=",")
                    segments.append(raw)
                    labels.append(activity)
                    metadata.append((activity, person, session_file))
    return segments, labels, metadata

def normalize_segments(segments):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return [scaler.fit_transform(seg) for seg in segments]

def window_segments(segments, labels, metadata, window=125, step=125):
    X, y, meta = [], [], []
    for seg, label, m in zip(segments, labels, metadata):
        for i in range(0, seg.shape[0] - window + 1, step):
            win = seg[i:i+window].T 
            X.append(win)
            y.append(label)
            meta.append(m)
    return np.stack(X), np.array(y), meta

def split_domains(X):
    return {
        'torso': X[:, 0:9, :],
        'right_arm': X[:, 9:18, :],
        'left_arm': X[:, 18:27, :],
        'right_leg': X[:, 27:36, :],
        'left_leg': X[:, 36:45, :]
    }

def compute_difference_vectors(X_s, X_t, metric='euclidean'):
    N, K, T = X_s.shape
    diffs = []
    for n in range(N):
        diff = []
        for k in range(K):
            a, b = X_s[n, k], X_t[n, k]
            if metric == 'euclidean': d = np.linalg.norm(a - b)
            elif metric == 'manhattan': d = cityblock(a, b)
            elif metric == 'dtw': d, _ = fastdtw(a, b)
            else: raise ValueError(f"Unknown metric: {metric}")
            diff.append(d)
        diffs.append(diff)
    return np.array(diffs)

def estimate_ipd(X_s, X_t, metric='euclidean', bandwidth=1.0, m=100):
    diffs = compute_difference_vectors(X_s, X_t, metric)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(diffs)
    samples = kde.sample(m)
    return np.mean(np.linalg.norm(samples, axis=1))

def compute_ipd_matrix(domain_data, metric='euclidean'):
    domains = list(domain_data.keys())
    ipd_matrix = np.zeros((len(domains), len(domains)))
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains):
            if i == j: continue
            ipd_matrix[i, j] = estimate_ipd(domain_data[d1], domain_data[d2], metric)
    return domains, ipd_matrix

def plot_ipd_matrix(names, matrix, metric):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=names, yticklabels=names, annot=True, cmap='viridis')
    plt.title(f"IPD Matrix Between Domains for metric {metric}")
    plt.tight_layout()
    plt.savefig(f'../data/{metric}_ipd.png')

# Vanilla LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_classes=19):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return self.softmax(logits)

def get_loader(X, y, batch_size=64):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32),
                                    torch.tensor(y, dtype=torch.long)),
                      batch_size=batch_size, shuffle=True)
    
def finetune(model, X, y, lr=1e-3, epochs=5, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = get_loader(X, y)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

        print(f"Epoch: {epoch} Training Loss: {train_loss / len(loader)}")
    return model

def evaluate(model, X, y, device='cpu'):
    model.eval()
    model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    with torch.no_grad():
        preds = torch.argmax(model(X_tensor), dim=1)
        acc = (preds == y_tensor).float().mean().item()
    return acc

# Fully convolutional network
def build_fcn(input_shape, nb_classes):
    x = keras.Input(shape=input_shape)
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(128, kernel_size=8, padding='same')(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_x = keras.layers.Dropout(0.2)(conv_x)
    full = keras.layers.GlobalAveragePooling1D()(conv_x)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    return keras.Model(inputs=x, outputs=out)

# ResNet
def build_resnet(input_shape, n_feature_maps, nb_classes):
    x = keras.Input(shape=input_shape)
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(n_feature_maps, kernel_size=8, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    shortcut_y = keras.layers.Conv1D(n_feature_maps, kernel_size=1, padding='same')(x)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=8, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=1, padding='same')(x1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=8, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=1, padding='same')(x1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    full = keras.layers.GlobalAveragePooling1D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    return keras.Model(inputs=x, outputs=out)

# Encoder
def build_encoder(input_shape, nb_classes):
    x = keras.Input(shape=input_shape)
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(x)
    conv1 = (tfa.layers.InstanceNormalization()(conv1) if tfa else keras.layers.BatchNormalization()(conv1))
    conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same')(conv1)
    conv2 = (tfa.layers.InstanceNormalization()(conv2) if tfa else keras.layers.BatchNormalization()(conv2))
    conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
    conv2 = keras.layers.Dropout(rate=0.2)(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)

    conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
    conv3 = (tfa.layers.InstanceNormalization()(conv3) if tfa else keras.layers.BatchNormalization()(conv3))
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.Dropout(rate=0.2)(conv3)

    attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

    dense_layer = keras.layers.Dense(units=256, activation='sigmoid')(multiply_layer)
    dense_layer = (tfa.layers.InstanceNormalization()(dense_layer) if tfa else keras.layers.BatchNormalization()(dense_layer))
    flatten_layer = keras.layers.Flatten()(dense_layer)
    out = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

    return keras.Model(inputs=x, outputs=out)

# Pretraining w/IPD
def adaptive_pretrain(model_fn, domain_data, y_encoded, ipd_matrix, domain_names, target, nb_classes, input_shape):
    idx_target = domain_names.index(target)
    g = {src: ipd_matrix[domain_names.index(src), idx_target] for src in domain_names if src != target}
    total = sum(g.values())
    alpha = {d: g[d] / total for d in g}
    sorted_sources = sorted(g, key=g.get)
    model = model_fn()
    for src in sorted_sources:
        X_src = domain_data[src]
        X_src = X_src if len(X_src.shape) == 3 else X_src.reshape(X_src.shape[0], input_shape[0], input_shape[1])
        X_train, X_val, y_train, Y_val = train_test_split(X_src, y_encoded, test_size=0.2, random_state=42)
        lr = 0.001 * (1 - alpha[src])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    return model

def evaluate_models(models_dict, distance_metrics, domain_data, domain_names, y_encoded, target):
    results = []
    for dist in distance_metrics:
        print(f"Computing IPD matrix using: {dist}")
        names, ipd_matrix = compute_ipd_matrix(domain_data, metric=dist)
        plot_ipd_matrix(names, ipd_matrix, dist)

        for model_name, model_fn in models_dict.items():
            print(f"→ Training model: {model_name} with distance: {dist}")
            input_shape = domain_data[target].shape[1:]  # (T, C) format
            model = adaptive_pretrain(lambda: model_fn(input_shape, len(np.unique(y_encoded))),
                                       domain_data, y_encoded, ipd_matrix,
                                       domain_names, target, len(np.unique(y_encoded)), input_shape)
            X_target = domain_data[target]
            X_target = X_target if len(X_target.shape) == 3 else X_target.reshape(X_target.shape[0], input_shape[0], input_shape[1])
            X_train, X_val, y_train, y_val = train_test_split(X_target, y_encoded, test_size=0.2, random_state=42)
            loss, acc = model.evaluate(X_val, y_val, verbose=0)
            results.append({'Model': model_name, 'Distance': dist, 'Accuracy': acc})
    df = pd.DataFrame(results)
    print("\nEvaluation Summary:")
    summary = df.pivot(index='Distance', columns='Model', values='Accuracy').round(4)
    print(summary)

    # RCC computation
    print("\nRelative Closeness Coefficient (RCC) Ranking:")
    ideal_best = summary.max()
    ideal_worst = summary.min()
    distances_to_best = (summary - ideal_best) ** 2
    distances_to_worst = (summary - ideal_worst) ** 2
    D_plus = np.sqrt(distances_to_best.sum(axis=0))
    D_minus = np.sqrt(distances_to_worst.sum(axis=0))
    rcc_scores = D_minus / (D_plus + D_minus)
    rcc_df = rcc_scores.sort_values(ascending=False).to_frame(name='RCC')
    print(rcc_df.round(4))

    return df, rcc_df

raw, labels, meta = load_dsa('../data')
norm = normalize_segments(raw)
X_seg, y_seg, _ = window_segments(norm, labels, meta)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_seg)
domain_data = split_domains(X_seg)
domain_names = list(domain_data.keys())
target = 'right_arm'

# Build dictionary of models
models_dict = {
    'FCN': lambda input_shape, nb_classes: build_fcn(input_shape, nb_classes),
    'ResNet': lambda input_shape, nb_classes: build_resnet(input_shape, 64, nb_classes),
    'Encoder': lambda input_shape, n: build_encoder(input_shape, n),
    'LSTM': lambda input_shape, n: keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, dropout=0.2),
        keras.layers.Dense(n, activation='softmax')
    ])
}

# Distance functions to evaluate
distance_metrics = ['euclidean', 'manhattan', 'dtw']

# Run evaluation
results_df = evaluate_models(models_dict, distance_metrics, domain_data, domain_names, y_encoded, target)
