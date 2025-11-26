
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PARAMETERS (adjust if desired)
# -----------------------------
GRID_ROWS = 10
GRID_COLS = 10
INPUT_COLS = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
N_EPOCHS = 100        # number of passes over the dataset (increase for longer training)
INIT_LR = 0.5         # initial learning rate
FINAL_LR = 0.01       # final learning rate (decay)
INIT_SIGMA = max(GRID_ROWS, GRID_COLS) / 2.0
FINAL_SIGMA = 1.0

# -----------------------------
# 1) Load and preprocess data
# -----------------------------
df = pd.read_csv('Iris.csv')   # make sure Iris.csv sits next to this script
X = df[INPUT_COLS].values.astype(float)
y = df['Species'].values.astype(str)
scaler = StandardScaler()
Xstd = scaler.fit_transform(X)  # standardize features (important for SOM distances)

n_samples, n_features = Xstd.shape

# -----------------------------
# 2) Initialize SOM weights
# -----------------------------
rng = np.random.RandomState(0)
weights = rng.normal(loc=0.0, scale=0.1, size=(GRID_ROWS, GRID_COLS, n_features))
# grid coordinates for every neuron (used to compute neighborhood)
neuron_coords = np.array([[(i, j) for j in range(GRID_COLS)] for i in range(GRID_ROWS)])

# helper to find best-matching unit (BMU) for a vector x
def find_bmu(weights_array, x):
    diff = weights_array - x  # shape (R,C,F)
    dist_sq = np.sum(diff ** 2, axis=2)
    bmu_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    return bmu_idx  # tuple (i,j)

# -----------------------------
# 3) Training (online updates)
# -----------------------------
total_iterations = N_EPOCHS * n_samples
iter_count = 0
for epoch in range(N_EPOCHS):
    perm = rng.permutation(n_samples)
    for idx in perm:
        x = Xstd[idx]
        # normalized time for decay (0 -> 1)
        t = iter_count / float(total_iterations)
        # exponential decay of learning rate and sigma
        lr = INIT_LR * ((FINAL_LR / INIT_LR) ** t)
        sigma = INIT_SIGMA * ((FINAL_SIGMA / INIT_SIGMA) ** t)

        bmu_i, bmu_j = find_bmu(weights, x)

        # squared distance on neuron grid to BMU
        dist_sq = (neuron_coords[:, :, 0] - bmu_i)**2 + (neuron_coords[:, :, 1] - bmu_j)**2
        # Gaussian neighborhood function
        h = np.exp(-dist_sq / (2 * (sigma**2)))

        # update rule (vectorized): w <- w + lr * h * (x - w)
        weights += lr * h[:, :, np.newaxis] * (x - weights)

        iter_count += 1

# -----------------------------
# 4) U-Matrix (average neighbor distance)
# -----------------------------
u_matrix = np.zeros((GRID_ROWS, GRID_COLS))
for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        neighbor_dists = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID_ROWS and 0 <= nj < GRID_COLS:
                neighbor_dists.append(np.linalg.norm(weights[i,j] - weights[ni,nj]))
        u_matrix[i,j] = np.mean(neighbor_dists) if neighbor_dists else 0.0

# Save U-Matrix image
plt.figure()
plt.imshow(u_matrix, aspect='equal')
plt.colorbar(label='avg neighbor distance')
plt.title('SOM U-Matrix (average neighbor distance)')
plt.xlabel('Neuron column')
plt.ylabel('Neuron row')
plt.tight_layout()
plt.savefig('fig_som_umatrix.png', dpi=200)
plt.close()

# -----------------------------
# 5) Hit map & majority-class overlay
# -----------------------------
hits = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
neuron_labels = [[[] for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

for i in range(n_samples):
    x = Xstd[i]
    bmu_i, bmu_j = find_bmu(weights, x)
    hits[bmu_i, bmu_j] += 1
    neuron_labels[bmu_i][bmu_j].append(y[i])

# majority class per neuron (empty string if no hits)
majority = np.empty((GRID_ROWS, GRID_COLS), dtype=object)
for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        if len(neuron_labels[i][j]) == 0:
            majority[i,j] = ''
        else:
            vals, counts = np.unique(neuron_labels[i][j], return_counts=True)
            majority[i,j] = vals[np.argmax(counts)]

# create short labels for neat overlays (e.g., 's','v','v' or 's','v','g')
unique_species = np.unique(y)
short_label = {lab: lab.split('-')[-1][0] for lab in unique_species}

plt.figure()
plt.imshow(hits, aspect='equal')
plt.colorbar(label='BMU hit count')
plt.title('SOM Hit Map (BMU counts) with majority-class overlay')
plt.xlabel('Neuron column')
plt.ylabel('Neuron row')

# overlay a short text label at each neuron and a marker sized by hits
for i in range(GRID_ROWS):
    for j in range(GRID_COLS):
        if hits[i,j] > 0:
            plt.text(j, i, short_label.get(majority[i,j], ''), ha='center', va='center', fontsize=8, fontweight='bold')
            plt.scatter(j, i, s=20 + hits[i,j]*10, alpha=0.6)

plt.tight_layout()
plt.savefig('fig_som_hits.png', dpi=200)
plt.close()

# -----------------------------
# 6) Save a compact summary
# -----------------------------
with open('som_iris_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'SOM grid: {GRID_ROWS}x{GRID_COLS}\n')
    f.write(f'Training epochs: {N_EPOCHS}, total iterations: {total_iterations}\n')
    f.write(f'Final approx LR: {lr:.4f}, final approx sigma: {sigma:.4f}\n')
    f.write('Top neurons by hit count (row,col,count,majority):\n')
    flat = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            if hits[i,j] > 0:
                flat.append((i,j,hits[i,j], majority[i,j]))
    flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)
    for item in flat_sorted[:10]:
        f.write(f'{item[0]},{item[1]},{item[2]},{item[3]}\n')

print('Saved: fig_som_umatrix.png, fig_som_hits.png, som_iris_summary.txt')
