from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from scipy.fft import fft
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def feature_extraction(data):
    features = []
    for window in data:
        mean = np.mean(window)
        std = np.std(window)
        min_val = np.min(window)
        max_val = np.max(window)
        amplitude = np.abs(fft(window))
        amplitude = amplitude[:len(amplitude)//2]
        peak = np.argmax(amplitude)
        energy = np.sum(amplitude**2)

        feature_vector = [mean, std, min_val, max_val, peak, energy]

        top_k_mags = amplitude[1:4]
        feature_vector.extend(top_k_mags.tolist())

        features.append(feature_vector)
        
    return np.array(features)

def sliding_window(window_size, data):
   
    result = []
    for i in range(data.shape[0]):
        sample = data.iloc[i, :].values
        curr = 0
        while True:
            if curr + window_size >= data.shape[1]:
                break
            
            window = sample[curr:curr + window_size]
            result.append(window)
            curr += window_size

    return np.array(result)

def clustering_workflow():
    data = pd.read_csv('./mitbih_train.csv', header=None)
    data_subset = data.iloc[:500, :-1]
   
    windowed = sliding_window(15, data_subset)
    extracted = feature_extraction(windowed)

    scaler = StandardScaler()
    extracted = scaler.fit_transform(extracted) #first 1000 windows

    fig,ax = plt.subplots(4,5, figsize=(10, 6))
    for i in range(4):
        for j in range(5):
            ax[i,j].scatter(extracted[:, i], extracted[:, j], marker='o')
            ax[i,j].set_xlabel(f"Feature {i+1}")
            ax[i,j].set_ylabel(f"Feature {j+1}")
            ax[i,j].set_title(f"Feature {i+1} vs Feature {j+1}")
    plt.tight_layout()
    plt.show()

    # DBSCAN Clustering 

    FEATURE_1 = 0
    FEATURE_2 = 1

    eps_,min_samples_ = search_optimal(extracted, FEATURE_1, FEATURE_2)

    dbscan = DBSCAN(eps=eps_, min_samples=min_samples_)
    clusters = dbscan.fit_predict(extracted)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(extracted[:, 0], extracted[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    print(f"Noises: {list(clusters).count(-1)}")

    mapping = map_to_timepoints(clusters, data_subset)

    SAMPLE_IDX = 7
    
    plt.plot(data_subset.iloc[SAMPLE_IDX, :], label="ECG Sample")

    m = mapping[mapping["idx"] == SAMPLE_IDX]
    for cluster in m["cluster"].unique():
        m_cluster = m[m["cluster"] == cluster]
        plt.scatter(
            m_cluster["center"],
            data_subset.iloc[SAMPLE_IDX, m_cluster["center"]],
            label=f"Cluster {cluster}"
        )
    plt.title(f"ECG Sample {SAMPLE_IDX} with Clusters")
    plt.legend()
    plt.show()

def map_to_timepoints(clusters, raw_data, window_size=15):
    res = {"idx": [], "center": [], "cluster": []}
    cluster_index = 0

    for row_idx in range(raw_data.shape[0]):
        for start in range(0, raw_data.shape[1] - window_size, window_size):
            center = start + window_size // 2
            if cluster_index >= len(clusters):
                break  # avoid index out of range

            res["idx"].append(row_idx)
            res["center"].append(center)
            res["cluster"].append(clusters[cluster_index])
            cluster_index += 1

    df = pd.DataFrame(res)
    print(df)
    return df


def search_optimal(data, feature1, feature2):
    feature1 = data[:, 0]
    feature2 = data[:, 1]

    features = np.array([feature1, feature2]).T

    # Find Optimal cluster numbers 
    N=10
    for n in range(3,N+1):
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(features)

        scores = silhouette_score(features, labels)
        print(f"Silhouette score for {n} clusters: {scores:.3f}")

    # Plot K-distance graph - to pack as many densely packed points as possible

    OPTIMAL = 3
    nn = NearestNeighbors(n_neighbors=OPTIMAL)
    nn.fit(features)
    distances, _ = nn.kneighbors(features)
    distances = np.sort(distances[:, OPTIMAL-1], axis=0) # Try to do this from scratch
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('# of Points')
    plt.ylabel(f'{OPTIMAL}-th Nearest Neighbor Distance')
    plt.title('K-distance Graph')
    plt.show()

    # Find best eps
    # ASSUME that min_samples is 3 as well. This may change later.
    scores = []
    for eps in np.arange(0.05, 0.4, 0.05):
        dbscan = DBSCAN(eps=eps, min_samples=OPTIMAL)
        dbscan.fit(features)
        labels = dbscan.fit_predict(features)
        silhouette = silhouette_score(features, labels)
        scores.append({'eps': eps, 'silhouette': silhouette})

    best_eps = pd.DataFrame(scores, columns=['eps', 'silhouette']).sort_values(by='silhouette', ascending=False).iloc[0]['eps']

    # Find best min_samples
    min_samples = []
    for m in range(3,10):
        dbscan = DBSCAN(eps=best_eps, min_samples=m)
        labels = dbscan.fit_predict(features)
        silhouette = silhouette_score(features, labels)
        min_samples.append({'min_samples': m, 'silhouette': silhouette, 'cluster_num': len(set(labels))})
    
    best_min_samples = pd.DataFrame(min_samples, columns=['min_samples', 'silhouette', 'cluster_num']).sort_values(by='silhouette', ascending=False).iloc[0]['min_samples']
    
    return best_eps, int(best_min_samples)

clustering_workflow()