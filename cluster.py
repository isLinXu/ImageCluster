import os
import shutil

import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from clip import clip
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import hashlib
from tqdm import tqdm


class ImageCluster:
    def __init__(self, image_folder, text, model_name="ViT-B/32", device=None):
        self.image_list = None
        self.image_folder = image_folder
        self.text = text
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.similarity_threshold = 0.1

    def _extract_features(self):
        image_features = []
        image_list = []
        image_hashes = set()
        # Tokenize and preprocess the text
        text_input = clip.tokenize([self.text]).to(self.device)

        # Encode the text
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)

        for image_name in tqdm(os.listdir(self.image_folder)):
            image_path = os.path.join(self.image_folder, image_name)
            try:
                image = Image.open(image_path)
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                # Compute the MD5 hash and check if it already exists
                image_hash = self._compute_md5(image_path)
                if image_hash in image_hashes:
                    continue
                image_hashes.add(image_hash)
                with torch.no_grad():
                    features = self.model.encode_image(image_input)
                    # Compute the similarity between the image and text features
                    similarity = torch.nn.functional.cosine_similarity(features, text_features)
                    # Expand the similarity tensor to 2D
                    similarity = similarity.unsqueeze(-1)
                    # Concatenate the features and similarity
                    features_with_similarity = torch.cat([features, similarity], dim=-1)

                # Filter out images with similarity below the threshold
                if similarity.item() > self.similarity_threshold:
                    image_features.append(features_with_similarity.cpu().numpy())
                    image_list.append(image_name)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

        return np.vstack(image_features), image_list

    def set_similarity_threshold(self, threshold):
        self.similarity_threshold = threshold

    def get_image_list(self):
        return os.listdir(self.image_folder)

    def get_image_count(self):
        return len(os.listdir(self.image_folder))
    def cluster_images(self, n_clusters=2, filter_similarity=True):
        image_features, image_list = self._extract_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_features)

        clusters = {i: [] for i in range(n_clusters)}
        cluster_features = {i: [] for i in range(n_clusters)}

        for image_name, label, features in zip(image_list, kmeans.labels_, image_features):
            clusters[label].append(image_name)
            cluster_features[label].append(features)

        if filter_similarity:
            # Compute the mean similarity for each cluster
            cluster_mean_similarities = {}
            for label, features in cluster_features.items():
                mean_similarity = self._compute_cluster_mean_similarity(features)
                cluster_mean_similarities[label] = mean_similarity

            # Filter out images with similarity below the threshold
            filtered_clusters = {}
            for label, images in clusters.items():
                mean_similarity = cluster_mean_similarities[label]
                filtered_images = [image_name for image_name, features in zip(images, cluster_features[label]) if
                                   features[-1] >= mean_similarity]
                filtered_clusters[label] = filtered_images

            clusters = filtered_clusters

        silhouette_avg = silhouette_score(image_features, kmeans.labels_)
        return clusters, silhouette_avg
    def _compute_cluster_mean_similarity(self, cluster_features):
        similarities = []
        for features in cluster_features:
            similarity = features[-1]
            similarities.append(similarity)
        return np.mean(similarities)

    def print_clusters(self, clusters):
        for i, images in clusters.items():
            print(f"Cluster {i + 1}:")
            for image_name in images:
                print(f"  {image_name}")

    def plot_silhouette_scores(self, max_clusters=10):
        image_features, _ = self._extract_features()
        n_samples = image_features.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)

        silhouette_avgs = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_features)
            silhouette_avg = silhouette_score(image_features, kmeans.labels_)
            silhouette_avgs.append(silhouette_avg)

        plt.plot(range(2, max_clusters + 1), silhouette_avgs)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.show()

    def save_clusters(self, clusters, dest_folder):
        for label, images in clusters.items():
            cluster_folder = os.path.join(dest_folder, f"Cluster_{label + 1}")
            os.makedirs(cluster_folder, exist_ok=True)
            for image_name in images:
                src_path = os.path.join(self.image_folder, image_name)
                dest_path = os.path.join(cluster_folder, image_name)
                shutil.copy(src_path, dest_path)

    def _compute_md5(self, image_path):
        with open(image_path, "rb") as f:
            file_data = f.read()
            md5_hash = hashlib.md5(file_data).hexdigest()
        return md5_hash

    def print_cluster_counts(self, clusters):
        for i, images in clusters.items():
            print(f"Cluster {i + 1}: {len(images)} images")

    def find_optimal_clusters(self, min_clusters=2, max_clusters=10):
        image_features, _ = self._extract_features()
        n_samples = image_features.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)

        silhouette_avgs = []
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_features)
            silhouette_avg = silhouette_score(image_features, kmeans.labels_)
            silhouette_avgs.append(silhouette_avg)

        optimal_clusters = silhouette_avgs.index(max(silhouette_avgs)) + min_clusters
        return optimal_clusters

    def visualize_clusters(self, clusters):
        image_features, _ = self._extract_features()
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(image_features)

        for cluster_id, image_list in clusters.items():
            cluster_features = reduced_features[[image_list.index(image_name) for image_name in image_list]]
            plt.scatter(cluster_features[:, 0], cluster_features[:, 1], label=f'Cluster {cluster_id + 1}')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    image_folder = "/Users/gatilin/test_img_split_point_data_0304/images"
    dest_folder = "/Users/gatilin/test_img_split_point_data_0304/cluster7"  # 目的文件夹

    text = "This is a scene of a outdoor"
    image_cluster = ImageCluster(image_folder, text)
    n_clusters = image_cluster.find_optimal_clusters(min_clusters=2, max_clusters=15)
    print(f"Optimal number of clusters: {n_clusters}")
    clusters, silhouette_avg = image_cluster.cluster_images(n_clusters=n_clusters)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    image_cluster.print_clusters(clusters)
    image_cluster.print_cluster_counts(clusters)
    image_cluster.save_clusters(clusters, dest_folder)  # 保存聚类结果
    image_cluster.visualize_clusters(clusters)