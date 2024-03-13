import os
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from clip import clip
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class ImageCluster:
    def __init__(self, image_folder, model_name="ViT-B/32", device=None):
        self.image_folder = image_folder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def _extract_features(self):
        image_features = []
        image_list = []
        for image_name in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, image_name)
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_input)

            image_features.append(features.cpu().numpy())
            image_list.append(image_name)

        return np.vstack(image_features), image_list

    def cluster_images(self, n_clusters=2):
        image_features, image_list = self._extract_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_features)

        clusters = {i: [] for i in range(n_clusters)}
        for image_name, label in zip(image_list, kmeans.labels_):
            clusters[label].append(image_name)

        silhouette_avg = silhouette_score(image_features, kmeans.labels_)
        return clusters, silhouette_avg

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

if __name__ == "__main__":
    image_folder = "/Users/gatilin/PycharmProjects/dinner-cls/dogs"
    image_cluster = ImageCluster(image_folder)
    clusters, silhouette_avg = image_cluster.cluster_images(n_clusters=2)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    image_cluster.print_clusters(clusters)
    image_cluster.plot_silhouette_scores(max_clusters=10)