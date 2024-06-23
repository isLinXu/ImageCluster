import os
import glob
import logging
import pickle
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm.autonotebook import tqdm
import torch
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClusterer:
    def __init__(self, img_folder, output_folder, model_name='clip-ViT-B-32', file_types=('*.jpg', '*.png'),
                 batch_size=128, threshold=0.9, min_community_size=10, init_max_size=1000):
        self.img_folder = img_folder
        self.output_folder = output_folder
        self.model_name = model_name
        self.file_types = file_types
        self.batch_size = batch_size
        self.threshold = threshold
        self.min_community_size = min_community_size
        self.init_max_size = init_max_size
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name} due to {str(e)}")
            raise

    def get_image_paths(self):
        img_names = []
        for file_type in self.file_types:
            img_names.extend(glob.glob(os.path.join(self.img_folder, file_type)))
        return img_names

    def encode_images(self, img_names):
        encoded_images = []
        for filepath in tqdm(img_names, desc="Encoding images"):
            encoded_images.append(torch.tensor(self.model.encode(Image.open(filepath))))
        return torch.stack(encoded_images)

    def community_detection(self, embeddings):
        cos_scores = util.cos_sim(embeddings, embeddings)
        top_k_values, _ = cos_scores.topk(k=min(self.min_community_size, len(cos_scores)), largest=True)
        extracted_communities = []
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= self.threshold:
                new_cluster = []
                top_val_large, top_idx_large = cos_scores[i].topk(k=min(self.init_max_size, len(cos_scores[i])),
                                                                  largest=True)
                top_idx_large = top_idx_large.tolist()
                top_val_large = top_val_large.tolist()
                if top_val_large[-1] < self.threshold:
                    for idx, val in zip(top_idx_large, top_val_large):
                        if val < self.threshold:
                            break
                        new_cluster.append(idx)
                else:
                    for idx, val in enumerate(cos_scores[i].tolist()):
                        if val >= self.threshold:
                            new_cluster.append(idx)
                extracted_communities.append(new_cluster)
        extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)
        unique_communities = []
        extracted_ids = set()
        for community in extracted_communities:
            add_cluster = True
            for idx in community:
                if idx in extracted_ids:
                    add_cluster = False
                    break
            if add_cluster:
                unique_communities.append(community)
                for idx in community:
                    extracted_ids.add(idx)
        return unique_communities

    def save_clusters(self, clusters, img_names):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)
        for i, cluster in enumerate(clusters):
            cluster_folder = os.path.join(self.output_folder, f"cluster_{i}")
            os.makedirs(cluster_folder, exist_ok=True)
            for idx in cluster:
                img_path = os.path.join(self.img_folder, img_names[idx])
                img = Image.open(img_path)
                img.save(os.path.join(cluster_folder, os.path.basename(img_path)))

    def cluster_images(self):
        img_names = self.get_image_paths()
        logger.info(f"Found {len(img_names)} images.")
        img_emb = self.encode_images(img_names)
        clusters = self.community_detection(img_emb)
        logger.info(f"Total number of clusters: {len(clusters)}")
        self.save_clusters(clusters, img_names)
        logger.info(f"Saved clusters to {self.output_folder}")


if __name__ == "__main__":
    img_folder = 'path/images'
    output_folder = 'path/cluster_images'
    clusterer = ImageClusterer(img_folder,
                               output_folder)
    clusterer.cluster_images()