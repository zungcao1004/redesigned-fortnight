import base64
import os
import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import ViTImageProcessor, ViTModel
from torch.nn.functional import cosine_similarity

class ImageSimilarity:
    def __init__(self, embedding_dir = "embeddings") -> None:
        self.embedding_dir = embedding_dir
        os.makedirs(self.embedding_dir, exist_ok=True)
        
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Initialize empty lists for URLs and embeddings
        self.all_image_urls = []
        self.all_image_embeddings = []
        self.all_image_doc_ids = []
        
        # Load existing embeddings
        self._load_existing_embeddings()


    def _load_existing_embeddings(self):
        """Load all precomputed embeddings and doc IDs from the embeddings folder."""
        for filename in os.listdir(self.embedding_dir):
            if filename.endswith(".pt"):
                embedding_path = os.path.join(self.embedding_dir, filename)

                # Decode the filename to get the original URL
                url = self._decode_filename_to_url(filename)

                # Load the embedding and doc ID from the file
                embedding, doc_id = torch.load(embedding_path)

                self.all_image_urls.append(url)
                self.all_image_embeddings.append(embedding)
                self.all_image_doc_ids.append(doc_id)
        
    def _infer(self, image):
        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.pooler_output   

    def _encode_url_to_filename(self, url):
        """Encode the URL using Base64 to create a valid filename."""
        url_bytes = url.encode('utf-8')
        encoded_url = base64.urlsafe_b64encode(url_bytes).decode('utf-8')
        return f"{encoded_url}.pt"

    def _decode_filename_to_url(self, filename):
        """Decode the Base64-encoded filename back to the original URL."""
        url_encoded = filename.split(".")[0]
        url_bytes = base64.urlsafe_b64decode(url_encoded)
        return url_bytes.decode('utf-8')
    
    def _decode_base64_image(self, base64_string):
        """Decode a base64 string to a PIL image."""
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image

    def store_embeddings(self, image_urls, doc_ids):
        """Store embeddings for a list of image URLs and their corresponding doc IDs."""
        for url, doc_id in zip(image_urls, doc_ids):
            embedding_path = os.path.join(self.embedding_dir, self._encode_url_to_filename(url))

            # Check if the embedding already exists
            if os.path.exists(embedding_path):
                print(f"Embedding already exists for {url}")
                continue

            try:
                # Download the image and convert to RGB
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw).convert("RGB")
                    image_embedding = self._infer([image])[0]  # Single image embedding

                    # Store the embedding and doc ID
                    torch.save((image_embedding, doc_id), embedding_path)
                    print(f"Embedding saved for {url} with doc ID: {doc_id}")
                else:
                    print(f"Error downloading image from {url}")
            except Exception as e:
                print(f"Error processing image from {url}: {e}")
    
    def find_similar_images(self, base64_image_string):
        """Find and return a list of tuples (URL, doc_id, similarity_score) for the top 5 similar images."""
        target_image = self._decode_base64_image(base64_image_string)
        embed_target = self._infer(target_image)

        # Calculate similarity scores and store in a dictionary
        similar_images = {}
        for embed_image, (url, doc_id) in zip(self.all_image_embeddings, zip(self.all_image_urls, self.all_image_doc_ids)):
            similarity_score = cosine_similarity(embed_target, embed_image, dim=1).item()
            if similarity_score > 0.6:
                similar_images[url] = (similarity_score, doc_id)

        # Sort the dictionary by similarity score in descending order
        sorted_similar_images = sorted(similar_images.items(), key=lambda x: x[1][0], reverse=True)

        return set(doc_id for url, (score, doc_id) in sorted_similar_images)