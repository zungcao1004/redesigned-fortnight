from fastapi import FastAPI
from pydantic import BaseModel
from image_similarity import ImageSimilarity

app = FastAPI()
image_similarity_module = ImageSimilarity()
class ImageRequest(BaseModel):
    base64_image_string: str

@app.post("/image_search")
async def image_search(request: ImageRequest):
    base64_image_string = request.base64_image_string
    return {"productDocId": image_similarity_module.find_similar_images(base64_image_string=base64_image_string)}