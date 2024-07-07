from fastapi import FastAPI
from deepface import DeepFace
import requests, io, tempfile, os
from PIL import Image
from pydantic import BaseModel, HttpUrl
from mangum import Mangum


app = FastAPI()
handler = Mangum(app)


class User(BaseModel):
    imageid: str
    url: HttpUrl


class Item(BaseModel):
    user: User
    images: list[User]


@app.get("/")
def welcome():
    return "Welcome Here"


@app.post("/detect")
def detection(data: Item):
    try:
        userID, user_url = data.user.imageid, data.user.url
        image_data = data.images
        imageIDs = []
        images = []
        results = {}

        def get_image(url):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Use a context manager for Image.open to ensure proper resource management
                return Image.open(io.BytesIO(response.content))
            else:
                return None

        user = get_image(user_url)

        for image in range(len(image_data)):
            imageIDs.append(image_data[image].imageid)
            images.append(get_image(image_data[image].url))

        with (tempfile.NamedTemporaryFile(mode='wb', delete=True, suffix='.jpg') as user_):
            user.save(user_)  # Save the image directly to the temporary file
            user_.flush()  # Ensure data is written to disk
            img1 = user_.name
            for image in images:
                with tempfile.NamedTemporaryFile(mode="wb", delete=True, suffix='.jpg') as image_:
                    image.save(image_)  # Save the image directly to the temporary file
                    image_.flush()  # Ensure data is written to disk
                    result = DeepFace.verify(
                        img1_path=img1,
                        img2_path=image_.name,
                        model_name="ArcFace",
                        threshold=0.4,
                        detector_backend="yunet",
                        enforce_detection=False,
                        align=False, )
                    results[f"{imageIDs[images.index(image)]}"] = result['verified']
                    # Optional: Close the temporary file after processing
                    image_.close()
            user_.close()  # No need to close explicitly within 'with' block

        return results

    except Exception as e:
        return {'error': str(e)}
