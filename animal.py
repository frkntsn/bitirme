import os
import requests
from PIL import Image
from io import BytesIO

# Klasör oluştur
os.makedirs("animals", exist_ok=True)

# Türler
animals = ["cat", "dog", "bird", "fish"]

# Her türden 4 görsel çek (picsum random kullanıyoruz)
image_list = []

for animal in animals:
    for i in range(4):
        url = f"https://loremflickr.com/320/320/{animal}"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        img = img.resize((256, 256))
        image_list.append(img)

# Grid oluştur (4x4)
grid_size = 4
img_size = 256

grid_img = Image.new("RGBA", (grid_size * img_size, grid_size * img_size))

for index, img in enumerate(image_list):
    x = (index % grid_size) * img_size
    y = (index // grid_size) * img_size
    grid_img.paste(img, (x, y))

# Kaydet
grid_img.save("animal_grid.png")

print("animal_grid.png oluşturuldu!")