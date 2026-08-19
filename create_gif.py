from pathlib import Path
from PIL import Image

VISUALIZATION_DIR = Path("visualizations")

image_files = [
    "sales_by_product.png",
    "sales_trend.png",
    "price_distribution.png",
    "sales_violin.png",
    "correlation_heatmap.png",
    "dashboard.png"
]

images = []

for file_name in image_files:

    file_path = VISUALIZATION_DIR / file_name

    if file_path.exists():
        image = Image.open(file_path).convert("RGB")
        images.append(image)

if not images:
    raise FileNotFoundError(
        "No visualization images were found."
    )

output_file = Path("dashboard_demo.gif")

images[0].save(
    output_file,
    save_all=True,
    append_images=images[1:],
    duration=1500,
    loop=0
)

print(
    f"GIF created successfully: {output_file.resolve()}"
)