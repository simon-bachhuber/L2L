import os

from moviepy.editor import ImageSequenceClip
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Directory containing the images
image_folder = "images"
output_video = "video.mp4"

# Output frame rate (1 frame per second)
framerate = 1

# Dynamically load a default font with a specific size
try:
    font = ImageFont.truetype("arial.ttf", size=48)  # Use Arial if available
except IOError:
    font = ImageFont.load_default(
        size=48
    )  # Fallback to default font if Arial is unavailable

# List of images sorted by their episode number
image_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith(".png")],
    key=lambda x: int(x.split("_")[-1].split(".")[0]),  # Sort by episode number
)

# Add overlayed text to each image
processed_images = []
for filename in image_files:
    # Load the image
    image_path = os.path.join(image_folder, filename)
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Create a drawing context
        draw = ImageDraw.Draw(img)

        # Add the filename as text overlay
        text = filename
        text_position = (10, 10)  # Top-left corner
        text_color = "white"
        box_color = "black"

        # Calculate text size using the default font
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # Draw a semi-transparent box behind the text
        box_position = [
            text_position[0],
            text_position[1],
            text_position[0] + text_width + 10,
            text_position[1] + text_height + 5,
        ]
        draw.rectangle(box_position, fill=box_color)

        # Draw the text
        draw.text(
            (text_position[0] + 5, text_position[1] + 2),
            text,
            fill=text_color,
            font=font,
        )

        # Convert the Pillow image to a numpy array
        processed_images.append(np.array(img))

# Create a video from the processed images
clip = ImageSequenceClip(processed_images, fps=framerate)
clip.write_videofile(output_video, codec="libx264", fps=framerate)
