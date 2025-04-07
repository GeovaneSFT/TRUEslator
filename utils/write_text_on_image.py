"""This module contains a function to add text to an image with a bounding box."""
import textwrap
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def add_text(image: np.ndarray, text: str, contour: np.ndarray) -> np.ndarray:
    """
    Add text to an image with a bounding box with improved font handling and text positioning.
    Returns the modified image with text added.
    """
    try:
        # Fallback fonts in case primary font fails
        font_paths = [
            "./fonts/mangat.ttf",
            "./fonts/GL-NovantiquaMinamoto.ttf",
            "./fonts/fonts_animeace_i.ttf"
        ]

        # Convert image to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Get bounding box dimensions
        x, y, w, h = cv2.boundingRect(contour)

        # Skip if contour is invalid or too small
        if w < 10 or h < 10:
            return image

        # Initialize text parameters
        line_height = min(16, h // 3)
        font_size = min(16, h // 3)
        wrapping_ratio = 0.075

        # Try different fonts if primary font fails
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
                break
            except Exception:
                continue

        # Fallback to default font if all custom fonts fail
        if font is None:
            font = ImageFont.load_default()

        # Handle empty or invalid text
        if not text or not isinstance(text, str):
            text = "[No text]"

        # Initial text wrapping
        wrapped_text = textwrap.fill(text, width=max(1, int(w * wrapping_ratio)),
                                    break_long_words=True)
        lines = wrapped_text.split('\n')
        total_text_height = len(lines) * line_height

        # Adjust text size to fit the box
        max_attempts = 10
        attempt = 0
        while total_text_height > h and attempt < max_attempts:
            line_height = max(8, line_height - 2)
            font_size = max(8, font_size - 2)
            wrapping_ratio = min(0.3, wrapping_ratio + 0.025)

            try:
                font = ImageFont.truetype(font_paths[0], size=font_size)
            except Exception:
                font = ImageFont.load_default()

            wrapped_text = textwrap.fill(text, width=max(1, int(w * wrapping_ratio)),
                                        break_long_words=True)
            lines = wrapped_text.split('\n')
            total_text_height = len(lines) * line_height
            attempt += 1

        # Calculate vertical position with padding
        padding = max(2, h // 20)
        text_y = y + padding + (h - total_text_height) // 2

        # Draw text with improved centering
        for line in lines:
            try:
                text_length = draw.textlength(line, font=font)
            except Exception:
                text_length = len(line) * (font_size // 2)  # Fallback estimation

            # Horizontal centering with padding
            text_x = x + max(2, (w - text_length) // 2)

            # Draw text with outline for better visibility
            draw.text((text_x-1, text_y), line, font=font, fill=(255, 255, 255))
            draw.text((text_x+1, text_y), line, font=font, fill=(255, 255, 255))
            draw.text((text_x, text_y-1), line, font=font, fill=(255, 255, 255))
            draw.text((text_x, text_y+1), line, font=font, fill=(255, 255, 255))
            draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))

            text_y += line_height

        # Convert back to OpenCV format
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    except Exception as e:
        print(f"Error in add_text: {str(e)}")
        return image  # Return original image if any error occurs