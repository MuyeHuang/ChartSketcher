import os
import sys
import math
import re
import json
import random
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageColor
import uuid

# Configure local vLLM service calls
MODEL_BASE = "http://localhost:8000/v1"  # Primary model service address

# Set model names to specified local paths
MODEL_NAME_main = "/path/to/main/model" 


class Entity:
    def __init__(self, entity_id, entity_type, *params):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.params = params
        self.color = "black"
        self.line_width = 1


class Scene:
    def __init__(self, background_image_path=None, width=None, height=None):
        if background_image_path is not None:
            try:
                self.image = Image.open(background_image_path).convert("RGB")
                self.draw = ImageDraw.Draw(self.image)
                self.width, self.height = self.image.size
            except FileNotFoundError:
                print(f"Background image not found at {background_image_path}. Creating a white canvas instead.")
                if width is None or height is None:
                    aspect_ratio = random.uniform(1/3, 3)
                    max_size = 800
                    if aspect_ratio >= 1:
                        width = max_size
                        height = int(max_size / aspect_ratio)
                    else:
                        width = int(max_size * aspect_ratio)
                        height = max_size
                self.image = Image.new("RGB", (width, height), "white")
                self.draw = ImageDraw.Draw(self.image)
                self.width, self.height = self.image.size
        else:
            if width is None or height is None:
                aspect_ratio = random.uniform(1/3, 3)
                max_size = 800
                if aspect_ratio >= 1:
                    width = max_size
                    height = int(max_size / aspect_ratio)
                else:
                    width = int(max_size * aspect_ratio)
                    height = max_size
            self.image = Image.new("RGB", (width, height), "white")
            self.draw = ImageDraw.Draw(self.image)
            self.width, self.height = self.image.size
        self.entities = {}

    def _get_line_width_normal(self):
        return max(1, int(min(self.width, self.height) * 0.005))

    def _get_point_size_normal(self):
        return max(3, int(min(self.width, self.height) * 0.02))

    def _get_line_width_border(self):
        return max(1, int(min(self.width, self.height) * 0.009))

    def _get_point_size_border(self):
        return max(3, int(min(self.width, self.height) * 0.026))

    def _validate_color(self, color):
        try:
            ImageColor.getrgb(color)
            return color
        except Exception:
            return random.choice(["black", "blue", "red"])

    def _is_bright_color(self, rgb):
        r, g, b = rgb
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > 128

    def _get_high_contrast_color(self, color):
        valid_color = self._validate_color(color)
        rgb = ImageColor.getrgb(valid_color)
        return "black" if self._is_bright_color(rgb) else "lightgray"

    def create_point(self, entity_id, x, y, color="black"):
        self.entities[entity_id] = Entity(entity_id, "point", x, y)
        self.entities[entity_id].color = color

    def create_line(self, entity_id, x1, y1, x2, y2, color="black"):
        self.entities[entity_id] = Entity(entity_id, "line", x1, y1, x2, y2)
        self.entities[entity_id].color = color

    def create_circle(self, entity_id, cx, cy, radius, color="black"):
        self.entities[entity_id] = Entity(entity_id, "circle", cx, cy, radius)
        self.entities[entity_id].color = color

    def create_rectangle(self, entity_id, x1, y1, x2, y2, color="black"):
        self.entities[entity_id] = Entity(entity_id, "rectangle", x1, y1, x2, y2)
        self.entities[entity_id].color = color

    def create_arrow(self, entity_id, x1, y1, x2, y2, color="black"):
        self.entities[entity_id] = Entity(entity_id, "arrow", x1, y1, x2, y2)
        self.entities[entity_id].color = color

    def translate(self, entity_id, dx, dy):
        entity = self.entities.get(entity_id)
        if entity:
            params = list(entity.params)
            if entity.entity_type in ["point", "line", "rectangle", "arrow"]:
                for i in range(0, len(params), 2):
                    params[i] += dx
                    params[i + 1] += dy
                self.entities[entity_id].params = tuple(params)
            elif entity.entity_type == "circle":
                for i in range(0, len(params) - 1, 2):
                    params[i] += dx
                    params[i + 1] += dy
                self.entities[entity_id].params = tuple(params)

    def rotate(self, entity_id, angle, center_x, center_y):
        entity = self.entities.get(entity_id)
        if entity:
            params = list(entity.params)
            radians = -math.radians(angle)
            if entity.entity_type in ["point", "line", "rectangle", "arrow"]:
                for i in range(0, len(params), 2):
                    x, y = params[i], params[i + 1]
                    y_inverted = 1 - y
                    center_y_inverted = 1 - center_y
                    new_x = center_x + math.cos(radians) * (x - center_x) - math.sin(radians) * (y_inverted - center_y_inverted)
                    new_y_inverted = center_y_inverted + math.sin(radians) * (x - center_x) + math.cos(radians) * (y_inverted - center_y_inverted)
                    new_y = 1 - new_y_inverted
                    params[i] = new_x
                    params[i + 1] = new_y
                self.entities[entity_id].params = tuple(params)
            elif entity.entity_type == "circle":
                x, y, radius = params
                y_inverted = 1 - y
                center_y_inverted = 1 - center_y
                new_x = center_x + math.cos(radians) * (x - center_x) - math.sin(radians) * (y_inverted - center_y_inverted)
                new_y_inverted = center_y_inverted + math.sin(radians) * (x - center_x) + math.cos(radians) * (y_inverted - center_y_inverted)
                new_y = 1 - new_y_inverted
                self.entities[entity_id].params = (new_x, new_y, radius)

    def delete(self, entity_id):
        if entity_id in self.entities:
            del self.entities[entity_id]
        else:
            print(f"Entity {entity_id} does not exist, cannot delete.")

    def render(self):
        border_line_width = self._get_line_width_border()
        border_point_size = self._get_point_size_border()
        normal_line_width = self._get_line_width_normal()
        normal_point_size = self._get_point_size_normal()

        for entity_id, entity in self.entities.items():
            border_color = self._get_high_contrast_color(entity.color)
            if entity.entity_type == "point":
                x, y = self._convert_to_pixels(entity.params[0], entity.params[1])
                self.draw.ellipse(
                    [x - border_point_size, y - border_point_size, x + border_point_size, y + border_point_size],
                    fill=border_color
                )
                self.draw.ellipse(
                    [x - normal_point_size, y - normal_point_size, x + normal_point_size, y + normal_point_size],
                    fill=entity.color
                )
            elif entity.entity_type == "line":
                x1, y1 = self._convert_to_pixels(entity.params[0], entity.params[1])
                x2, y2 = self._convert_to_pixels(entity.params[2], entity.params[3])
                self.draw.line([x1, y1, x2, y2], fill=border_color, width=border_line_width)
                self.draw.line([x1, y1, x2, y2], fill=entity.color, width=normal_line_width)
            elif entity.entity_type == "circle":
                cx, cy = self._convert_to_pixels(entity.params[0], entity.params[1])
                radius = int(entity.params[2] * min(self.width, self.height))
                self.draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                                  outline=border_color, width=border_line_width)
                self.draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                                  outline=entity.color, width=normal_line_width)
            elif entity.entity_type == "rectangle":
                x1, y1, x2, y2 = entity.params
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                px1, py1 = self._convert_to_pixels(x_min, y_min)
                px2, py2 = self._convert_to_pixels(x_max, y_max)
                self.draw.rectangle([px1, py1, px2, py2], outline=border_color, width=border_line_width)
                self.draw.rectangle([px1, py1, px2, py2], outline=entity.color, width=normal_line_width)
            elif entity.entity_type == "arrow":
                x1, y1 = self._convert_to_pixels(entity.params[0], entity.params[1])
                x2, y2 = self._convert_to_pixels(entity.params[2], entity.params[3])
                self._draw_arrow(x1, y1, x2, y2, border_color, border_line_width)
                self._draw_arrow(x1, y1, x2, y2, entity.color, normal_line_width)
        return self.image

    def _convert_to_pixels(self, x, y):
        return int(x * self.width), int(y * self.height)

    def _draw_arrow(self, x1, y1, x2, y2, color, width):
        self.draw.line([x1, y1, x2, y2], fill=color, width=width)
        arrow_size = width * 5
        arrow_angle = math.pi / 6
        angle = math.atan2(y2 - y1, x2 - x1)
        left_x = x2 - arrow_size * math.cos(angle - arrow_angle)
        left_y = y2 - arrow_size * math.sin(angle - arrow_angle)
        right_x = x2 - arrow_size * math.cos(angle + arrow_angle)
        right_y = y2 - arrow_size * math.sin(angle + arrow_angle)
        self.draw.line([x2, y2, left_x, left_y], fill=color, width=width)
        self.draw.line([x2, y2, right_x, right_y], fill=color, width=width)


def parse_pseudocode(scene, pseudocode):
    lines = pseudocode.strip().split('\n')
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "END":
            print("Encountered END, stopping further parsing.")
            break
        command = tokens[0]
        if command == "create_point":
            entity_id = tokens[1]
            x, y = float(tokens[2]), float(tokens[3])
            color = tokens[4] if len(tokens) > 4 else "black"
            scene.create_point(entity_id, x, y, color)
        elif command == "create_line":
            entity_id = tokens[1]
            x1, y1, x2, y2 = map(float, tokens[2:6])
            color = tokens[6] if len(tokens) > 6 else "black"
            scene.create_line(entity_id, x1, y1, x2, y2, color)
        elif command == "create_circle":
            entity_id = tokens[1]
            cx, cy, radius = float(tokens[2]), float(tokens[3]), float(tokens[4])
            color = tokens[5] if len(tokens) > 5 else "black"
            scene.create_circle(entity_id, cx, cy, radius, color)
        elif command == "create_rectangle":
            entity_id = tokens[1]
            x1, y1, x2, y2 = map(float, tokens[2:6])
            color = tokens[6] if len(tokens) > 6 else "black"
            scene.create_rectangle(entity_id, x1, y1, x2, y2, color)
        elif command == "create_arrow":
            entity_id = tokens[1]
            x1, y1, x2, y2 = map(float, tokens[2:6])
            color = tokens[6] if len(tokens) > 6 else "black"
            scene.create_arrow(entity_id, x1, y1, x2, y2, color)
        elif command == "translate":
            entity_id = tokens[1]
            dx, dy = float(tokens[2]), float(tokens[3])
            scene.translate(entity_id, dx, dy)
        elif command == "rotate":
            entity_id = tokens[1]
            angle = float(tokens[2])
            cx, cy = float(tokens[3]), float(tokens[4])
            scene.rotate(entity_id, angle, cx, cy)
        elif command == "delete":
            entity_id = tokens[1]
            scene.delete(entity_id)


def extract_pseudocode(text):
    pattern = re.compile(r"BEGIN(.*?)END", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""


def build_structured_content(user_input, image_path=None):
    content = []
    if "<image>" in user_input and image_path:
        parts = user_input.replace("<image>", "")
        if parts.strip():
            content.append({"type": "text", "text": parts.strip()})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"file:///path/to/data{image_path}"}
        })
    else:
        content.append({"type": "text", "text": user_input})
    return content


def get_chat_completion(conversation, model_name, base_url):
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": conversation,
        "temperature": 0.01,
        "max_tokens": 2048,
        "do_sample": True,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def main():
    print("Add <image> placeholder in the text for multimodal conversation.")
    conversation_main = []
    temp_dir = '/path/to/temp/output' # Please fill in the intermediate inference image path folder you want to save here.

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    image_counter = 0

    user_input = input("You: ").strip()
    if user_input.lower() == "clear":
        conversation_main = []
        print("Conversation context cleared.")
        return

    image_path = None
    if "<image>" in user_input:
        print("Image tag detected, please enter the image path (without file:// prefix):") # On my Ascend machine, you inexplicably need to add 'file://.' Delete it if you don't need it.
        image_path = input().strip()

    structured_content = build_structured_content(user_input, image_path)
    user_message = {
        "role": "user",
        "content": structured_content
    }
    conversation_main.append(user_message)

    if image_path:
        initial_image_path = image_path
    else:
        print("No initial image path provided, program ending.")
        return

    commands_history = []
    latest_image = None

    while True:
        try:
            response_main = get_chat_completion(conversation_main, MODEL_NAME_main, MODEL_BASE)
        except Exception as e:
            print(f"Error requesting main model: {e}")
            break

        try:
            reply_main = response_main['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error parsing response: {e}")
            break

        print(f"Main Model: {reply_main}")
        conversation_main.append({
            "role": "assistant",
            "content": reply_main
        })

        pseudocode = extract_pseudocode(reply_main)
        if pseudocode:
            commands_history.append(pseudocode)
            scene = Scene(initial_image_path)
            for commands in commands_history:
                parse_pseudocode(scene, commands)
            image = scene.render()
            image_counter += 1
            latest_image_path = os.path.join(temp_dir, f"output_{image_counter}.png")
            image.save(latest_image_path)
            latest_image = latest_image_path
            print(f"Image saved to {latest_image_path}")
        else:
            print("No valid pseudocode found. ")
            break

        feedback_content = []
        feedback_content.append({
            "type": "image_url",
            "image_url": {"url": f"file://path/to/data{latest_image_path}"}
        })
        feedback_message = {
            "type": "text",
            "text": ' '
        }
        feedback_messages = [{
            "role": "user",
            "content": feedback_content + [feedback_message]
        }]
        conversation_main.append(feedback_messages[0])


if __name__ == "__main__":
    main()
