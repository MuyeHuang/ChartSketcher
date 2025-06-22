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
from PIL import Image, ImageDraw
import json
import os
import random
import time
import uuid
from PIL import Image, ImageDraw, ImageColor
import math
import random
random.seed(42)

# Config

TEST_JSON_PATH = "data/test_data.json"
PNG_FOLDER = "data/images"
SAVE_RESULT_PATH = "results/mcts_results.json"
TEMP_RENDER_PATH = "temp/mcts_render"
ROLLOUT_RENDER_PATH = "temp/mcts_rollout_render"
IS_SAMPLE = False
SAMPLE_NUM = 7000

MAX_WORKERS = 80
SAVE_EVERY = 50

MAX_DEPTH = 8
MAX_CHILDREN = 3
C_PUCT = 3.0
LAMBDA_LEN = 0                  # Used for penalizing/not penalizing node depth in UCB. If penalization is desired, set a small coefficient.
SIMULATIONS_LIMIT = 15
SUCCESS_LIMIT = 3
EPSILON = 1e-8

NEW_MODEL_ENDPOINT = "http://localhost:8000/v1/chat/completions"
ROLLOUT_SERVER = "http://localhost:7000/v1/chat/completions"
NEW_MODEL_NAME = "your/model/name"
MAX_TOKENS = 1500
DO_SAMPLE = True
TEMPERATURE = 0.1
HIGH_TEMP = 1.3                 # It is not recommended to set the value too high. Overly unstable sampling can easily harm performance.

# Drawing Related

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
            self.image = Image.open(background_image_path).convert("RGB")
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
        except:
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
                    params[i+1] += dy
                self.entities[entity_id].params = tuple(params)
            elif entity.entity_type == "circle":
                for i in range(0, len(params)-1, 2):
                    params[i] += dx
                    params[i+1] += dy
                self.entities[entity_id].params = tuple(params)

    def rotate(self, entity_id, angle, center_x, center_y):
        entity = self.entities.get(entity_id)
        if entity:
            params = list(entity.params)
            radians = -math.radians(angle)
            if entity.entity_type in ["point", "line", "rectangle", "arrow"]:
                for i in range(0, len(params), 2):
                    x, y = params[i], params[i+1]
                    y_inverted = 1 - y
                    center_y_inverted = 1 - center_y
                    new_x = center_x + math.cos(radians) * (x - center_x) - math.sin(radians) * (y_inverted - center_y_inverted)
                    new_y_inverted = center_y_inverted + math.sin(radians) * (x - center_x) + math.cos(radians) * (y_inverted - center_y_inverted)
                    new_y = 1 - new_y_inverted
                    params[i] = new_x
                    params[i+1] = new_y
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
            print(f"Entity {entity_id} not found.")

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


# Utility Functions

def extract_pseudocode(text):
    pattern = re.compile(r"BEGIN(.*?)END", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return []
    return [m.strip() for m in matches]

def parse_pseudocode(scene, pseudocode):
    lines = pseudocode.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]
        if cmd == "END":
            break
        elif cmd == "create_point":
            entity_id = tokens[1]
            x, y = float(tokens[2]), float(tokens[3])
            color = tokens[4] if len(tokens) > 4 else "black"
            scene.create_point(entity_id, x, y, color)
        elif cmd == "create_line":
            entity_id = tokens[1]
            x1, y1, x2, y2 = map(float, tokens[2:6])
            color = tokens[6] if len(tokens) > 6 else "black"
            scene.create_line(entity_id, x1, y1, x2, y2, color)
        elif cmd == "create_circle":
            entity_id = tokens[1]
            cx, cy, radius = map(float, tokens[2:5])
            color = tokens[5] if len(tokens) > 5 else "black"
            scene.create_circle(entity_id, cx, cy, radius, color)
        elif cmd == "create_rectangle":
            entity_id = tokens[1]
            x1, y1, x2, y2 = map(float, tokens[2:6])
            color = tokens[6] if len(tokens) > 6 else "black"
            scene.create_rectangle(entity_id, x1, y1, x2, y2, color)
        elif cmd == "create_arrow":
            entity_id = tokens[1]
            x1, y1, x2, y2 = map(float, tokens[2:6])
            color = tokens[6] if len(tokens) > 6 else "black"
            scene.create_arrow(entity_id, x1, y1, x2, y2, color)
        elif cmd == "translate":
            entity_id = tokens[1]
            dx, dy = float(tokens[2]), float(tokens[3])
            scene.translate(entity_id, dx, dy)
        elif cmd == "rotate":
            entity_id = tokens[1]
            angle = float(tokens[2])
            cx, cy = float(tokens[3]), float(tokens[4])
            scene.rotate(entity_id, angle, cx, cy)
        elif cmd == "delete":
            entity_id = tokens[1]
            scene.delete(entity_id)

def build_structured_content(user_input, image_path=None):
    content = []
    segments = user_input.split("<image>")
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if seg:
            content.append({"type": "text","text": seg})
        if i < len(segments)-1:
            if image_path and os.path.exists(image_path):
                content.append({"type":"image_url","image_url": {"url": f"file://data/{os.path.abspath(image_path)}"}})
            else:
                content.append({"type":"text","text":"[Warning: Missing image!]"})
    return content

def is_right(final_answer, conversation, label):
    import requests
    import time
    import re
    processed_conversation = []
    user_encountered = False

    for entry in conversation:
        if entry['role'] == 'user':
            if not user_encountered:
                user_encountered = True
            else:
                continue

        new_entry = entry.copy()
        if isinstance(new_entry.get('content'), list):
            new_entry['content'] = [
                item for item in new_entry['content']
                if item.get('type') != 'image_url'
            ]
        processed_conversation.append(new_entry)

    PREFIX = (
        "Please judge the correctness of the subjective question based on the following content. "
        "First, briefly analyze or calculate, and finally, add a [true] tag if correct, and a [false] tag if incorrect. "
        "Note that you only need to focus on whether the answer conclusion is correct, do not analyze and solve the problem yourself. You mainly look at others' work, mainly relying on the conversation and final answer. "
        "The correctness of the process is not judged. At the same time, you do not have the option of \"neither right nor wrong\", you must draw a conclusion of right or wrong. \n"
        "The principles are as follows: label refers to the correct answer. If the label is a number, it is called a numerical answer. If the label is not a numerical value, it is called a non-numerical answer. Years are non-numerical answers, for example, 2012 and 2011 are not the same year. "
        "For non-numerical questions, you need to carefully analyze and understand before judging right or wrong. For color-type answers, similar colors are acceptable. "
        "For numerical answers, you need to judge right or wrong after excluding factors such as units, and a 1% error in the number is allowed. "
        "If an API error occurs, as long as the correct answer can be found before the error, it is acceptable. 1% tolerance means that for all numerical labels, within the +-1% error range, try to count them as correct. "
        "For example, numerical labels 27 and 28. The final answer obtained by the assistant can contain redundant information, but it must directly answer the question asked. If it does not directly answer, consider judging it as wrong. Think carefully and be the strictest teacher. \n\n"
    )

    TRUE_FALSE_PATTERN = re.compile(r'\[(true|false)\]', re.IGNORECASE)

    def parse_deepseek_response(response_text: str) -> bool:
        match = TRUE_FALSE_PATTERN.search(response_text)
        if match:
            value = match.group(1).lower()
            return value == 'true'
        return False

    MODEL_CONFIG = {
        "api_url": ROLLOUT_SERVER,
        "model": "your/judge/model/name",
        "temperature": 0.5,
        "max_tokens": 8192,
        "top_p": 0.8,
        "retries": 3,
        "retry_delay": 5
    }

    OPENAI_API_KEY = ""

    def send_deepseek_request_sync(prompt: str) -> str:
        payload = {
            "model": MODEL_CONFIG["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": MODEL_CONFIG["temperature"],
            "max_tokens": MODEL_CONFIG["max_tokens"],
            "top_p": MODEL_CONFIG["top_p"]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}" if OPENAI_API_KEY else ""
        }

        for attempt in range(1, MODEL_CONFIG["retries"] + 1):
            try:
                resp = requests.post(
                    MODEL_CONFIG["api_url"],
                    json=payload,
                    headers=headers,
                    timeout=600
                )
                if resp.status_code != 200:
                    print(f"API Error {resp.status_code}: {resp.text}")
                    if attempt < MODEL_CONFIG["retries"]:
                        time.sleep(MODEL_CONFIG["retry_delay"])
                    else:
                        return ""
                else:
                    resp_data = resp.json()
                    reply = resp_data['choices'][0]['message']['content'].strip()
                    return reply
            except Exception as e:
                print(f"Request failed (attempt {attempt}), error: {e}")
                if attempt < MODEL_CONFIG["retries"]:
                    time.sleep(MODEL_CONFIG["retry_delay"])
                else:
                    print("Max retries reached.")
                    return ""

        return ""

    content = (
        f"This is the conversation content: {processed_conversation}, "
        f"This is the robot's answer: {final_answer}, "
        f"This is the correct answer: {label}"
    )
    prompt = PREFIX + content + '\nPlease think carefully and answer seriously! Add a [true] tag if correct, and a [false] tag if incorrect.'

    response_text = send_deepseek_request_sync(prompt)
    is_correct = parse_deepseek_response(response_text)

    return is_correct

def get_assistant_code_lines(conv):
    for turn in reversed(conv):
        if turn["role"] == "assistant":
            codes = extract_pseudocode(turn["content"])
            lines = []
            for code in codes:
                processed_lines = []
                for line in code.splitlines():
                    line = line.strip()
                    if line:
                        if ' ' in line:
                            last_space_index = line.rindex(' ')
                            processed_line = line[:last_space_index]
                            processed_lines.append(processed_line)
                        else:
                            pass
                lines.extend(processed_lines)
            return lines
    return []

def has_redundant_commands(conv_a, conv_b):
    lines_a = get_assistant_code_lines(conv_a)
    lines_b = get_assistant_code_lines(conv_b)

    for line in lines_a:
        if line in lines_b:
            return True
    return False

def has_subset_commands(conv_a, conv_b):
    lines_a = get_assistant_code_lines(conv_a)
    lines_b = get_assistant_code_lines(conv_b)
    i = 0
    for line in lines_a:
        if line in lines_b:
            i = i + 1
    j = 0
    for line in lines_b:
        if line in lines_a:
            j = j + 1
    if (i == len(lines_a) and i != 0) or (j == len(lines_b) and j != 0):
        return True
    else:
        return False

def call_model_api(conversation, temperature=TEMPERATURE):
    payload = {
        "model": NEW_MODEL_NAME,
        "messages": conversation,
        "max_tokens": MAX_TOKENS,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "temperature": temperature
    }
    try:
        r = requests.post(NEW_MODEL_ENDPOINT, json=payload)
        r.raise_for_status()
        resj = r.json()
        return resj["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error in call_model_api: {e}]"


# MCTSNode Definition


class MCTSNode:
    def __init__(self, conversation, depth=0, parent=None, label=None):
        self.conversation = conversation
        self.depth = depth
        self.parent = parent
        self.children = []

        self.is_virtual = False

        self.is_terminal = False
        self.is_fully_expanded = False

        self.Q = 0
        self.N = 0
        self.reward = 0
        self.best_final_answer = None

        self.node_id = id(self)
        self.label = label
        self.is_rollout = False

    def __repr__(self):
        return f"<Node id={self.node_id} depth={self.depth} Q={self.Q} N={self.N} r={self.reward} virtual={self.is_virtual}>"

    def ucb_score(self, c_puct=C_PUCT):
        if self.parent is None:
            return 0
        if self.is_virtual:
            return float('-inf')
        if self.N == 0:
            return float('inf')
        penalty = LAMBDA_LEN * (0.01 * self.depth + 0.3 * (math.exp(max(0, self.depth - 4) * 0.7) - 1))
        return (self.Q / (self.N + 1e-8)) + c_puct * math.sqrt(math.log(self.parent.N+1) / self.N) - penalty


# 1) Selection

def select_node(root):
    current = root
    while True:
        valid_children = [c for c in current.children if not c.is_virtual]
        if not valid_children or current.is_terminal:
            return current
        best_c = max(valid_children, key=lambda x: x.ucb_score())
        if best_c.is_terminal:
            return best_c
        current = best_c


# 2) Expansion


def expand_node(node):
    real_children_count = sum(not c.is_virtual for c in node.children)
    if real_children_count >= MAX_CHILDREN:
        node.is_fully_expanded = True
        return

    if node.depth >= MAX_DEPTH or node.is_terminal:
        node.is_terminal = True
        return

    attempts = 6
    new_children = []
    tried_signatures = set()

    for _ in range(attempts):
        real_children_count = sum(not c.is_virtual for c in node.children) + sum(not c.is_virtual for c in new_children)
        if real_children_count >= MAX_CHILDREN:
            break

        base_conv = node.conversation[:]
        assistant_reply = call_model_api(base_conv, temperature=HIGH_TEMP)

        sign = assistant_reply.strip()
        if sign in tried_signatures:
            continue
        tried_signatures.add(sign)

        tmp_conv = node.conversation[:]
        tmp_conv.append({"role":"assistant","content":assistant_reply})

        codes_list = extract_pseudocode(assistant_reply)
        if "BEGIN" not in assistant_reply and "END" not in assistant_reply and not codes_list:
            final_ans = assistant_reply
            success = is_right(final_ans, tmp_conv, node.label)
            child = MCTSNode(conversation=tmp_conv, depth=node.depth+1, parent=node, label=node.label)
            child.is_terminal = True
            child.reward = 1 if success else 0
            child.best_final_answer = final_ans if success else None
        else:
            if ("BEGIN" in assistant_reply or "END" in assistant_reply) and not codes_list:
                tmp_conv[-1]["content"] += "BEGIN\nit_is_false 1 2 3 4 5\nEND"

            if not codes_list:
                child = MCTSNode(conversation=tmp_conv, depth=node.depth+1, parent=node, label=node.label)
                child.is_terminal = True
                child.reward = 0
                child.best_final_answer = "[No valid code => forced fail]"
                child.is_virtual = True
                node.children.append(child)
                continue

            init_img_path = None
            if tmp_conv and tmp_conv[0]["role"] == "user":
                first_msg = tmp_conv[0]["content"]
                if isinstance(first_msg, list):
                    for seg in first_msg:
                        if seg["type"] == "image_url":
                            init_img_path = seg["image_url"]["url"].replace("file://data/","")
                            break

            if not init_img_path or not os.path.exists(init_img_path):
                child = MCTSNode(conversation=tmp_conv, depth=node.depth+1, parent=node, label=node.label)
                child.is_terminal = True
                child.reward = 0
                child.best_final_answer = "[No image => fail]"
                child.is_virtual = True
                node.children.append(child)
            else:
                scene = Scene(init_img_path)
                render_fail = False
                aggregated_code = ""
                for msg in tmp_conv:
                    if msg["role"] == "assistant":
                        code_blocks = extract_pseudocode(msg["content"])
                        if code_blocks:
                            aggregated_code += "\n".join(code_blocks) + "\n"

                try:
                    parse_pseudocode(scene, aggregated_code)
                except:
                    render_fail = True

                if render_fail:
                    tmp_conv[-1]["content"] = tmp_conv[-1]["content"].replace("BEGIN\nit_is_false 1 2 3 4 5\nEND","")
                    child = MCTSNode(conversation=tmp_conv, depth=node.depth+1, parent=node, label=node.label)
                    child.is_terminal = True
                    child.reward = 0
                    child.best_final_answer = "[Phase fail]"
                    child.is_virtual = True
                    node.children.append(child)
                else:
                    if not os.path.exists(TEMP_RENDER_PATH):
                        os.makedirs(TEMP_RENDER_PATH, exist_ok=True)
                    while True:
                        rndid = random.randint(10000000000, 99999999999)
                        outfn = os.path.join(TEMP_RENDER_PATH, f"expand_{rndid}_{threading.get_ident()}.png")
                        if not os.path.exists(outfn):
                            break

                    try:
                        scene.render().save(outfn)
                    except Exception as e:
                        child = MCTSNode(conversation=tmp_conv, depth=node.depth+1, parent=node, label=node.label)
                        child.is_terminal = True
                        child.reward = 0
                        child.best_final_answer = "[Render fail]"
                        child.is_virtual = True
                        node.children.append(child)


                    user_fb = [
                        {"type":"text","text": " "},
                        {"type":"image_url","image_url":{"url": f"file://data/{os.path.abspath(outfn)}"}}
                    ]
                    tmp_conv2 = tmp_conv[:]
                    tmp_conv2.append({"role":"user","content":user_fb})

                    child = MCTSNode(conversation=tmp_conv2, depth=node.depth+1, parent=node, label=node.label)

        duplicate = False
        if (not child.is_terminal) and (not child.is_virtual):
            not_virtual_children = [cand for cand in (node.children + new_children) if not cand.is_virtual]
            for existing_child in not_virtual_children:
                if has_redundant_commands(existing_child.conversation, child.conversation):
                    duplicate = True
                    child.is_virtual = True
                    child.is_terminal = True
                    child.best_final_answer = "[Duplicate]"
                    child.reward = 0
                    break
            if has_subset_commands(node.conversation, child.conversation):
                duplicate = True
                child.is_virtual = True
                child.is_terminal = True
                child.best_final_answer = "[Duplicate]"
                child.reward = 0
                if len(node.children + new_children) - len(not_virtual_children) < 3 :
                    new_children.append(child)

            if not duplicate:
                new_children.append(child)
        else:
            not_virtual_children = [cand for cand in (node.children + new_children) if not cand.is_virtual]
            if child.is_virtual == True and len(node.children + new_children) - len(not_virtual_children) < 3 :
                new_children.append(child)
            elif child.is_virtual == False:
                new_children.append(child)

        if sum(not c.is_virtual for c in node.children) >= MAX_CHILDREN:
            break

    node.children.extend(new_children)

    real_children_count = sum(not c.is_virtual for c in node.children)
    if len(new_children) == 0:
        node.is_terminal = True
    if real_children_count >= MAX_CHILDREN:
        node.is_fully_expanded = True


# 3) Rollout


def rollout(node):
    if node.is_terminal or node.is_virtual:
        return node.reward, node.best_final_answer, node.conversation

    import copy
    sim_conv = copy.deepcopy(node.conversation)
    cur_depth = node.depth

    while cur_depth < MAX_DEPTH:
        reply = call_model_api(sim_conv, temperature=0.4)
        sim_conv.append({"role":"assistant","content":reply})
        codes_list = extract_pseudocode(reply)
        if not codes_list:
            final_ans = reply
            success = is_right(final_ans, sim_conv, node.label)
            r = 1 if success else 0
            return r, final_ans, sim_conv
        else:
            init_img_path = None
            if sim_conv and sim_conv[0]["role"] == "user":
                first_msg = sim_conv[0]["content"]
                if isinstance(first_msg, list):
                    for seg in first_msg:
                        if seg["type"] == "image_url":
                            init_img_path = seg["image_url"]["url"].replace("file://data/","")
                            break
            if not init_img_path or not os.path.exists(init_img_path):
                return 0, "[No image => fail in rollout]", sim_conv

            scene = Scene(init_img_path)
            aggregated_code = ""
            for msg in sim_conv:
                if msg["role"] == "assistant":
                    code_blocks = extract_pseudocode(msg["content"])
                    if code_blocks:
                        aggregated_code += "\n".join(code_blocks) + "\n"

            render_fail = False
            try:
                parse_pseudocode(scene, aggregated_code)
            except:
                render_fail = True

            if render_fail:
                return 0, "[Render exception in rollout]", sim_conv

            if not os.path.exists(ROLLOUT_RENDER_PATH):
                os.makedirs(ROLLOUT_RENDER_PATH, exist_ok=True)
            randid = random.randint(10000000000,99999999999)
            outpath = os.path.join(ROLLOUT_RENDER_PATH, f"roll_{randid}_{threading.get_ident()}.png")
            scene.render().save(outpath)

            user_msg = [
                {"type":"text","text":" "},
                {"type":"image_url","image_url":{"url": f"file://data/{os.path.abspath(outpath)}"}}
            ]
            sim_conv.append({"role":"user","content": user_msg})

            cur_depth += 1

    return 0, "[Depth limit reached in rollout]", sim_conv


# 4) Backprop


def backprop(node, reward):
    cur = node
    while cur is not None:
        if not cur.is_virtual:
            cur.N += 1
            cur.Q += reward
        cur = cur.parent


# Main Function: mcts_search


def mcts_search(root, simulations=20, success_limit=3):
    success_count = 0
    total_simulations = 0

    while total_simulations < simulations and success_count < success_limit:
        leaf = select_node(root)

        if not leaf.is_terminal and not leaf.is_fully_expanded:
            expand_node(leaf)
            for child in leaf.children:
                if (
                    child.best_final_answer
                    and child.best_final_answer.strip() != ""
                    and child.best_final_answer
                    not in [
                        "[Render fail]", "[No image => fail]",
                        "[No image => fail in rollout]",
                        "[Render exception in rollout]",
                        "[Depth limit reached in rollout]",
                        "[Duplicate]"
                    ]
                    and not child.is_virtual
                ):
                    if child.is_terminal and child.reward == 1 and not child.is_virtual:
                        success_count += 1
            valid_children = [c for c in leaf.children if not c.is_virtual]
            if valid_children:
                leaf = random.choice(valid_children)

        if not leaf.is_terminal and not leaf.is_rollout and not leaf.is_virtual:
            r, ans, conv = rollout(leaf)
            leaf.reward = r
            leaf.is_rollout = True
        else:
            r = leaf.reward

        backprop(leaf, leaf.reward)
        total_simulations += 1

    best_child = None
    best_score = -1e9
    for c in root.children:
        if c.is_virtual:
            continue
        score = c.Q/(c.N + 1e-8)
        if score > best_score:
            best_score = score
            best_child = c
    return best_child


# Parallel Processing Example


def serialize_tree(node):
    data = {
        "node_id": node.node_id,
        "depth": node.depth,
        "Q": node.Q,
        "N": node.N,
        "reward": node.reward,
        "is_terminal": node.is_terminal,
        "is_rollout": node.is_rollout,
        "is_fully_expanded": node.is_fully_expanded,
        "is_virtual": node.is_virtual,
        "conversation": node.conversation,
        "best_final_answer": node.best_final_answer,
        "children": []
    }
    for c in node.children:
        data["children"].append(serialize_tree(c))
    return data

def find_best_terminal_by_path(tree_data):
    best_score = -float('inf')
    best_node = None
    best_pool = []

    def dfs(node, path):
        nonlocal best_score, best_node, best_pool

        new_path = path + [node]

        if node.get("is_terminal", False) and not node.get("is_virtual", False):
            ans = node.get("best_final_answer")
            if ans is not None and ans.strip() != "" and ans not in [
                "[Render fail]", "[No image => fail]", "[No image => fail in rollout]",
                "[Render exception in rollout]", "[Depth limit reached in rollout]", "[Duplicate]"
            ]:
                valid_path = path
                path_length = len(valid_path)
                if path_length > 0:
                    path_value = sum(n.get("Q", 0) / (n.get("N", 0) + EPSILON) for n in valid_path)
                    avg_score = path_value / path_length
                    valid_depth = valid_path[-1].get("depth", path_length - 1)
                else:
                    avg_score = 0.0
                    valid_depth = 0

                penalty = LAMBDA_LEN * (0.01 * valid_depth + 0.3 * (math.exp(max(0, valid_depth - 4) * 0.7) - 1))
                composite_score = avg_score - penalty

                if composite_score > best_score:
                    best_score = composite_score
                    best_node = node
                    best_pool = [node]
                elif composite_score == best_score:
                    best_pool.append(node)
        for child in node.get("children", []):
            dfs(child, new_path)

    dfs(tree_data, [])

    if best_node is None:
        return None, None

    return random.choice(best_pool), best_score

def single_question_mcts(idx, entry):
    imgname = entry.get("imgname","")
    query = entry.get("query","")
    label = entry.get("label","")

    user_input = query + " <image>"
    image_path = os.path.join(PNG_FOLDER, imgname)
    first_user_msg = build_structured_content(user_input, image_path)

    conversation = [{"role":"user","content": first_user_msg}]

    root_node = MCTSNode(conversation=conversation, depth=0, parent=None, label=label)

    best_leaf = mcts_search(root_node, simulations=SIMULATIONS_LIMIT, success_limit=SUCCESS_LIMIT)

    tree_data = serialize_tree(root_node)
    best_terminal, best_path_score = find_best_terminal_by_path(tree_data)
    result = {
        "index": idx,
        "imgname": imgname,
        "query": query,
        "label": label,
        "best_final_answer": best_terminal.get("best_final_answer", "") if best_terminal else None,
        "best_reward": best_terminal.get("reward", 0) if best_terminal else 0,
        "search_tree": tree_data,
        "path_score": best_path_score if best_terminal else None
    }
    return result

def main():
    start_time = time.time()

    if not os.path.exists(TEST_JSON_PATH):
        print(f"[Error] {TEST_JSON_PATH} not found.")
        sys.exit(1)

    try:
        with open(TEST_JSON_PATH,"r",encoding="utf-8") as f:
            test_data = json.load(f)
            if IS_SAMPLE == True:
                test_data = random.sample(test_data, SAMPLE_NUM)
    except Exception as e:
        print(f"[Error reading dataset] {e}")
        sys.exit(1)

    results = []
    if os.path.exists(SAVE_RESULT_PATH):
        try:
            with open(SAVE_RESULT_PATH,"r",encoding="utf-8") as rf:
                results = json.load(rf)
        except:
            pass
    done_indices = {x["index"] for x in results if x.get("best_final_answer") is not None}
    to_process = [(i,d) for i,d in enumerate(test_data) if i not in done_indices]

    print(f"Total {len(test_data)} items, {len(done_indices)} completed, {len(to_process)} remaining.")
    if not to_process:
        print("No data to process.")
        return

    lock = threading.Lock()
    all_res = results[:]

    def worker(idx,entry):
        try:
            r_ = single_question_mcts(idx, entry)
            with lock:
                all_res.append(r_)
        except Exception as e:
            print(f"[Error] idx={idx}, {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = []
        for idx,entry in to_process:
            futs.append(exe.submit(worker, idx, entry))
        count_done = 0
        for fut in as_completed(futs):
            fut.result()
            count_done += 1
            print(f"\rDone count: {count_done}, Time per task: {(time.time() - start_time)/count_done:.2f}s", end="", flush=True)
            if count_done % SAVE_EVERY == 0:
                with lock:
                    all_res.sort(key=lambda x: x["index"])
                    with open(SAVE_RESULT_PATH,"w",encoding="utf-8") as wf:
                        json.dump(all_res,wf,ensure_ascii=False,indent=2)
                print(f"[Info] finish={count_done}, partial saved => {SAVE_RESULT_PATH}")

    all_res.sort(key=lambda x: x["index"])
    with open(SAVE_RESULT_PATH,"w",encoding="utf-8") as wf:
        json.dump(all_res,wf,ensure_ascii=False,indent=2)
    print(f"All completed => {SAVE_RESULT_PATH}")

if __name__=="__main__":
    main()
