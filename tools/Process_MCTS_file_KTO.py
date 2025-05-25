#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random

def remove_duplicates(samples):
    import json
    seen = set()
    unique_samples = []
    for s in samples:
        s_str = json.dumps(s, sort_keys=True, ensure_ascii=False)
        if s_str not in seen:
            unique_samples.append(s)
            seen.add(s_str)
    return unique_samples

def label_node(node):
    """
    Label the node based on criteria.
    True if:
      - best_final_answer is not None and node is terminal with reward 1
    False if:
      - is_virtual is True
      - is_terminal is True and reward is 0 and best_final_answer is None (and not virtual)
    Return None otherwise.
    """
    if node.get("is_virtual", False):
        return False
    elif node.get("is_terminal", False) and node.get("reward", 1) == 1 and node.get("best_final_answer") is not None:
        return True
    elif node.get("is_terminal", False) and node.get("reward", 0) == 0 and node.get("best_final_answer") is None:
        return False
    else:
        return None

def build_kto_record(node, label_val):
    messages = []
    images = []

    for msg in node.get("conversation", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url", "").strip().replace("file://data/", '').strip()
                        if img_url:
                            images.append(img_url)
                        text_parts.append("<image>")
            text = " ".join(t.strip() for t in text_parts)
        else:
            text = str(content)

        messages.append({
            "role": role,
            "content": text.strip()
        })

    return {
        "messages": messages,
        "label": label_val,
        "images": images
    }

def collect_kto_data(node, kto_samples):
    lbl = label_node(node)
    if lbl is not None:
        record = build_kto_record(node, lbl)
        kto_samples.append(record)

    for child in node.get("children", []):
        collect_kto_data(child, kto_samples)

def parse_pseudocode(pseudocode):
    results = []
    lines = pseudocode.strip().split('\n')
    in_block = False
    for line in lines:
        line = line.strip()
        if line == "BEGIN":
            in_block = True
            continue
        elif line == "END":
            in_block = False
            continue
        if in_block and line:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            cmd = tokens[0]
            entity = tokens[1]
            results.append((cmd, entity, line))
    return results

def extract_entities_from_message(message_text):
    entities = set()
    segments = message_text.split("BEGIN")
    for seg in segments:
        if "END" not in seg:
            continue

        seg = "BEGIN" + seg
        parsed = parse_pseudocode(seg)
        for cmd, entity, _raw_line in parsed:
            entities.add(entity)
    return entities

def _get_last_assistant_message(conv):
    last_assistant_text = ""
    for m in reversed(conv):
        if m.get("role", "") == "assistant":
            content = m.get("content", "")
            if isinstance(content, str):
                last_assistant_text = content
            elif isinstance(content, list):
                temp_text = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        temp_text.append(item.get("text", ""))
                last_assistant_text = " ".join(temp_text)
            break
    return last_assistant_text

def extract_entities_from_node(node):
    conv = node.get("conversation", [])
    last_assistant_text = _get_last_assistant_message(conv)
    entity_set = extract_entities_from_message(last_assistant_text)
    return entity_set

def is_last_occurrence_of_entities(node_path, current_node_index):
    current_node = node_path[current_node_index]
    current_entities = extract_entities_from_node(current_node)
    if not current_entities:
        return True

    for i in range(current_node_index + 1, len(node_path)):
        next_node = node_path[i]
        next_entities = extract_entities_from_node(next_node)
        if current_entities & next_entities:
            return False
    return True

def is_best_final_answer_node(node):
    return (
        node.get("is_terminal", False)
        and node.get("reward", 0) == 1
        and node.get("best_final_answer") is not None
    )

def find_best_path(root):
    stack = [(root, [root])]
    while stack:
        node, path = stack.pop()
        if is_best_final_answer_node(node):
            return path
        for child in node.get("children", []):
            stack.append((child, path + [child]))
    return []

def build_kto_record_from_node(node):
    return build_kto_record(node, label_val=True)

def add_intermediate_node_sample(best_path, kto_samples):
    if len(best_path) < 3:
        return

    candidates = []
    for i in range(1, len(best_path) - 1):
        if is_last_occurrence_of_entities(best_path, i):
            candidates.append(best_path[i])

    if not candidates:
        return

    chosen_node = random.choice(candidates)
    record = build_kto_record_from_node(chosen_node)
    kto_samples.append(record)

def main():
    input_path = "results/input_data.json"
    output_path = "results/output_kto_data.json"

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kto_samples = []
    for record in data:
        search_tree = record.get("search_tree")
        if not search_tree:
            continue

        collect_kto_data(search_tree, kto_samples)

        best_path = find_best_path(search_tree)
        if best_path:
            # if random.random() <= 0.1: 
            add_intermediate_node_sample(best_path, kto_samples)

    unique_samples = remove_duplicates(kto_samples)

    total_count = len(unique_samples)
    true_count = sum(1 for s in unique_samples if s["label"] is True)
    false_count = total_count - true_count

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_samples, f, ensure_ascii=False, indent=2)

    print(f"Total saved {total_count} items, True={true_count}, False={false_count}.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
