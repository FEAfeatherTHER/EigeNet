import os
import json
from math import pi

from typing import Optional
from importlib.resources import files

import random
random.seed(114)
from glob import glob
import json
import random
from collections import defaultdict

def split_dataset(input_file, train_file, test_file, root_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    data_by_scene = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            item = json.loads(line.strip())
            
            for path_key in ['rir_path', 'depth_path']:
                if path_key in item and item[path_key]:
                    if os.path.isabs(item[path_key]) and item[path_key].startswith(root_dir):
                        item[path_key] = os.path.relpath(item[path_key], root_dir)
            
            data_by_scene[item['scene_name']].append(item)
                
    train_data = []
    test_data = []
    
    print(f"{'Scene Name':<20} | {'Total':<8} | {'Train':<8} | {'Test':<8}")
    print("-" * 55)
    
    for scene, items in data_by_scene.items():
        random.shuffle(items)
        
        split_idx = int(len(items) * train_ratio)
        
        for i, item in enumerate(items):
            if i < split_idx:
                item['split'] = 'train'
                train_data.append(item)
            else:
                item['split'] = 'test'
                test_data.append(item)
            
        train_count = split_idx
        test_count = len(items) - split_idx
        print(f"{scene:<20} | {len(items):<8} | {train_count:<8} | {test_count:<8}")
        
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("-" * 55)
    print("done!")
    print(f"train set ({len(train_data)} items) saved to: {train_file}")
    print(f"test set ({len(test_data)} items) saved to: {test_file}")


if __name__ == "__main__":
    HAA_ROOT = 'path/to/haa'
    INPUT_JSONL = 'path/to/haa.jsonl'
    TRAIN_JSONL = 'path/to/haa_train.jsonl'
    TEST_JSONL = 'path/to/haa_test.jsonl'
    split_dataset(
        input_file=INPUT_JSONL, 
        train_file=TRAIN_JSONL, 
        test_file=TEST_JSONL,
        root_dir=HAA_ROOT,
        train_ratio=0.8,
    )