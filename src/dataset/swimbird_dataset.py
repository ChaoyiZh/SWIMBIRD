import re
import os
import json
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset, Features, Value, Sequence
from qwen_vl_utils import process_vision_info
from .data_utils import *
from src.constants import SYSTEM_MESSAGE
from src.constants import (
    LATENT_TOKEN,
    LATENT_START_TOKEN,
    LATENT_END_TOKEN,
    PLAN_START_TOKEN,
    PLAN_END_TOKEN,
)

DEBUG_SAMPLE_PREVIEW_BUDGET = int(os.environ.get("SWIMBIRD_DEBUG_SAMPLE_PREVIEW", "0"))
DEBUG_BATCH_PREVIEW_BUDGET = int(os.environ.get("SWIMBIRD_DEBUG_BATCH_PREVIEW", "0"))


def _truncate_debug_text(value, limit=800):
    if value is None:
        return "<none>"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def normalize_conversations_debug(conversations):
    if isinstance(conversations, list):
        return conversations
    if isinstance(conversations, dict):
        keys = list(conversations.keys())
        if not keys:
            return []
        length = len(conversations[keys[0]])
        return [{k: conversations[k][i] for k in keys} for i in range(length)]
    return []

def is_plan_segment(text: str) -> bool:
    stripped = text.strip()
    return (
        PLAN_START_TOKEN in stripped
        and PLAN_END_TOKEN in stripped
        and stripped.index(PLAN_START_TOKEN) < stripped.index(PLAN_END_TOKEN)
    )

# ========== Dataset Class ==========
class SwimBirdSFTDataset(Dataset):
    def __init__(self, data_root: Union[str, List[str]]):
        super().__init__()
        self.raw_dataset = self._load_from_source(data_root)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, i: int):
        """Returns the raw sample at the given index, without preprocessing."""
        return self.raw_dataset[i]

    def _collect_json_files(self, path: Path) -> List[str]:
        """Helper to recursively find .json files from a given path."""
        if not path.exists():
            logging.warning(f"Path does not exist, skipping: {path}")
            return []

        if path.is_dir():
            # Find all .json files in the directory.
            found_files = [str(p) for p in path.glob('*.json') if p.is_file()]
            if not found_files:
                logging.warning(f"No .json files found in directory: {path}")
            return found_files

        if path.is_file() and path.suffix == '.json':
            return [str(path)]

        logging.warning(f"Path is not a valid .json file or directory, skipping: {path}")
        return []

    def _load_from_source(self, data_root: Union[str, List[str]]):
        """Main method to parse the data source and load the dataset."""
        # 1. Normalize the input into a list of strings
        if isinstance(data_root, str):
            # Split if it's a comma-separated string, otherwise wrap in a list
            paths_to_process = data_root.split(',') if ',' in data_root and not os.path.exists(data_root) else [
                data_root]
        elif isinstance(data_root, list):
            paths_to_process = data_root
        else:
            raise TypeError(f"Unsupported data_root type: {type(data_root)}. Must be str or list.")

        # 2. Collect all JSON files from all paths
        all_json_files = []
        for path_str in paths_to_process:
            path = Path(path_str.strip())
            all_json_files.extend(self._collect_json_files(path))

        # 3. Ensure we found at least one file
        if not all_json_files:
            raise ValueError("No valid .json files were found in any of the provided sources.")

        unique_files = sorted(list(set(all_json_files)))
        logging.info(f"Loading data from {len(unique_files)} unique JSON file(s).")

        # 4. Define the Generator Function
        # This function reads files one by one and yields standardized dictionaries
        def gen():
            for file_path in unique_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle case where JSON is a list of objects or a single object
                    if isinstance(data, dict):
                        data = [data]

                    for item in data:
                        # Extract only necessary fields and handle missing ones
                        yield {
                            "conversations": item.get("conversations", []),
                            "image": item.get("image", []),  # Default to empty list
                            "reasoning_image": item.get("reasoning_image", []),  # Default to empty list
                            "answer": item.get("answer", "")
                        }
                except Exception as e:
                    logging.warning(f"Error reading file {file_path}: {e}")
                    continue

        # 5. Define Explicit Features (Schema)
        # This prevents errors if the first example has an empty list and Arrow can't infer the type
        features = Features({
            'conversations': Sequence(feature={
                'from': Value('string'),
                'value': Value('string')
            }),
            'image': Sequence(Value('string')),
            'reasoning_image': Sequence(Value('string')),
            'answer': Value('string')
        })

        # 6. Create Dataset from Generator
        try:
            combined_dataset = HFDataset.from_generator(gen, features=features)
            logging.info(f"Successfully loaded a total of {len(combined_dataset)} samples.")
            return combined_dataset
        except Exception as e:
            raise IOError(f"Failed to load dataset from JSON files.") from e

# ========== data processing ==========
def cot_preprocess_function(example, max_pixels=5120*32*32, min_pixels=128*32*32, latent_max_pixels=64*32*32):
    """
    Converts the JSON format to the required format for multimodal training.

    Input format (example):
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "Question text with <image> placeholders..."},
            {"from": "gpt", "value": "Reasoning text with <image> placeholders..."}
        ],
        "image": ["path/to/question_image_1.png", ...],
        "reasoning_image": ["path/to/reasoning_image_1.png", ...],
        "answer": "Final answer string"
    }

    Output format:
    [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [...]}
    ]
    """
    conversations = example.get('conversations', []) 
    if isinstance(conversations, dict):
        try:
            keys = list(conversations.keys()) 
            length = len(conversations[keys[0]]) 
            new_conversations = [] 
            for i in range(length): 
                turn = {k: conversations[k][i] for k in keys}
                new_conversations.append(turn)
            conversations = new_conversations 
        except Exception as e: 
            print(example)
            logging.error(f"Failed to normalize conversations for ID {example.get('id', 'N/A')}: {e}")
            return None

    # 1. Separate human and gpt content from 'conversations'
    human_turn = None
    gpt_turn = None
    for turn in conversations:
        if turn.get('from') == 'human':
            human_turn = turn
        elif turn.get('from') == 'gpt':
            gpt_turn = turn

    if not human_turn or not gpt_turn:
        logging.warning(f"Sample {example.get('id', 'N/A')} is missing 'human' or 'gpt' turn, skipping.")
        return None # Return None to be filtered out later

    # ==================== 2. Process User Content ====================
    user_content = []
    question_text = human_turn.get('value', '')
    question_image_paths = example.get('image', [])
    
    # Use re.split and capture the delimiter to easily interleave text and images.
    # e.g., "text1<image>text2" -> ['text1', '<image>', 'text2']
    question_parts = re.split(r'(<image>)', question_text)
    
    question_image_idx = 0
    for part in question_parts:
        part = part.strip()
        if not part:
            continue
        
        if part == '<image>':
            # This is an image placeholder
            if question_image_idx < len(question_image_paths):
                img_path = question_image_paths[question_image_idx]
                try:
                    # The JSON contains image paths, so we need to load them with Pillow.
                    #image_data = Image.open(img_path).convert('RGB')
                    user_content.append({
                        "type": "image",
                        "image": img_path,
                        "max_pixels": max_pixels,
                        "min_pixels": min_pixels
                    })
                    question_image_idx += 1
                except FileNotFoundError:
                    logging.warning(f"User image not found: {img_path}")
                except Exception as e:
                    logging.error(f"Error loading user image {img_path}: {e}")
            else:
                logging.warning(f"An <image> tag was found in text, but there are not enough image paths in the 'image' list.")
        else:
            # This is a text part
            user_content.append({"type": "text", "text": part})

    # ================== 3. Process Assistant Content ==================
    assistant_content = []
    reasoning_text = gpt_turn.get('value', '')
    reasoning_image_paths = example.get('reasoning_image', [])

    reasoning_parts = re.split(r'(<image>)', reasoning_text)
    reasoning_image_idx = 0

    for part in reasoning_parts:
        part = part.strip()
        if not part:
            continue
        
        if part == '<image>':
            # This is a reasoning image placeholder
            if reasoning_image_idx < len(reasoning_image_paths):
                img_path = reasoning_image_paths[reasoning_image_idx]
                try:
                    #image_data = Image.open(img_path).convert('RGB')
                    assistant_content.append({
                        "type": "image",
                        "image": img_path,
                        "max_pixels": latent_max_pixels,
                    })
                    # assistant_content.insert(-1, {"type": "text", "text": "\n"})
                    assistant_content.append({"type": "text", "text": "\n"})
                    reasoning_image_idx += 1
                except FileNotFoundError:
                    logging.warning(f"Reasoning image not found: {img_path}")
                except Exception as e:
                    logging.error(f"Error loading reasoning image {img_path}: {e}")
            else:
                logging.warning(f"An <image> tag was found in the GPT response, but there are not enough image paths in the 'reasoning_image' list.")
        else:
            # This is a reasoning text part
            if is_plan_segment(part):
                assistant_content.append({
                    "type": "text",
                    "text": f"{part.strip()}\n"
                })
            else:
                # Remove "THOUGHT x: " tags
                cleaned_text = re.sub(r'THOUGHT \d+:\s*', '', part).strip()
                # cleaned_text = cleaned_text.replace('\n\n', '\n')
                if cleaned_text:
                    assistant_content.append({
                        "type": "text",
                        # Wrap with <reason> tag
                        "text": f"<reason>{cleaned_text}</reason>\n"
                    })

    # 4. Append the final answer
    final_answer = example.get('answer', '')
    if final_answer:
        assistant_content.append({
            "type": "text",
            "text": f"<answer>{final_answer}</answer>"
        })

    # 5. Assemble and return the final result
    if not user_content or not assistant_content:
        logging.warning(f"user_content or assistant_content is empty after processing, skipping sample {example.get('id', 'N/A')}")
        return None

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


# ==========  Collator ==========
class SwimBirdDataCollator:
    
    def __init__(self, processor, args):
        self.processor = processor
        self.args = args
        self._debug_sample_preview_remaining = DEBUG_SAMPLE_PREVIEW_BUDGET
        self._debug_batch_preview_remaining = DEBUG_BATCH_PREVIEW_BUDGET
        
        # Precompute token IDs once
        self.latent_token_idx = processor.tokenizer(LATENT_TOKEN, return_tensors="pt")["input_ids"][0]
        self.latent_start_idx = processor.tokenizer(LATENT_START_TOKEN, return_tensors="pt")["input_ids"][0]
        self.latent_end_idx = processor.tokenizer(LATENT_END_TOKEN, return_tensors="pt")["input_ids"][0]
        self.pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
        self.answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    def _debug_print_sample_preview(self, raw_examples, processed_examples, assistant_texts):
        if self._debug_sample_preview_remaining <= 0:
            return

        for raw_example, processed_example, assistant_text in zip(raw_examples, processed_examples, assistant_texts):
            if self._debug_sample_preview_remaining <= 0:
                break

            conversations = normalize_conversations_debug(raw_example.get("conversations", []))
            gpt_turn = next((turn for turn in conversations if turn.get("from") == "gpt"), {})
            assistant_turn = next(
                (turn for turn in processed_example if turn.get("role") == "assistant"),
                {"content": []},
            )

            print("=" * 80)
            print("[debug-seg0-sample]")
            print(f"sample_id={raw_example.get('id', 'N/A')}")
            print(f"raw_gpt={_truncate_debug_text(gpt_turn.get('value', ''))}")
            print(
                "processed_assistant_content="
                + _truncate_debug_text(assistant_turn.get("content", []), limit=1200)
            )
            print(f"assistant_template={_truncate_debug_text(assistant_text, limit=1200)}")
            self._debug_sample_preview_remaining -= 1

    def _debug_print_batch_summary(self, batch):
        if self._debug_batch_preview_remaining <= 0:
            return

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        image_out_mask = batch.get("image_out_mask")
        plan_start_id = self.processor.tokenizer.convert_tokens_to_ids(PLAN_START_TOKEN)
        plan_end_id = self.processor.tokenizer.convert_tokens_to_ids(PLAN_END_TOKEN)
        latent_id = self.processor.tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

        supervised_counts = (labels != -100).sum(dim=1).tolist()
        plan_start_counts = (input_ids == plan_start_id).sum(dim=1).tolist()
        plan_end_counts = (input_ids == plan_end_id).sum(dim=1).tolist()
        latent_counts = (input_ids == latent_id).sum(dim=1).tolist()
        image_out_counts = image_out_mask.sum(dim=1).tolist() if image_out_mask is not None else None

        print("=" * 80)
        print("[debug-seg0-batch]")
        print(f"batch_size={input_ids.shape[0]}")
        print(f"supervised_token_counts={supervised_counts}")
        print(f"plan_start_counts={plan_start_counts}")
        print(f"plan_end_counts={plan_end_counts}")
        print(f"latent_token_counts={latent_counts}")
        if image_out_counts is not None:
            print(f"image_out_mask_counts={image_out_counts}")

        for idx in range(min(2, input_ids.shape[0])):
            supervised_token_ids = labels[idx][labels[idx] != -100].tolist()
            decoded_supervised = self.processor.tokenizer.decode(supervised_token_ids, skip_special_tokens=False)
            decoded_input = self.processor.tokenizer.decode(input_ids[idx].tolist(), skip_special_tokens=False)
            print("-" * 80)
            print(f"[debug-seg0-batch] sample_index={idx}")
            print(f"input_text={_truncate_debug_text(decoded_input, limit=1200)}")
            print(f"supervised_text={_truncate_debug_text(decoded_supervised, limit=1200)}")

        self._debug_batch_preview_remaining -= 1
    
    def __call__(self, raw_examples):
        """Process batch of raw examples."""
        examples = [
            cot_preprocess_function(
                ex, 
                self.args.image_max_pixels, 
                self.args.image_min_pixels, 
                self.args.max_latent_token * 32 * 32, 
                # self.args.pattern
            ) 
            for ex in raw_examples
        ]

        texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in examples]

        texts = replace_visual_spectial_tokens(texts)

        image_inputs, _ = process_vision_info(examples,image_patch_size=16)
        
        user_examples = remove_assistant_images(examples)
        user_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in user_examples]
        user_image_inputs, _ = process_vision_info(user_examples,image_patch_size=16)
        
        assistant_examples = remove_user_images(examples)
        assistant_texts = [self.processor.apply_chat_template(ex, tokenize=False) for ex in assistant_examples]
        assistant_texts = replace_visual_spectial_tokens(assistant_texts)
        assistant_image_inputs, _ = process_vision_info(assistant_examples,image_patch_size=16)

        self._debug_print_sample_preview(raw_examples, examples, assistant_texts)
        
        # Step 6: Tokenize and create batches
        user_batch = self.processor(text=user_texts, images=user_image_inputs, return_tensors="pt", padding=True)
        assistant_batch = self.processor(text=assistant_texts, images=assistant_image_inputs, return_tensors="pt", padding=True)
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        
        # Step 7: Combine pixel values
        batch['pixel_values'] = user_batch.get('pixel_values', None)
        batch['image_grid_thw'] = user_batch.get('image_grid_thw', None)
        batch['pixel_values_latent'] = assistant_batch.get('pixel_values', None)
        batch['image_grid_thw_latent'] = assistant_batch.get('image_grid_thw', None)
       
        new_input_ids, new_attention_mask = replace_latent(
            batch["input_ids"], batch["attention_mask"], 
            self.latent_start_idx, self.latent_end_idx, self.latent_token_idx, 
            self.answer_start_token_pattern, self.pad_token_idx
        )

        batch["input_ids"] = new_input_ids
        batch["attention_mask"] = new_attention_mask

        labels = generate_labels_after_multi_token_start(
            batch["input_ids"], self.answer_start_token_pattern, 
            self.pad_token_idx, self.latent_token_idx
        )
        batch["labels"] = labels
        
        if batch['pixel_values_latent'] is not None:
            image_out_mask = mask_image_output_tokens(
                batch["input_ids"],
                self.latent_start_idx,
                self.latent_token_idx,
                self.latent_end_idx,
            )
            batch["image_out_mask"] = image_out_mask

        self._debug_print_batch_summary(batch)

        return batch


def make_supervised_data_module(processor, args):
    """Make dataset and collator for SwimBrid training."""
    
    dataset = SwimBirdSFTDataset(data_root=args.data_path)
    
    data_collator = SwimBirdDataCollator(
        processor=processor,
        args=args
    )
    
    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
