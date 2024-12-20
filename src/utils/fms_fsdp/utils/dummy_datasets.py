from dataclasses import dataclass
from typing import Dict, Sequence, List
from torch.utils.data import Dataset
import copy
import json
import torch
import transformers
from src.utils.fms_fsdp.utils.dummy_data_utils import Conversions, CustomDataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|im_end|>"
DEFAULT_UNK_TOKEN = "<|unk|>"


PROMPT_DICT = {
    "instruct_prompt": "[Instructions]:\n{instruction}\n\n[Response]:",
}


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def load_data(files):
    data = []

    for file in files:
        f = open(file, mode="r")
        if "jsonl" in file:
            _data = f.read()
            _data = _data.splitlines()
            _data = [json.loads(js) for js in _data]
        else:
            _data = json.load(f)

        f.close()

        data += _data

    return data


def process_example(example: Dict):
    return example.get("output")


def format_sources(example):
    return PROMPT_DICT["instruct_prompt"].format_map(example)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: list[str],
    ):
        super(SupervisedDataset, self).__init__()
        list_data_dict = load_data(data_path)
        sources = [format_sources(example) for example in list_data_dict]
        targets = [
            f"{process_example(example)}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        print("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class CustomSftDataset(Dataset):
    #
    def __init__(
        self,
        preprocessor,
        data_path,
        data_type="train",
        task_type="intent_classification",
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.data_path = data_path
        self.data = self._load_data()
        self.data_type = data_type
        self.task_type = task_type
        self.conversion = Conversions()

    def _load_data(self):
        data_path = os.path.join(self.data_path, self.data_type + ".json")
        with open(data_path, "r") as f:
            self.data = json.load(f)[self.task_type]

    def _helpe_genereate_prompt(self, instruction):
        instruction = self.conversion.parse_conversation(instruction["instruction"])
        instruction = self.preprocessor.apply_chat_template(
            instruction, add_general_prompt=True
        )
        return instruction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = [
            Image.open(os.path.join(self.data_path, "images", image))
            for image in self.data[index]["image"]
        ]
        outputs = deepcopy(self.data[index])
        # print(f"outputs:{outputs}")
        outputs["image"] = images
        outputs["image_id"] = self.data[index]["image"]
        # list[{},{},]
        outputs["instruction"] = self._helpe_genereate_prompt(outputs)
        return outputs


# def collate_fn(batch:List[List[Dict]]):
#     # helping organize the return of the dataset call
#     # 需要进行批量化的padding，得到一些掩码，和ignore,构造完整的 sft instruction


class DataSftCollator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.ignore_index = IGNORE_INDEX
        self.split_idx = self.preprocessor.tokenizer.convert_tokens_to_ids("\\n")

    def _get_full_prompts(self, instances):
        full_prompts = [
            [instance["instruction"] + instance["output"] + DEFAULT_EOS_TOKEN]
            for instance in instances
        ]

    def _get_exact_number_index(lst):
        for index, value in enumerate(reversed(lst), target_number=self.split_idx):
            if value == self.split_idx:
                # len(lst) - 1 - index 是为了计算原始列表中的索引位置
                last_index = len(lst) - 1 - index
                break
            else:
                last_index = None  # 如果没有找到，则设置为 None 或其他默认值

    def _get_response_start_idx(self, instances):
        response_index_list = [
            self._get_exact_number_index(instance["input_ids"].tolist(), self.split_idx)
            for instance in instances
        ]
        return response_index_list

    def _get_labels(self, input_ids, response_index_list):
        labels = []
        input_length = input_ids.shape[1]
        for ignore_length, input_id in zip(response_index_list, input_ids):
            ignore_index = torch.zeros_like(input_ids[0], dtype=torch.long)
            ignore_index[:ignore_length] = self.ignore_index
            label = torch.where(
                ignore_index == self.ignore_index, ignore_index, input_id
            )
            labels.append(label)
        return torch.stack(labels, dim=0)

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # 将 response和instruction进行拼接获取一个完整的text list,获取一个完整的image list
        #  之后进行preprocessor ，得到模型的输入
        full_prompts = self._get_full_prompts(instances)
        inputs_features_dict = self.preprocessor(
            text=full_prompts, images=instances, return_tensors="pt"
        )
        response_index_list = self._get_response_start_idx(instances)
        input_ids = inputs_features_dict["input_ids"]
        labels = self._get_labels(input_ids)
        return inputs_features_dict, labels


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
