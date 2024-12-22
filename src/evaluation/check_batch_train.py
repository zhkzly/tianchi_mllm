import os

os.environ["HF_HOME"] = "huggingface"
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLProcessor,
)
from src.utils.fms_fsdp.utils.dummy_data_utils import CustomDataset, Conversions
from torch.utils.data import DataLoader
from rich import print
import logging
import json
from tqdm import tqdm
import re
from src.utils.fms_fsdp.utils.dummy_data_utils import convert_to_json
import math
from torch.profiler import ProfilerActivity
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)s",
)
logger = logging.getLogger(__name__)

MODEL_OUTPUT_CONFIG = {
    "use_cache": False,
    "return_dict": True,
    "output_hidden_states": False,
    "output_attentions": False,
}


def merge_dict(dict1, dict2):
    output = {**dict1, **dict2}
    return output


def get_profiler():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            dir_name="log/profiler"
        ),
        profile_memory=True,
        record_shapes=True,
    )


def main(data_path="datas/train", task_type="0", data_type="train"):
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    for name, param in model.named_parameters():
        print(f"Parameter {name} has dtype: {param.dtype}")
    print(f"the device of model:{model.device}")
    # Qwen2VLForConditionalGeneration
    logger.info(f"the type of model:{type(model)}")
    print(model)
    # Qwen2VLProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    logger.info(f"the type of processor:{type(processor)}")

    data_set = CustomDataset(
        data_path=data_path, data_type=data_type, task_type=task_type
    )
    total_data = len(data_set)
    print(f"total data:{total_data}")
    i = 0
    j = 0
    datas = {"instruction": [], "image": [], "output": [], "id": [], "image_id": []}
    print("starting the data process")
    while True:
        if len(data_set[j]["image"]) == 1:
            datas["instruction"].append(data_set[j]["instruction"])
            # print(data_set[j]['instruction'])
            # print(data_set[j]['output'])
            datas["image"].append(data_set[j]["image"][0])
            datas["output"].append(data_set[j]["output"])
            datas["id"].append(data_set[j]["id"])
            datas["image_id"].append(data_set[j]["image_id"])
            i += 1
            # print(i)
        j += 1
        if j >= total_data:
            break
        continue
    print("data process finished")
    print(f"evaluation num:{len(datas['instruction'])}")
    logger.debug(datas)
    conversation_helper = Conversions()

    def resize_image(image, scale=1):
        origin_w, origin_h = image.size
        logger.debug(f"the origin_w,origin_h:{origin_w,origin_h}")
        new_w, new_h = math.floor(origin_w / scale), math.floor(origin_h / scale)
        image = image.resize((new_w, new_h))
        logger.debug(f"the new_w,new_h:{new_w,new_h}")
        return image

    print("image process started")
    datas["image"] = [resize_image(image, scale=4) for image in datas["image"]]
    print("image process finished")
    conversation = [
        conversation_helper.parse_conversation(instruction)
        for instruction in datas["instruction"]
    ]
    print(f"starting the chat template process")
    logger.debug(conversation)
    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    print(f"the text_prompt:{text_prompt[0:5]}")
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    print(f"chat template process finished")

    print(f"inputs process finished")
    processed_output_texts = []
    raw_output_texts = []

    profiler = get_profiler()
    with torch.no_grad():
        for text_prompt, image in tqdm(
            zip(text_prompt, datas["image"]), total=len(text_prompt), desc="inference"
        ):
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            input_dict = merge_dict(inputs, MODEL_OUTPUT_CONFIG)
            output_ids = model(labels=inputs.input_ids, **input_dict)

            print("++++++++++++++++++++++++++++++")
            print(f"the type of output_ids:{type(output_ids)}")
            loss = output_ids.loss if hasattr(output_ids, "loss") else output_ids
            print(f"the loss:{loss}")
            print(f"the type of loss:{type(loss)}")
            print("+++++++++++++++++++++++++++++")


if __name__ == "__main__":
    data_path = "datas/train"
    task_type = "1"
    data_type = "val"
    for task_type in ["1", "1"]:
        main(data_path=data_path, task_type=task_type, data_type=data_type)
