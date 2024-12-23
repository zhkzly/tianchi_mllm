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


def get_profiler():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            dir_name="logs/profiler"
        ),
        profile_memory=True,
        record_shapes=True,
    )


def main(data_path="datas/train", task_type="0", data_type="train"):
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
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
    print(f"the text_prompt:{text_prompt[6:9]}")
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
    print(f"chat template process finished")

    print(f"inputs process finished")
    processed_output_texts = []
    raw_output_texts = []

    profiler = get_profiler()

    for text_prompt, image in tqdm(
        zip(text_prompt, datas["image"]), total=len(text_prompt), desc="inference"
    ):
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # 需要总的任务进行分别观察
        # print(f"the output_text:{output_text}")
        output_text = [re.sub("'", '"', text) for text in output_text]
        raw_output_texts.append(output_text)
        # print(f"the raw_output_text:{raw_output_texts[-1]}")
        # print(f"the type of output_text:{(output_text[0])}")
        # logger.info(f"the output_text:{output_text}")
        processed_output_texts.append(convert_to_json(output_text[0]))
        profiler.step()
    positive_pred = [
        1
        for pred, label in zip(processed_output_texts, datas["output"])
        if pred[0] == label
    ]
    print(f"positive_pred_tasks_{task_type}:{len(positive_pred)}")
    print(f"output_texts_tasks_{task_type}:{len(processed_output_texts)}")
    print(
        f"evaluation score_tasks_{task_type}:{len(positive_pred)/len(processed_output_texts)}"
    )
    print(f"saving the prediction results_tasks_{task_type}")
    task = "图片分类" if task_type == "0" else "意图分类"
    saving_results = {
        "task": task,
        "total_num": len(datas["instruction"]),
        "positive_pred": len(positive_pred),
        "score": len(positive_pred) / len(processed_output_texts),
    }
    saving_results["id_label_pred_raw_image"] = [
        [id, label, pred[0], raw, image_id]
        for id, label, pred, raw, image_id in zip(
            datas["id"],
            datas["output"],
            processed_output_texts,
            raw_output_texts,
            datas["image_id"],
        )
    ]

    with open(
        os.path.join("datas/train", f"{data_type}_task_" + task_type + "_results.json"),
        "w",
    ) as f:
        json.dump(saving_results, f, ensure_ascii=False)
    print(f"prediction results saved")


if __name__ == "__main__":
    data_path = "datas/train"
    task_type = "1"
    data_type = "train"
    for task_type in ["1", "1"]:
        main(data_path=data_path, task_type=task_type, data_type=data_type)