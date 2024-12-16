from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
import os
os.environ['HF_HOME']='huggingface'
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from data_utils import CustomDataset,Conversions
from torch.utils.data import DataLoader

def main():
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    data_set=CustomDataset(data_path='datas/train',data_type='train')
    data_0=data_set[2]
    
    conversation_helper=Conversions()
    # Image
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    # image = Image.open(requests.get(url, stream=True).raw)
    image=data_0['image'][0]
    origin_w,origin_h = image.size
    new_w,new_h = origin_h//2,origin_w//2
    image=image.resize((new_w,new_h))

    conversation=conversation_helper.parse_conversation(data_0['instruction'])

    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

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
    print(output_text)


if __name__=='__main__':
    main()