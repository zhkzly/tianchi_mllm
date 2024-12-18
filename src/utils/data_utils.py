

from torch.utils.data import Dataset
import os
import json
from PIL import Image
from rich import print
from copy import deepcopy
import time
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(linneno)s')

class CustomDataset(Dataset):
    def __init__(self,data_path,data_type,task_type) -> None:
        super().__init__()
        self.data_path=data_path
        self.data_type=data_type
        self.task_type=task_type
        self._load_data()
    
    def _load_data(self):
        data_path=os.path.join(self.data_path,self.data_type+'.json')
        with open(data_path, 'r') as f:
            self.data=json.load(f)[self.task_type]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        images=[Image.open(os.path.join(self.data_path,'images',image)).convert('RGB') for image in self.data[index]['image']]
        outputs=deepcopy(self.data[index])
        # print(f"outputs:{outputs}")
        outputs['image']=images
        outputs['image_id']=self.data[index]['image']
        return outputs


class Conversions():
    def __init__(self) -> None:
        pass
    
    def get_prompt(self,raw_prompt):
        system_prompt=raw_prompt.split('\n')[0]
    @staticmethod
    def parse_conversation(dialogue):
        conversation = []
        lines = dialogue.strip().split('\n')
        if not lines[0].startswith("Picture"):
            for line in lines:
                if line.startswith("用户:"):
                    if line.endswith("<image>"):
                        conversation.append({"role": "user", "content": [{"type": "image"}]})
                    else:
                        conversation.append({"role": "user", "content": [{"type": "text", "text": line[len("用户: "):]}]})
                elif line.startswith("客服:"):
                    conversation.append({"role": "assistant", "content": [{"type": "text", "text": line[len("客服: "):]}]})
                elif line.startswith("<用户与客服的对话"):
                    continue
                elif line.startswith("你是一个"):
                    conversation.append({"role": "system", "content": [{"type": "text", "text": line}]})
                # elif line.startswith("Picture:"):
                #     conversation.append({"role": "user", "content": [{"type": "image"}]})
                else:
                    conversation.append({"role": "user", "content": [{"type": "text", "text": line}]})
            # print(conversation)
        
        else:
            for line in lines:
                if line.startswith("Picture"):
                    conversation.append({"role": "user", "content": [{"type": "image"}]})
                else:
                    conversation.append({"role": "user", "content": [{"type": "text", "text": line}]}) 
        return conversation


import re

def convert_to_json(input_string):
    if not input_string:
        return None, "输入字符串为空"
    
    # 去除前后空白字符
    stripped_string = input_string.strip()
    
    # 替换单引号为双引号
    corrected_string = stripped_string.replace("'", '"')
    
    # 替换常见特殊引号为标准双引号
    corrected_string = corrected_string.replace('‘', '"').replace('’', '"').replace('“', '"').replace('”', '"')
    
    
    
    # 确保整个字符串用方括号包围
    if not corrected_string.startswith('[') or not corrected_string.endswith(']'):
        corrected_string = f"[{corrected_string}]"
    
    # 移除多余的逗号或括号
    corrected_string = re.sub(r'\]\s*,\s*\[', ',', corrected_string)
    corrected_string = re.sub(r'(?<!\[)\s*,\s*(?!\])', ', ', corrected_string)
    
    # 确保每个元素都被双引号包围
    corrected_string = re.sub(r'(?<=\[|\,)\s*([^"\[\],]+)\s*(?=\,|\])', r'"\1"', corrected_string)
    
    try:
        result_list = json.loads(corrected_string)
        return result_list
    except json.JSONDecodeError as e:
        return None, f"JSON 解码错误: {e}"



IGNORE_INDEX = -100


def collate_fn(batch):
    # 各种batched data的处理[dict...]
    input_list = []
    output_list = []
    for data_dict in batch:
        input_list.append(data_dict["input"])
        output_list.append(data_dict["output"])
    return {"input": input_list, "output": output_list}


def tokenizer_helper(tokenizer, text_input, text_output, is_training=True):
    """help to generate model input for transformer model

    Args:
        tokenizer (_type_): _description_
        text_intput (_type_): _description_
        text_output (_type_): _description_

    Returns:
        dict: {'input_ids':input_ids,'attention_mask':attention_mask,'ingore_index':ingore_number}
    """
    # tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    # text: [dict]
    ignore_index = []
    texts = []
    logger.debug(f"the length of text_input is {len(text_input)}")
    for input, output in zip(text_input, text_output):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": input},
        ]
        # str
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(f"the type of text is {type(text)}")
        # print(text):不能直接使用 len(text)
        ignore_index.append([IGNORE_INDEX] * (len(text.split(" ")) - 1))
        label_texts = output + "<|im_end|>" + "<|endoftext|>"

        texts.append(text + label_texts)
        # 'input_ids','attention_mask',
    input_token = tokenizer(
        texts, padding=True, max_length=1024, truncation=True, return_tensors="pt"
    )
    # print(f"the shape of input_ids is {input_token['input_ids'].shape}")
    L = input_token["input_ids"][0].shape[0]
    inputs_len = []
    # for input,ignore in zip(input_token['input_ids'],ignore_index):
    #     inputs_len.append((len(input),len(ignore)))
    # print(f"the length of inputs_len is {inputs_len}")

    ignore_index = [ignore + [0] * (L - len(ignore)) for ignore in ignore_index]

    input_ids = input_token["input_ids"]
    attention_mask = input_token["attention_mask"]
    # ignore_number=[]
    # for index in ignore_index:
    #     ignore_number.append(len(index))
    # print(f"the output of ignore_number is {ignore_number}")

    logger.debug(f"the shape of ignore_index is {torch.LongTensor(ignore_index).shape}")
    model_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ignore_index": torch.LongTensor(ignore_index),
    }
    return model_input



def split_dataset(data_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,saving=True,seed=567):
    # 用来产生训练集、验证集、测试集的函数，输入的data_path 是一个文件夹，下面需要有一个data.json文件，里面是json格式的数据，
    # 该函数会将之后的结果保存到和data.json同级的train.json、val.json、test.json文件中。
    
    np.random.seed(seed)
    data_json_path=os.path.join(data_path, 'data.json')
    with open(data_json_path, 'r') as f:
        data_json=json.load(f)
    index=np.arange(len(data_json))
    shullfed_index=np.random.permutation(index)
    train_index=shullfed_index[:int(len(data_json)*train_ratio)]
    val_index=shullfed_index[int(len(data_json)*train_ratio):int(len(data_json)*(train_ratio+val_ratio))]
    test_index=shullfed_index[int(len(data_json)*(train_ratio+val_ratio)):]
    def split_task(data_json,index):
        task_0_index=[]
        task_1_index=[]
        for i in index:
            if data_json[i]['instruction'].startswith('Picture'):
                task_0_index.append(i)
            else:
                task_1_index.append(i)
        return task_0_index,task_1_index
    train_0_index,train_1_index=split_task(data_json,train_index)
    val_0_index,val_1_index=split_task(data_json,val_index)
    test_0_index,test_1_index=split_task(data_json,test_index)
    
    train_0_json=[data_json[i] for i in train_0_index]
    train_1_json=[data_json[i] for i in train_1_index]
    val_0_json=[data_json[i] for i in val_0_index]
    val_1_json=[data_json[i] for i in val_1_index]
    test_0_json=[data_json[i] for i in test_0_index]
    test_1_json=[data_json[i] for i in test_1_index]
    
    train_json=train_0_json+train_1_json
    val_json=val_0_json+val_1_json
    test_json=test_0_json+test_1_json   
    
    train_dict={'0':train_0_json,'1':train_1_json,'all':train_json}
    val_dict={'0':val_0_json,'1':val_1_json,'all':val_json}
    test_dict={'0':test_0_json,'1':test_1_json,'all':test_json}
    
    
    if saving:
        for json_name,json_data in zip(['train', 'val', 'test'], [train_dict, val_dict,test_dict]):
            with open(os.path.join(data_path, f'{json_name}.json'), 'w') as f:
                json.dump(obj=json_data, fp=f,ensure_ascii=False)
    return train_json, val_json, test_json
    

    
# maping :0.01s-0.02s   
# train [for ]:0.01s-0.02s
    
    
    



if __name__ == '__main__':
    import sys
    print(sys.path)


















