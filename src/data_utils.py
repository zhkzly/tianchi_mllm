

from torch.utils.data import Dataset
import os
import json
from PIL import Image
from rich import print
from copy import deepcopy

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

    
    


if __name__ == '__main__':
    import sys
    print(sys.path)


















