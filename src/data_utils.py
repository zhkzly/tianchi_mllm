

from torch.utils.data import Dataset
import os
import json
from PIL import Image
from rich import print
from copy import deepcopy

class CustomDataset(Dataset):
    def __init__(self,data_path,data_type) -> None:
        super().__init__()
        self.data_path=data_path
        self.data_type=data_type
        self._load_data()
    
    def _load_data(self):
        data_path=os.path.join(self.data_path,self.data_type+'.json')
        with open(data_path, 'r') as f:
            self.data=json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        images=[Image.open(os.path.join(self.data_path,'images',image)).convert('RGB') for image in self.data[index]['image']]
        outputs=deepcopy(self.data[index])
        # print(f"outputs:{outputs}")
        outputs['image']=images
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
            else:
                conversation.append({"role": "user", "content": [{"type": "text", "text": line}]})
        # print(conversation)
        
        return conversation




if __name__ == '__main__':
    import sys
    print(sys.path)


















