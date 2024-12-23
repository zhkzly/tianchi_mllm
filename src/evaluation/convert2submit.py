import json
from pathlib import Path
import pandas as pd


# def convert2submit(test_file: Path, prediction_file: Path, save_path: Path):
#     pred_label_list = []

#     for line in open(prediction_file, "r"):
#         prediction_data = json.loads(line)

#         pred_label = prediction_data["predict"]
#         pred_label_list.append(pred_label)

#     test_data = json.load(open(test_file, "r"))
#     save_data = []
#     for i, example in enumerate(test_data):
#         example["predict"] = pred_label_list[i]
#         save_data.append(example)

#     df = pd.DataFrame(save_data)

#     df.to_csv(save_path, index=None, encoding="utf-8-sig")

import pandas as pd
import os
from src.utils.fms_fsdp.utils.dummy_data_utils import convert_to_json
import json
def convert_help(item):
    print(f"the type of item:{type(item)}")
    item=convert_to_json(item)

    return item[0]
def convert2submit(pred_file_path,file_name='task_all_train_pred.csv',columns_to_read=['id','predict'],save_name='submit.csv'):
    file_path=os.path.join(pred_file_path,file_name)
    df=pd.read_csv(filepath_or_buffer=file_path,usecols=columns_to_read)
    df['predict']=df['predict'].apply(convert_help)
    print(df)
    save_path=os.path.join(pred_file_path,save_name)
    df.to_csv(save_path,index=None,encoding='utf-8-sig')
    print('saving successfully !')




if __name__ == "__main__":

    pred_file_path='../datas/test1'
    convert2submit(pred_file_path=pred_file_path)

# end main
