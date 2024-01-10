# -*- coding: utf-8 -*-
from peft import PeftModel,get_peft_model
from transformers import AutoTokenizer, AutoModel
import torch
from utils import IND4EVAL
import json
from accelerate import Accelerator
import tqdm
accelerator = Accelerator()

device = torch.device(0)
model_path = "/workspace/pangyunhe/models/ZhipuAI/chatglm3-6b-32k"
lora_path = "/workspace/pangyunhe/source_code/finetune_basemodel_demo/2024-1-5latest/checkpoint-2250"
author_data_path = "/workspace/pangyunhe/dataset/whoiswho/train_author.json"
pub_data_path = "/workspace/pangyunhe/dataset/whoiswho/train_pub.json"
eval_path = 'dataset/val_data.json'
batch_size = 1


model = AutoModel.from_pretrained(model_path, load_in_8bit=False, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
lora_model = PeftModel.from_pretrained(model, lora_path).half()
print('done loading peft model')

YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")


with open(author_data_path, "r", encoding="utf-8") as f:
    author_data = json.load(f)
with open(pub_data_path, "r" , encoding = "utf-8") as f:
    pub_data = json.load(f)
with open(eval_path, "r", encoding="utf-8") as f: 
    eval_data = json.load(f)
eval_dataset = IND4EVAL(
    (eval_data,author_data,pub_data),
    tokenizer,
    max_source_length = 16000,
    max_target_length = 128,
) 
print('done reading dataset')

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids', 'author', 'pub', 'label')}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return batch_input,batch['author'],batch['pub'],batch['label']

dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size ,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)
result = []
for batch in val_data:
    batch_input, author, pub, label = batch

    response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 128, return_dict_in_generate=True, output_scores=True)
    # tokenizer.decode(response[0], skip_special_tokens=True)
    yes_prob, no_prob = response.scores[0][:,YES_TOKEN_IDS],response.scores[0][:,NO_TOKEN_IDS]
    pred = yes_prob.ge(no_prob).to(int)
    node_result = [(author[i],pub[i],pred[i].item(),yes_prob[i].item(),no_prob[i].item(),label[i]) for i in range(batch_size)]
    batch_result = accelerator.gather_for_metrics(node_result)
    print(node_result)
    if accelerator.is_main_process:
        result.extend(batch_result)
if accelerator.is_main_process: 
    with open('result.json', 'w') as f:
        json.dump(result, f)


#TODO 将存储格式改为数据集的同样格式
        
