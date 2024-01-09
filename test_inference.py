from peft import PeftModel,get_peft_model
from transformers import AutoTokenizer, AutoModel
import torch
from utils import IND4EVAL
import json
from accelerate import Accelerator
accelerator = Accelerator()

device = torch.device(0)
model_path = "/workspace/pangyunhe/models/ZhipuAI/chatglm3-6b-32k"
lora_path = "/workspace/pangyunhe/source_code/finetune_basemodel_demo/2024-1-5latest/checkpoint-2250"
author_data_path = "/workspace/pangyunhe/dataset/whoiswho/train_author.json"
pub_data_path = "/workspace/pangyunhe/dataset/whoiswho/train_pub.json"
eval_path = '/workspace/pangyunhe/source_code/finetune_basemodel_demo/dataset/val_data.json'

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
with open(eval_path, "r", encoding="utf-8") as f: #从author数据中sample出来的训练数据
    eval_data = json.load(f)
eval_dataset = IND4EVAL(
    (eval_data,author_data,pub_data),
    tokenizer,
    max_source_length = 16000,
    max_target_length = 128,
) 
print('done reading dataset')

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids', 'author', 'pub', )}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    # ['input_ids']
    return batch_input,batch['author'],batch['pub']
dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = 1,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)
result = []
for batch in val_data:
    batch_input, author, pub = batch

    response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 128, return_dict_in_generate=True, output_scores=True)
    # response = model.generate(input_ids=batch['input_ids'],max_length=batch['input_ids'].shape[-1] + 128, return_dict_in_generate=True, output_scores=True, )
    # response = response[0, batch["input_ids"].shape[-1]:]
    tokenizer.decode(response[0], skip_special_tokens=True)
    yes_prob, no_prob = response.scores[0][0,YES_TOKEN_IDS],response.scores[0][0,NO_TOKEN_IDS]
    res = 1 if yes_prob > no_prob else 0
    batch_result = {
        "author":author,
        "pub":pub,
        "result":res,
        "logits": [yes_prob, no_prob]
    }
    gathered_items = accelerator.gather_for_metrics(batch_result)
    result.append(gathered_items)
    breakpoint()
    # metric.add_batch(gathered_items)
print('done')

    #TODO 1.9日测试一下Deepspeed单机多卡的eval，用nvidia-smi看一下模型推理是否是多卡并行的
    # 用accelerate来分布式推理
        
