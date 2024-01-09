import torch
from accelerate import Accelerator

accelerator = Accelerator()
dataloader = torch.utils.data.DataLoader(range(9), batch_size=5)
dataloader = accelerator.prepare(dataloader)
batch = next(iter(dataloader))
gathered_items = accelerator.gather_for_metrics(batch)
len(gathered_items)