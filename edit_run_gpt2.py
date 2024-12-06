import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from typing import Dict

from contextlib import contextmanager
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.gpt2 import GPT2LM
from models.gpt2_prefetch import PrefetchGPT2LM
from utils import event_measure


cfgs: Dict[str, Dict[str, int]] = {
    'gpt2_small': {'embed_dim': 768, 'num_heads': 12, 'num_layers': 12},
    'gpt2_medium': {'embed_dim': 1024, 'num_heads': 16, 'num_layers': 24},
    'gpt2_large': {'embed_dim': 1280, 'num_heads': 20, 'num_layers': 36},
    'gpt2_xl': {'embed_dim': 1600, 'num_heads': 25, 'num_layers': 48},
    'gpt3_6.7b': {'embed_dim': 4096, 'num_heads': 32, 'num_layers': 32},
    'gpt3_13b': {'embed_dim': 5200, 'num_heads': 40, 'num_layers': 40},
    'gpt3_175b': {'embed_dim': 12288, 'num_heads': 96, 'num_layers': 96},
}


class MockDataset(Dataset):
    def __init__(self, seq_len=1024):
        self.seq_len = seq_len

    def __len__(self):
        return float(1e9)  # Infinite dataset

    def __getitem__(self, idx):
        return torch.randint(low=0, high=50256, size=(self.seq_len,))
"""
class MockDataset(Dataset):
    def __init__(self, dsize, seq_len=1024):
        self.dsize = dsize
        self.data = []
        for i in range(dsize):
            self.data.append(torch.randint(low=0, high=50256, size=(seq_len,)))

    def __len__(self):
        return float('inf')  # Infinite dataset
        #return len(self.data)

    def __getitem__(self, idx):
        #actual_idx = idx % len(self.data)
        #d = self.data[idx]
        #return d
        return torch.randint(low=0, high=50256, size=(self.seq_len,))
        #return self.data[actual_idx]
"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2_xl',
                        const='gpt2_xl', nargs='?',
                        choices=['gpt2_small', 'gpt2_medium', 'gpt2_large', 'gpt2_xl',
                                 'gpt3_6.7b', 'gpt3_13b', 'gpt3_175b'],
                        help='model type')
    parser.add_argument('--enable-prefetch', action='store_true',
                        help='whether to enable prefetch optimization')
    parser.add_argument('--enable-cudnn-benchmark', action='store_true',
                        help='whether to enable cudnn benchmark option')
    parser.add_argument('--num-streams', type=int, default=3, help='# of prefetch streams')
    parser.add_argument('--warmups', type=int, default=2, help='# of warm up steps')
    return parser.parse_args()
def main():
    args = get_args()
    print("###############################")
    print("#           configs           #")
    print("###############################")
    print(vars(args))

    model_config = cfgs[args.model]

    if args.enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if args.enable_prefetch:
        model_config['num_prefetch_streams'] = args.num_streams
        model = PrefetchGPT2LM(**model_config).eval().cuda()
    else:
        model = GPT2LM(**model_config).eval().cuda()

    num_warmup = args.warmups
    SEQ_LEN = 1024
    BATCH_SIZE = 32  # Make batch size dynamic if needed
    #synthetic_dataset = MockDataset(50 + num_warmup, SEQ_LEN)
    synthetic_dataset = MockDataset(SEQ_LEN)

    dataloader = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Ensure dataset size is sufficient
    total_batches = len(dataloader)
    if total_batches <= num_warmup:
        raise ValueError("Dataset size is too small to accommodate warmup steps and performance testing.")

    fw_times = []
    for step, inp in enumerate(tqdm(dataloader)):
        if step < num_warmup:
            # Warmup phase (exclude timing)
            out = model(inp.cuda())
        else:
            # Timed evaluation
            with torch.no_grad(), event_measure() as result:
                out = model(inp.cuda())
            fw_times.append(result['time'])

    # Validate fw_times
    if len(fw_times) == 0:
        raise RuntimeError("No valid steps processed after warmup. Check dataset size or configuration.")

    # Compute average forward pass time and throughput
    avg_fw_time = np.mean(fw_times)
    avg_throughput = (SEQ_LEN * BATCH_SIZE) / (avg_fw_time / 1000)

    print(f"Avg. step time: {avg_fw_time:.2f} ms \tAvg. throughput: {avg_throughput:.2f} tokens/sec")


if __name__ == '__main__':
    main()