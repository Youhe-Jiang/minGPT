import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

from mingpt.utils import set_seed
set_seed(42)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
class AdditionDataset(Dataset):
    """
    Returns addition problems of up to some number of digits in the inputs. Recall
    that all GPT cares about are sequences of integers, and completing them according to
    patterns in the data. Therefore, we have to somehow encode addition problems
    as a sequence of integers.

    The sum of two n-digit numbers gives a third up to (n+1)-digit number. So our
    encoding will simply be the n-digit first number, n-digit second number,
    and (n+1)-digit result, all simply concatenated together. Because each addition
    problem is so structured, there is no need to bother the model with encoding
    +, =, or other tokens. Each possible sequence has the same length, and simply
    contains the raw digits of the addition problem.

    As a few examples, the 2-digit problems:
    - 85 + 50 = 135 becomes the sequence [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 becomes the sequence [0, 6, 3, 9, 0, 4, 5]
    etc.

    We will also only train GPT on the final (n+1)-digits because the first
    two n-digits are always assumed to be given. So when we give GPT an exam later,
    we will e.g. feed it the sequence [0, 6, 3, 9], which encodes that we'd like
    to add 6 + 39, and hope that the model completes the integer sequence with [0, 4, 5]
    in 3 sequential steps.

    fun exercise: does it help if the result is asked to be produced in reverse order?
    """

    def __init__(self, ndigit, split):
        self.split = split # train/test
        self.ndigit = ndigit
        self.vocab_size = 10 # 10 possible digits 0..9
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = ndigit + ndigit + ndigit + 1 - 1

        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028"
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y
'''
# create a dataset for e.g. 2-digit addition
ndigit = 2
train_dataset = AdditionDataset(ndigit=ndigit, split='train')
test_dataset = AdditionDataset(ndigit=ndigit, split='test')

from mingpt.model import GPT, GPTConfig, GPT1Config

# initialize a baby GPT model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=12, n_head=12, n_embd=768)

model = GPT(mconf)
'''
import os
import sys
import torch
import torch.nn as nn
from torch.nn import Linear, Module, Sequential
from torch import distributed as dist
import torchvision
import numpy as np
from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear
from torch.optim import SGD
from torch.optim import Adam
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
)
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, run_tests
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

class TestUnevenParamShard(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self):
        """Test FSDP with parameter shards."""
        # create a dataset for e.g. 2-digit addition
        ndigit = 2
        train_dataset = AdditionDataset(ndigit=ndigit, split='train')
        test_dataset = AdditionDataset(ndigit=ndigit, split='test')

        from mingpt.model import GPT, GPTConfig, GPT1Config

        # initialize a baby GPT model
        mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=12, n_head=12, n_embd=768)

        model = GPT(mconf)
        print_peak_memory("Peak Memory before loading model", self.rank)
        my_lr = 0.1
        model = FSDP(model).cuda()
        print_peak_memory("Peak Memory before loading model", self.rank)
        print_peak_memory("Load model to GPU...", self.rank)
        optim = SGD(model.parameters(), lr=my_lr)
        '''
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=20, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('../num-20/test_mem_to_time_100/model_12'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
        '''

        for i in range(30):
            #data = torch.randn(1, 1024, 3, 3)
            #target = torch.randn(1, 1024, 3, 3)
            data = train_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=512,
                                num_workers=4)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.cuda()
                y = y.cuda()

                # forward the model
    
                logits, loss = model(x, y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                optim.step()

if __name__ == "__main__":
    run_tests()
