import logging
import math
import os
import time
import warnings
from argparse import ArgumentParser
from functools import partial
from typing import List, Any

import hivemind
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchmetrics import Accuracy
from transformers import AutoTokenizer


def lamb_step(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        ratio = exp_avg / denom / bias_correction1 + weight_decay * param

        ratio_norm = ratio.norm().clamp(min=1e-9)  # configurable?
        param_norm = param.norm()

        step_size = lr * param_norm / ratio_norm

        param.add_(ratio, alpha=-step_size)


class LAMB(torch.optim.Optimizer):
    r"""Implements LAMB algorithm. Based on PyTorch's Adam"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, clip_max_norm=None):
        betas = float(betas[0]), float(betas[1])
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.clip_max_norm = clip_max_norm
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        if self.clip_max_norm is not None:
            iter_params = (param for group in self.param_groups for param in group['params'])
            torch.nn.utils.clip_grad_norm_(iter_params, self.clip_max_norm)

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("LAMB does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            lamb_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )
        return loss


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end.

    I believe I could have just used torch.nn.MultiheadAttention but their documentation is all but absent and code ugly
    so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PPL(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('loss', torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')

    def update(self, logits, targets) -> None:
        self.loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def compute(self) -> Any:
        return torch.exp(self.loss)


class GPT(pl.LightningModule):
    """the full GPT language model, with a context size of block_size."""

    def __init__(
            self,
            vocab_size,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            learning_rate=3e-4,
            n_embd=768,
            block_size=128,
            n_layer=12,
            n_head=4,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            attention="linformer",
            hidden_layer_multiplier=4,
    ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(resid_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.block_size = self.config.block_size
        self.apply(self._init_weights)
        self.blocks = nn.Sequential(
            *(Block(self.config) for _ in range(self.config.n_layer))
        )
        self.accuracy = Accuracy()
        self.train_ppl = PPL()
        self.val_ppl = PPL()
        rank_zero_info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self):
        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        return LAMB(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas, clip_max_norm=1.0
        )

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
                              :, :t, :
                              ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        # same action as inference
        logits = self(idx)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss.mean())
        self.log("train_ppl", self.train_ppl(logits, targets))

        optimizer: hivemind.Optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            self.log("lr_scheduler", param_group["lr"], on_step=True)
        self.log("local_epoch", optimizer.local_epoch, on_step=True)
        return loss

    def validation_step(self, batch, _):
        src, targets = batch
        logits = self(src)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("valid_loss", loss)

        values, indices = torch.max(logits, dim=-1)
        self.log(
            "val_acc", self.accuracy(indices, targets), prog_bar=True, on_epoch=True
        )
        self.log("val_ppl", self.val_ppl(logits, targets))


class CharDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        column_names = dataset.column_names

        dataset = dataset.map(
            lambda x: tokenizer(x["text"]),
            batched=True,
            remove_columns=column_names
        )

        # append 1 for next word prediction
        seq_length = self.block_size + 1

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # block size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            total_length = (total_length // seq_length) * seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + seq_length]
                    for i in range(0, total_length, seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=8,
        )
        self.data = dataset["input_ids"]
        rank_zero_info(f"data has {len(self.data)} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        chunk = self.data[i]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class CharDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int, block_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.dataset = load_dataset("wikitext", name="wikitext-103-v1")
        self.block_size = block_size
        self.train_dataset = CharDataset(self.dataset["train"], self.tokenizer, block_size)
        self.val_dataset = CharDataset(self.dataset["validation"], self.tokenizer, block_size)

    def sampler(self, dataset, shuffle):
        return DistributedSampler(
            dataset=dataset,
            num_replicas=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["HIVEMIND_RANK"]),
            shuffle=shuffle
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler(self.train_dataset, shuffle=True)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler(self.val_dataset, shuffle=False)
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


class DummyLearningRateScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    This is a dummy example that demonstrates how hivemind.Optimizer can use PyTorch LR Schedulers.
    This scheduler is based not on local iterations, but on the number of global updates performed by all peers.

    As an alternative, one can run a custom scheduler via Lightning callback based on LightningOptimizer.local_epoch.
    """

    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.num_warmup_steps, self.num_training_steps = num_warmup_steps, num_training_steps
        self._base_lr = self._prev_lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer, self._linear_decay_with_warmup)

    def _linear_decay_with_warmup(self, current_step: int):
        if current_step < self.num_warmup_steps:
            new_learning_rate = float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            new_learning_rate = max(
                0.0, float(self.num_training_steps - current_step
                           ) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            )
        if self.optimizer.param_groups[0]['lr'] != self._base_lr * new_learning_rate:
            print(f"changing learning rate to {self._base_lr * new_learning_rate:.5f}")
        return new_learning_rate


class HiveMindCallback(Callback):
    def __init__(self, target_batch_size: int, batch_size: int, num_warmup_steps: int, num_training_steps: int,
                 dht: hivemind.DHT):
        super().__init__()
        self.dht = dht
        self.target_batch_size = target_batch_size
        self.batch_size = batch_size
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    @property
    def optimizer_kwargs(self):
        return dict(
            matchmaking_time=5.0,
            averaging_timeout=30.0,
            delay_optimizer_step=False,
            grad_compression=hivemind.Float16Compression(),
            state_averaging_compression=hivemind.Float16Compression(),
            averager_opts=dict(request_timeout=1.0),
            verbose=True,
        )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(trainer.optimizers) > 1:
            raise MisconfigurationException("Hivemind only supports training with one optimizer.")
        (optimizer,) = trainer.optimizers
        # Set up a decentralized optimizer that will average with peers in background
        opt = hivemind.Optimizer(
            dht=self.dht,
            run_id="lightning_run",
            params=optimizer.param_groups,
            optimizer=type(optimizer),
            scheduler=partial(DummyLearningRateScheduler, num_warmup_steps=self.num_warmup_steps,
                              num_training_steps=self.num_training_steps),
            target_batch_size=self.target_batch_size,
            batch_size_per_step=self.batch_size * 4,
            **self.optimizer_kwargs
        )
        opt.load_state_from_peers()
        trainer.optimizers = [opt]

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logging.getLogger("pytorch_lightning").info("Shutting down hivemind DHT.")
        self.dht.shutdown()


if __name__ == "__main__":
    torch.set_num_threads(1)  # prevent peers from competing for cpu threads
    # ^-- this is necessary only when running multiple hivemind peers on a single machine

    parser = ArgumentParser()
    parser.add_argument("--n_layer", default=8, type=int)
    parser.add_argument("--n_head", default=16, type=int)
    parser.add_argument("--n_embd", default=1024, type=int)
    parser.add_argument("--learning_rate", default=6e-3, type=float)
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--target_batch_size", default=16384, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    args = parser.parse_args()
    

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open("input.txt").read()  # don't worry we won't run out of file handles

    dm = CharDataModule(args.batch_size, args.num_workers, args.block_size)
    model = GPT(
        vocab_size=dm.vocab_size,
        block_size=dm.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
    )


    class ConvergenceCallback(Callback):

        def __init__(self, threshold: float, monitor: str):
            self.threshold = threshold
            self.monitor = monitor
            self.logged = False

        def on_train_start(
                self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
        ) -> None:
            self.start = time.time()

        def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            loss = trainer.callback_metrics[self.monitor]
            if (loss <= self.threshold) and not self.logged:
                trainer.logger.log_metrics({"Time till loss reached": time.time() - self.start})
                self.logged = True


    initial_peers = os.environ["INITIAL_PEERS"].split(",")

    dht = hivemind.DHT(
        start=True,
        initial_peers=initial_peers,
    )

    num_steps = args.max_epochs * (len(dm.train_dataset) // args.target_batch_size)
    trainer = pl.Trainer(
        gpus=args.gpus,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        precision=16,
        gradient_clip_val=99999999999999999999999999,
        accumulate_grad_batches=4,
        callbacks=[
            ConvergenceCallback(threshold=4, monitor='valid_loss'),
            HiveMindCallback(
                target_batch_size=args.target_batch_size,
                batch_size=args.batch_size,
                num_warmup_steps=num_steps * 0.25,
                num_training_steps=num_steps,
                dht=dht
            ),
        ],
        # logger=WandbLogger(
        #     project="swarm",
        #     name=f"hivemind_2_rank{(os.environ['HIVEMIND_RANK'])}"
        # ),
        replace_sampler_ddp=False,
        enable_checkpointing=False
    )

    trainer.fit(model, datamodule=dm)
