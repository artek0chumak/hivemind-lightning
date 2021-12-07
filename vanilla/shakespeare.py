import math
import os
import time
from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy


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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
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
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
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


class LearningRateDecayCallback(pl.Callback):
    def __init__(
            self, learning_rate, warmup_tokens=375e6, final_tokens=260e9, lr_decay=True
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.tokens = 0
        self.final_tokens = final_tokens
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        optimizer = trainer.optimizers[0]
        _, y = batch

        if self.lr_decay:
            self.tokens += (
                    y >= 0
            ).sum()  # number of tokens processed this step (i.e. label is not -100)

            if self.tokens < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
            else:
                # linear learning rate decay
                lr_mult = 1 - 0.9 * min(1, float(self.tokens - self.warmup_tokens) / float(
                    max(1, self.final_tokens - self.warmup_tokens)))
            lr = self.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

                pl_module.log("lr_scheduler", lr, on_step=True)


class CharDataset(Dataset):
    def __init__(self, data, block_size, chars=None):
        if chars is None:
            chars = sorted(set(data))

        self.chars = chars
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - (self.block_size + 1)

    def __getitem__(self, i):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        # i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        y = i + self.block_size + 1
        chunk = self.data[i:y]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class CharDataModule(pl.LightningDataModule):
    def __init__(self, text: str, batch_size: int, block_size: int):
        super().__init__()
        num_workers = 0
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_size = int(0.8 * len(text))
        self.train_dataset = CharDataset(text[:train_size], block_size)
        self.val_dataset = CharDataset(
            text[train_size:], block_size, chars=self.train_dataset.chars
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @property
    def vocab_size(self) -> int:
        return self.train_dataset.vocab_size

    @property
    def block_size(self) -> int:
        return self.train_dataset.block_size


if __name__ == "__main__":
    target_size = 8  # todo: probably expose this eventually

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--n_layer", default=4, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open("input.txt").read()  # don't worry we won't run out of file handles

    dm = CharDataModule(text, args.batch_size, args.block_size)
    model = GPT(
        vocab_size=dm.train_dataset.vocab_size,
        block_size=dm.train_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
    )

    final_tokens = 20 * (len(dm.train_dataset) // target_size) * args.block_size

    lr_decay = LearningRateDecayCallback(
        learning_rate=args.learning_rate,  ##6e-4 * target_size,
        warmup_tokens=375e6,  # 512 * 20,
        final_tokens=final_tokens,
    )


    class TimeCallback(Callback):
        def on_train_start(
                self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
        ) -> None:
            self.start = time.time()

        def on_train_end(
                self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
        ) -> None:
            rank_zero_info(f"Time till convergence: {(time.time() - self.start):.2f}")


    trainer = pl.Trainer(
        gpus=target_size,
        log_every_n_steps=1,
        strategy='ddp',
        max_epochs=10,
        precision=16,
        gradient_clip_val=1,
        accumulate_grad_batches=4,
        callbacks=[
            lr_decay,
            TimeCallback(),
        ],
    )

    trainer.fit(model, datamodule=dm)
