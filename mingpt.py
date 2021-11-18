"""
Minimal example to train an xFormer model using the Tiny Shakespeare dataset
Reference: https://github.com/williamFalcon/minGPT & https://github.com/karpathy/minGPT.
"""
import logging
import math
import os
import time
from argparse import ArgumentParser

import hivemind
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy


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
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

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
        self.blocks = nn.Sequential(*(Block(self.config) for _ in range(self.config.n_layer)))
        self.accuracy = Accuracy()

        rank_zero_info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)
        return optimizer

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
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
        self.log("val_acc", self.accuracy(indices, targets), prog_bar=True, on_epoch=True)


class LearningRateDecayCallback(pl.Callback):
    def __init__(self, learning_rate, warmup_tokens=375e6, final_tokens=260e9, lr_decay=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.tokens = 0
        self.final_tokens = final_tokens
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        optimizer = trainer.optimizers[0]
        _, y = batch

        if self.lr_decay:
            self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
            if self.tokens < self.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.warmup_tokens) / float(
                    max(1, self.final_tokens - self.warmup_tokens)
                )
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

                pl_module.log("lr_scheduler", lr, on_step=True)


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
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

        self.full_dataset = CharDataset(text, block_size)
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(self.full_dataset, [train_size, val_size])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def vocab_size(self) -> int:
        return self.full_dataset.vocab_size

    @property
    def block_size(self) -> int:
        return self.full_dataset.block_size


class HiveMindCallback(Callback):
    def __init__(self, hm_target_group_size: int, dht: hivemind.DHT):
        super().__init__()
        self.dht = dht
        self.target_group_size = hm_target_group_size

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(trainer.optimizers) > 1:
            raise MisconfigurationException("Hivemind only supports training with one optimizer.")
        (optimizer,) = trainer.optimizers
        # Set up a decentralized optimizer that will average with peers in background
        opt = hivemind.optim.DecentralizedOptimizer(
            opt=optimizer,
            dht=self.dht,
            prefix="lightning_run",
            compression=hivemind.Float16Compression(),
            average_parameters=True,
            average_gradients=False,
            client_mode=False,
            verbose=True,
            target_group_size=self.target_group_size
        )
        # opt.averager.load_state_from_peers()
        trainer.optimizers = [WrapperOptimizer(opt)]

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logging.getLogger("pytorch_lightning").info("Shutting down hivemind DHT.")
        self.dht.shutdown()


class WrapperOptimizer:
    def __init__(self, opt: hivemind.DecentralizedOptimizerBase):
        self.__dict__ = {k: v for k, v in opt.__dict__.items() if k not in ("step",)}
        self.__class__ = type("Lightning" + opt.__class__.__name__, (self.__class__, opt.__class__), {})
        self.opt = opt

    def step(self, closure=None):
        if closure:
            closure()
        return self.opt.step()


if __name__ == "__main__":
    target_size = 8  # todo: probably expose this eventually

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--n_layer", default=8, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--learning_rate", default=6e-4, type=float)
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open("input.txt").read()  # don't worry we won't run out of file handles

    dm = CharDataModule(text, args.batch_size, args.block_size)
    model = GPT(
        vocab_size=dm.full_dataset.vocab_size,
        block_size=dm.full_dataset.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
    )

    lr_decay = LearningRateDecayCallback(
        learning_rate=6e-4 * target_size,
        warmup_tokens=512 * 20,
        final_tokens=2 * len(dm.train_dataset) * args.block_size,
    )


    class TimeCallback(Callback):
        def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            self.start = time.time()

        def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            rank_zero_info(f"Time till convergence: {(time.time() - self.start):.2f}")


    initial_peers = os.environ["INITIAL_PEERS"].split(",")

    dht = hivemind.DHT(
        start=True,
        initial_peers=initial_peers,
    )

    trainer = pl.Trainer(
        gpus=1,
        log_every_n_steps=1,
        max_epochs=10,
        precision=16,
        gradient_clip_val=1,
        callbacks=[
            HiveMindCallback(hm_target_group_size=target_size, dht=dht),
            lr_decay,
            TimeCallback(),
        ],
        val_check_interval=0.25,
    )

    trainer.fit(model, datamodule=dm)
