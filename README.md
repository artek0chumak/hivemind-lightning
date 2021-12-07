Repo contains WIP code to try get hivemind to work with Lightning & minGPT.

## Usage

```
pip install -r requirements.txt
```

We use a coordinator server to generate a peer list to send to everyone else (might be removed later).

```bash
python coordinator.py
```

We have a CLI tool to spin up local processes for testing.

```
python run.py --gpu_per_client --num_local_clients 8 mingpt.py
```