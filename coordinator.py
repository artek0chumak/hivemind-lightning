import ipaddress

import hivemind
import uvicorn
from fastapi import FastAPI

running_trainings = {}

app = FastAPI()


@app.get("/get-dht/{training_id}")
async def get_dht(training_id: str):
    # not thread-safe but we are not running threads here
    dht = running_trainings.get(training_id)
    if dht is None:
        dht = hivemind.DHT(
            start=True,
            host_maddrs=["/ip4/0.0.0.0/tcp/0"],
        )
        running_trainings[training_id] = dht

    visible_peers = [str(a) for a in dht.get_visible_maddrs() if not ipaddress.ip_address(a.values()[0]).is_loopback]
    print("\n".join(visible_peers))
    return {"peers": visible_peers}


if __name__ == "__main__":
    uvicorn.run("coordinator:app", host="0.0.0.0", port=1339, log_level="debug")
