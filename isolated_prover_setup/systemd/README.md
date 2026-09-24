# Keeping the hand-deployed compose projects up

The projects here run outside Kubernetes, so nothing is watching them. These
three template units give them a control plane: started at boot, stopped
gracefully at shutdown, repaired if they drift. Installed by hand on the node.

## What each layer covers

| Layer | Covers | Does not cover |
|---|---|---|
| `restart: unless-stopped` in each compose file | the process crashing; the Docker daemon restarting | a container someone removed; a compose file edited but never applied |
| `infver-compose@<project>.service` | bringing the project up at boot; graceful ordered shutdown | drift between boots |
| `infver-compose-reassert@<project>.timer` | drift, every 10 minutes | nothing else — it is the backstop |

They overlap on purpose: a restart policy only restarts containers that
already exist, and neither it nor the boot unit notices a project whose file
changed or whose container was removed by hand.

The re-assert runs `docker compose up -d`, never `restart`. A project already
matching its file is a no-op, so this never interrupts an inference or resets
the health check's cadence.

## Installing

Projects live one directory each under `/opt/infver`. That path is the units'
only assumption about the node; getting it wrong fails with `200/CHDIR` and no
hint as to which directory was wanted, so if you keep them elsewhere say so
once rather than editing the units:

```sh
echo 'INFVER_COMPOSE_ROOT=/srv/infver' | sudo tee /etc/default/infver-compose
```

```sh
sudo mkdir -p /opt/infver
sudo cp -r isolated_prover_setup/*/ /opt/infver/   # each needs its own .env
sudo install -m 0644 isolated_prover_setup/systemd/infver-compose*.{service,timer} \
                     /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable docker        # or the compose restart policies never fire

sudo systemctl enable --now infver-compose@tapped-link-health
sudo systemctl enable --now infver-compose@gpt-oss-120b
sudo systemctl enable --now infver-compose-reassert@tapped-link-health.timer
sudo systemctl enable --now infver-compose-reassert@gpt-oss-120b.timer
```

`@` takes the directory name under `/opt/infver`, so a third project needs no
new unit files — only two `systemctl enable` lines.

## Checking and testing

```sh
systemctl status infver-compose@tapped-link-health
systemctl list-timers 'infver-compose-reassert@*'
journalctl -u infver-compose-reassert@tapped-link-health.service --since -1h
```

A re-assert that repaired something logs what it recreated. Repeated repairs
of the same project mean something is killing it, and the container's own logs
are where that shows.

```sh
docker rm -f tapped-link-health                     # simulate the drift
systemctl start infver-compose-reassert@tapped-link-health
docker ps | grep tapped-link-health                 # back
```

Then reboot the node once and confirm both projects return without anyone
logging in. A recovery path nobody has exercised is not a recovery path — and
for the health check specifically, a sender that quietly fails to come back
reports as a dead *link*, which is the one wrong answer this whole scheme
exists to avoid.
