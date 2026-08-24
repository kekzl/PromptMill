#!/usr/bin/env bash
# PromptMill one-command start.
#
#   ./start.sh          auto-detect GPU, start, wait for health, print the URL
#   ./start.sh cpu      force the CPU image
#   ./start.sh gpu      force the GPU image
#   ./start.sh stop     stop and remove the container
#
# Only Docker is required; the images are pulled from GHCR.

set -euo pipefail

cd "$(dirname "$0")"

# PROMPTMILL_PORT may come from the environment or from .env, which compose
# reads on its own; read it here too so the printed URL is right.
PORT="${PROMPTMILL_PORT:-}"
if [ -z "$PORT" ] && [ -f .env ]; then
    PORT="$(sed -n 's/^[[:space:]]*PROMPTMILL_PORT=//p' .env | tail -1)"
fi
PORT="${PORT:-7610}"
URL="http://localhost:${PORT}"

die() { printf 'error: %s\n' "$1" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not installed - see https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || die "docker compose v2 is required (try: docker compose version)"

has_gpu() {
    # The nvidia runtime is what actually matters; nvidia-smi on the host can be
    # present without the container toolkit being wired up.
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia
}

case "${1:-auto}" in
    stop)
        docker compose --profile gpu --profile cpu down
        echo "PromptMill stopped."
        exit 0
        ;;
    gpu|cpu)
        PROFILE="$1"
        ;;
    auto)
        if has_gpu; then
            PROFILE=gpu
            echo "NVIDIA container runtime detected - using the GPU image."
        else
            PROFILE=cpu
            echo "No NVIDIA container runtime - using the CPU image."
        fi
        ;;
    *)
        die "unknown argument '$1' (expected: gpu, cpu, stop, or nothing)"
        ;;
esac

echo "Pulling the ${PROFILE} image (first run only, this can take a few minutes)..."
docker compose --profile "$PROFILE" pull --quiet

docker compose --profile "$PROFILE" up -d

# Wait on Docker's own healthcheck so no extra tooling (curl, wget) is needed.
CONTAINER="promptmill-${PROFILE}"
printf 'Waiting for PromptMill to come up'
for _ in {1..60}; do
    STATE="$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo missing)"
    if [ "$STATE" = "healthy" ]; then
        echo
        echo "PromptMill is running: ${URL}"
        echo "API docs:              ${URL}/docs"
        echo "Stop it with:          ./start.sh stop"
        exit 0
    fi
    printf '.'
    sleep 2
done

echo
echo "PromptMill did not answer on ${URL}/health within 120s."
echo "Container logs:"
docker compose --profile "$PROFILE" logs --tail 40
exit 1
