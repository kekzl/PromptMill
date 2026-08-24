# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 3.3.x   | :white_check_mark: |
| < 3.3   | :x:                |

## Security Considerations

### Network Binding

By default, PromptMill binds to `127.0.0.1` (localhost only) for security. This prevents unauthorized network access.

**For Docker/network deployment:**
- Set `SERVER_HOST=0.0.0.0` to allow external connections
- **Always use a reverse proxy** (nginx, traefik, caddy) for production
- Configure SSL/HTTPS at the proxy level
- Consider adding authentication at the proxy level

### Container Security

The Docker images:
- Run as non-root user (`promptmill`, UID 1000)
- Have health checks enabled
- Bind `0.0.0.0` inside the container only; the port is reachable solely through
  an explicit `-p` mapping
- Pin the CUDA base image to a patch version; the CPU base tracks
  `python:3.14-slim-trixie`

### HTTP API

`/api/generate` and `/api/generate/stream` are **unauthenticated**. Anyone who can
reach the port can trigger inference and, on first use, a model download. Treat the
port as trusted-network only, or put authentication in the reverse proxy.

### Model Downloads

Models are downloaded from Hugging Face Hub:
- All downloads use HTTPS
- Models are cached locally after first download
- No authentication required for public models

### Input Validation

Both the UI and the REST API validate through the same domain value object:
- Prompt length limit (10,000 characters)
- Temperature range (0.1-2.0)
- Token limit (100-2000)
- Unknown targets and models are rejected with 404 before any model is loaded

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Email the maintainer directly or use GitHub's private vulnerability reporting
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We aim to respond within 48 hours and will work with you to address the issue.

## Security Best Practices

When deploying PromptMill:

1. **Use a reverse proxy** for production deployments
2. **Enable HTTPS** via your reverse proxy
3. **Add authentication at the proxy**: the API has none of its own
4. **Keep Docker images updated** for security patches
5. **Limit network access** to trusted users
6. **Monitor resource usage**: generation is CPU/GPU bound and unthrottled
