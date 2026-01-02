# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.3.x   | :white_check_mark: |
| < 2.3   | :x:                |

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
- Run as non-root user (`promptmill`)
- Have health checks enabled
- Use pinned base images for reproducibility

### Model Downloads

Models are downloaded from Hugging Face Hub:
- All downloads use HTTPS
- Models are cached locally after first download
- No authentication required for public models

### Input Validation

The application includes:
- Prompt length limits (10,000 characters)
- Temperature clamping (0.1-2.0)
- Token limits (100-2000)
- GPU layer validation (0-100)

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
3. **Keep Docker images updated** for security patches
4. **Limit network access** to trusted users
5. **Monitor resource usage** to prevent DoS
