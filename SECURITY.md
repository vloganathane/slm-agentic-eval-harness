# Security Policy

## Supported Versions

We support the latest version of `slm-eval` on the `main` branch.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Instead, report them privately by emailing:

```
security@example.com
```

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to respond within **48 hours** and provide a fix within **7 days** for critical issues.

## Security Considerations

This harness handles:
- **Model artifacts** (weights, adapters) — store securely, not committed to repo
- **API keys** (OpenAI, Modal, HF, W&B) — use `.env`, never commit `.env` files
- **Evaluation datasets** — may contain PII; keep local or in secure storage
- **Run artifacts** (`runs/` directory) — contains model outputs, may be sensitive

### Best Practices

1. **Never commit `.env`** — already in `.gitignore`
2. **Use environment variables** for secrets
3. **Restrict file permissions** on `runs/` directory
4. **Review dataset contents** before sharing reports
5. **Pin dependency versions** in production deployments

## Dependency Security

We use Dependabot for automated dependency updates. Review PRs regularly.

To audit locally:
```bash
pip install pip-audit
pip-audit
```
