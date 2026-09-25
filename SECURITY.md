# Security Policy

## Reporting a vulnerability

Please do not disclose a security vulnerability in a public issue. Use GitHub's private
vulnerability reporting if it is enabled for this repository. If it is not enabled, contact
the repository maintainers privately through GitHub before opening a public issue.

Include the affected version or commit, reproduction steps, impact, and a minimal proof of
concept. Remove tokens, personal recordings, transcripts, and other sensitive data before
sharing any report.

## Scope

Security reports are especially useful for command execution, unsafe path handling, model
or cache loading, credential handling, accidental network transmission, and loss or exposure
of input recordings and transcript artifacts.

This project is local-first, but downloaded dependencies and models have their own security
and privacy policies. Keep them updated and review upstream advisories before deploying the
application in a sensitive environment.
