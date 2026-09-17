# JJ Office Server

This directory is intentionally separate from `D:\workspace\JJ-Office`.

- `JJ-Office` contains the Electron desktop client.
- `JJ-Office-Server` contains independently deployable backend services.
- Secrets, Python environments, build output, and deployment files must not be shared with the client repository.

The first backend component is [`ai-service`](./ai-service/README.md), an OpenAI-compatible
DeepSeek gateway written in Python.

