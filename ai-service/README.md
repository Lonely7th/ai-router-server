# JJ Office AI Service

JJ Office 的独立 Python AI 网关。它把桌面端的 OpenAI Chat Completions 请求安全地转发到 DeepSeek，并在服务端统一处理 API Key、身份、限流、并发、用量和监控。

桌面端仓库位于 `D:\workspace\JJ-Office`，本服务位于独立的 `D:\workspace\JJ-Office-Server\ai-service`，两端没有共享源码或构建产物。

## 已实现

- `POST /v1/chat/completions`：非流式及 SSE 流式响应
- `GET /v1/models`：模型白名单
- `GET /health/live`、`GET /health/ready`：容器健康检查
- `/metrics`：Prometheus 指标
- DeepSeek API Key 只保存在服务端
- 开发期免鉴权、静态令牌鉴权、生产 JWT/RS256 鉴权
- 单用户每分钟限流和并发控制；可选 Redis 以支持多进程/多实例
- 可选的 CloudBase 额度预留、结算和释放接口
- 请求体、输出 Token、模型白名单和客户端参数约束
- 上游短暂故障重试，且不记录文档正文或对话内容

## 本地启动（PowerShell）

需要 Python 3.12：

```powershell
cd D:\workspace\JJ-Office-Server\ai-service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
AI_AUTH_MODE=static
AI_STATIC_TOKENS=请替换为足够长的随机令牌
DEEPSEEK_API_KEY=请填写新的DeepSeek密钥
```

启动：

```powershell
uvicorn jj_office_ai.main:app --host 127.0.0.1 --port 8000 --reload
```

访问 `http://127.0.0.1:8000/docs` 可查看开发环境接口文档。已有测试 Key 曾出现在前端代码和对话中，因此部署服务端时应先在 DeepSeek 控制台废止它并生成新 Key。

## 调用示例

桌面端未来只保存本服务签发的短期用户访问令牌，不保存 DeepSeek Key。现阶段静态鉴权测试示例：

```powershell
$headers = @{
  Authorization = "Bearer 请替换为AI_STATIC_TOKENS中的令牌"
  "X-User-Id" = "local-test-user"
}
$body = @{
  model = "deepseek-flash"
  messages = @(@{ role = "user"; content = "把这段文字改得更简洁" })
  stream = $false
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/chat/completions" -Headers $headers -ContentType "application/json" -Body $body
```

GenOffice 当前客户端会在基础地址后追加 `/chat/completions`，因此接入时将基础地址设置为 `https://你的域名/v1`。

## 鉴权模式

| 模式 | 用途 | 请求要求 |
|---|---|---|
| `none` | 仅限本机开发 | 可选 `X-User-Id` |
| `static` | 第一阶段联调 | `Authorization: Bearer ...` 和 `X-User-Id` |
| `jwt` | 正式用户系统 | CloudBase 使用私钥签发 RS256 JWT，本服务用公钥验证 |

生产环境只允许 `AI_AUTH_MODE=jwt`。JWT 必须包含 `sub`、`iat`、`exp`，并匹配配置的 `iss` 和 `aud`。`static` 只是联调手段，因为桌面程序中的共享令牌最终都可以被提取。

## CloudBase 计费接口约定

设置 `AI_QUOTA_SERVICE_URL` 和 `AI_QUOTA_SERVICE_TOKEN` 后，每次模型调用会使用以下内部接口：

1. `POST /v1/internal/ai/reservations` 预留最大输出额度，成功返回 `{"reservation_id":"..."}`。
2. `POST /v1/internal/ai/reservations/{id}/finalize` 上报 DeepSeek 的输入、缓存命中、
   缓存未命中和输出 Token，由 CloudBase 按成本加权额度结算。
3. `POST /v1/internal/ai/reservations/{id}/release` 仅在尚未开始上游生成时释放预留额度。

预留采用失败关闭策略：计费服务不可用时不调用模型，防止绕过余额限制。上游已开始但没有拿到最终用量时，仍调用 `finalize` 并携带失败原因；计费服务可按预留上限或产品规则结算，避免用户通过断开流来逃费。结算服务应通过 `request_id` 保证幂等，并定期回收长时间未结算的预留记录。

DeepSeek 返回的 `prompt_cache_hit_tokens` 和 `prompt_cache_miss_tokens` 会原样传给
额度服务；没有缓存明细的兼容模型，其全部输入按缓存未命中计算。

## Docker

```powershell
Copy-Item .env.example .env
# 编辑 .env 后：
docker compose up --build
```

Compose 会同时启动 Redis。单个异步进程即可承载大量等待上游的连接；正式环境建议按容器横向扩容并保留 Redis，而不是在一个容器内堆叠 worker。由 Nginx/Caddy 终止 TLS，关闭 SSE 响应缓冲，并只开放 443；8000 和 Redis 端口不应暴露到公网。

## 质量检查

```powershell
ruff check .
pytest
```

不要提交 `.env`、私钥、DeepSeek Key 或生产访问令牌。
