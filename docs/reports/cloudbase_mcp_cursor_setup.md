# Cursor CloudBase MCP 配置说明

依据：https://docs.cloudbase.net/ai/cloudbase-ai-toolkit/ide-setup/cursor

## 已完成

1. 项目 `.cursor/mcp.json` 已写入 `cloudbase` MCP（本机全局 `cloudbase-mcp`）。
2. 已安装 CloudBase Skills：`.agents/skills/cloudbase`。
3. Node.js 版本满足要求（当前 v22）。

## 你需要做的（登录必须人工确认）

1. 打开 Cursor：**Settings → MCP**，确认 `cloudbase` 为 Enabled；若未出现，执行 **Developer: Reload Window**。
2. 首次启用时若弹出权限/工具确认，请允许。
3. 回到本对话发送：`登录云开发`  
   AI 会通过 MCP 打开浏览器登录腾讯云，并引导选择环境（用于后续自动管理 `wf_samples` / `wf_phase_labels`）。

## 配置文件

```json
{
  "mcpServers": {
    "cloudbase": {
      "command": "cloudbase-mcp",
      "args": [],
      "env": {
        "INTEGRATION_IDE": "Cursor",
        "npm_config_registry": "https://registry.npmmirror.com"
      }
    }
  }
}
```

若全局命令不可用，可改回官方写法：

```json
"command": "npx",
"args": ["@cloudbase/cloudbase-mcp@latest"]
```

## 与本仓库 P2 的关系

- MCP：适合在对话里创建集合、查库、改安全规则、部署。
- `scripts/push_*.py` / `pull_*.py`：适合批量上传宁夏 h5 + 7182 标签。
- 两者共用同一 CloudBase 环境 ID 即可。
