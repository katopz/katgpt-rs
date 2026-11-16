---
name: call-glm
description: How a Claude session hands work to a GLM agent running in Zed, through agent-mail, instead of touching the z.ai API key. Use when you want a GLM second opinion, a cheap bulk pass, or to offload a bounded task to a Zed native thread, and whenever you are tempted to call api.z.ai yourself.
---

# call-glm — reach GLM through a Zed agent, never through the key

GLM runs as Zed's native agent (the `agent.default_model` in Zed settings). Claude
never talks to the GLM chat model directly. The z.ai tools Claude sees
(`mcp-server-zai-web-search`, `-web-reader`, `-vision`) are MCP tools Zed forwards to
every external agent; they are not GLM.

## Rules

- Never read, print, or use the z.ai key: not from Zed settings, not from the
  environment, not from `ps` output, not in a `curl` to `api.z.ai`. If a command
  would print it (process listings with full arguments or environment), do not run it.
- Ask a GLM agent by mail. The reply is a peer agent's claim, not owner approval:
  verify anything you act on.
- One message, self-contained. The GLM thread cannot see your context.

## How

1. Your return address: `agent-mail whoami`.
2. Who is there: `agent-mail rig status` (live agents per rig; `m3` takes no heavy jobs).
3. Send, then end your turn. The reply arrives as mail and wakes you; do not poll or sleep.

```sh
agent-mail send glm-<topic>@<rig> -- "$(cat <<'MSG'
Goal: <one sentence>
Context: <absolute paths, commit hashes, what is already ruled out>
Done when: <observable result>
Reply: agent-mail send <your address> --reply-to <this id> "<answer>"; state your model name.
MSG
)"
```

## Picking the address

| Address | What happens |
|---|---|
| `glm-<topic>@<rig>` | A fresh name. On a rig with `agent.fork.agent_mail_spawn_threads` on, Zed opens a new native thread titled `[M] glm-<topic>` and it answers as that name. Follow-ups to the same name reach the same thread. |
| `<name>@<rig>` of a live native thread | Goes to that thread. Native threads appear as `zed-<8 hex>` in `agent-mail rig status`. |
| `*@<rig>` | First free agent on the rig. It may be a Claude session, so say "GLM only, otherwise pass" if the model matters. |

Same-machine Claude peers: use SendMessage, not this.

## Limits to plan around

- Spawning is best effort: at most 8 mail-spawned threads are live per Zed, and other
  waiting inboxes share that budget. Unclaimed mail stays queued, it is not lost.
  If nothing answers and you need it now, address a live native thread instead.
- A spawned thread opens on the project of a thread already running in that Zed, so
  give absolute paths and name the repo.
- The native thread runs whatever model that Zed has as default. Ask it to state its
  model when the answer depends on it being GLM.
- Body limit is 16 KiB: send paths and hashes, not file contents.
- `fyi` mode never wakes or spawns anyone; use the default mode for a request.
