"""Codex app-server client: fresh ephemeral, tool-free, environment-free inference.

Uses the installed Codex login, not an API key. Never reads auth material. Only
structured final outputs and usage/provenance events are retained; reasoning
content is neither requested nor stored. App-server schema is locally captured.
"""
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import tomllib

from core import ROOT, canonical, digest

DISABLED = ['apps', 'plugins', 'hooks', 'shell_tool', 'unified_exec', 'multi_agent',
            'browser_use', 'computer_use', 'image_generation', 'in_app_browser',
            'memories', 'goals', 'code_mode', 'code_mode_host', 'code_mode_only',
            'workspace_dependencies', 'tool_suggest', 'shell_snapshot', 'personality',
            'current_time_reminder', 'auth_elicitation', 'default_mode_request_user_input',
            'standalone_web_search', 'multi_agent_v2']


def isolated_config():
    cfg = {'web_search': 'disabled', 'project_doc_max_bytes': 0,
           'skills': {'include_instructions': False, 'bundled': {'enabled': False}},
           'include_apps_instructions': False, 'include_collaboration_mode_instructions': False,
           'include_permissions_instructions': False, 'model_reasoning_summary': 'none',
           'features': {k: False for k in DISABLED},
           'tools': {'update_plan': {'enabled': False}, 'experimental_request_user_input': {'enabled': False}},
           'developer_instructions': '', 'personality': 'none'}
    p = Path.home() / '.codex/config.toml'
    if p.exists():
        # Enumerate server names only, do not expose connection credentials.
        local = tomllib.loads(p.read_text())
        cfg['mcp_servers'] = {k: {'enabled': False} for k in local.get('mcp_servers', {})}
    return cfg


def thread_params(model, effort, service_tier, cwd, system):
    cfg = isolated_config()
    cfg['model_reasoning_effort'] = effort
    return {'model': model, 'allowProviderModelFallback': False,
            'approvalPolicy': 'never', 'sandbox': 'read-only', 'cwd': str(cwd),
            'baseInstructions': system, 'developerInstructions': '',
            'config': cfg, 'environments': [], 'runtimeWorkspaceRoots': [],
            'selectedCapabilityRoots': [], 'dynamicTools': [], 'ephemeral': True,
            'experimentalRawEvents': False, 'serviceTier': service_tier, 'personality': 'none'}


def turn_params(thread_id, prompt, schema, model, effort, service_tier):
    return {'threadId': thread_id, 'input': [{'type': 'text', 'text': prompt, 'text_elements': []}],
            'model': model, 'effort': effort, 'summary': 'none', 'serviceTier': service_tier,
            'environments': [], 'runtimeWorkspaceRoots': [], 'outputSchema': schema,
            'personality': 'none'}


class CodexClient:
    def __init__(self, cwd, timeout=600):
        self.cwd = Path(cwd).resolve()
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        args = ['codex', 'app-server', '--stdio']
        # Apply disabling flags before server startup as well as thread creation.
        for name in DISABLED:
            args += ['-c', f'features.{name}=false']
        cfg = isolated_config()
        for k in cfg.get('mcp_servers', {}):
            if not re.fullmatch(r'[A-Za-z0-9_-]+', k):
                raise ValueError('Unsupported MCP server key; cannot safely disable via CLI')
            args += ['-c', f'mcp_servers.{k}.enabled=false']
        args += ['-c', 'web_search="disabled"', '-c', 'project_doc_max_bytes=0',
                 '-c', 'skills.include_instructions=false', '-c', 'skills.bundled.enabled=false']
        env = {k: v for k, v in os.environ.items() if not k.startswith(('CODEX_THREAD', 'CODEX_TURN', 'CODEX_PARENT'))}
        # No API credential override; normal Codex authentication remains in place.
        self.proc = subprocess.Popen(args, cwd=self.cwd, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.messages, self.errors = queue.Queue(), []
        self.serial = 0
        self.events = []
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._errors, daemon=True).start()
        self.rpc('initialize', {'clientInfo': {'name': 'routerbench_reasoning_probe', 'version': '1.0'},
                                'capabilities': {'experimentalApi': True}})
        self.send({'method': 'initialized'})

    def _reader(self):
        for line in self.proc.stdout:
            try:
                self.messages.put(json.loads(line))
            except json.JSONDecodeError:
                self.errors.append('non-JSON stdout: ' + line[:200])
        self.messages.put({'_eof': True})

    def _errors(self):
        for line in self.proc.stderr:
            self.errors.append(line.rstrip())

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + '\n')
        self.proc.stdin.flush()

    def receive(self, timeout=None):
        try:
            m = self.messages.get(timeout=timeout or self.timeout)
        except queue.Empty:
            raise TimeoutError('Codex event timeout')
        if m.get('_eof'):
            raise RuntimeError('Codex exited: ' + '\n'.join(self.errors[-8:]))
        if 'method' in m and 'id' in m:
            # No tool invocation/approval is ever executed by this client.
            self.send({'id': m['id'], 'error': {'code': -32601, 'message': 'Tools unavailable for routing experiment'}})
            raise RuntimeError('Unexpected server tool/approval request: ' + m['method'])
        return m

    def rpc(self, method, params):
        self.serial += 1
        ident = self.serial
        self.send({'id': ident, 'method': method, 'params': params})
        deadline = time.monotonic() + self.timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Codex RPC timeout: ' + method)
            m = self.receive(max(.1, deadline - time.monotonic()))
            if m.get('id') == ident:
                if 'error' in m:
                    raise RuntimeError(f'{method}: {m["error"]}')
                return m['result']
            self.events.append(m)

    def infer(self, prompt, schema, system, cfg, effort):
        start_args = thread_params(cfg['model'], effort, cfg['service_tier'], self.cwd, system)
        started = time.perf_counter()
        resolved = self.rpc('thread/start', start_args)
        if resolved['model'] != cfg['model'] or resolved.get('reasoningEffort') != effort:
            raise RuntimeError('Model or reasoning effort substituted: ' + canonical(resolved))
        if resolved.get('instructionSources') or resolved.get('runtimeWorkspaceRoots'):
            raise RuntimeError('Unexpected context sources or workspace roots')
        tid = resolved['thread']['id']
        turn_args = turn_params(tid, prompt, schema, cfg['model'], effort, cfg['service_tier'])
        turn = self.rpc('turn/start', turn_args)
        turn_id = turn['turn']['id']
        usage, final, safe_events = None, None, []
        deadline = time.monotonic() + self.timeout
        pending, self.events = self.events, []
        while True:
            m = pending.pop(0) if pending else self.receive(max(.1, deadline - time.monotonic()))
            method, p = m.get('method', ''), m.get('params', {})
            if p.get('threadId') not in (None, tid):
                continue
            if method == 'thread/tokenUsage/updated':
                usage = p['tokenUsage']['last']
                safe_events.append(m)
            if method in ['item/started', 'item/completed']:
                item = p.get('item', {})
                typ = item.get('type')
                if typ not in ('userMessage', 'agentMessage', 'reasoning'):
                    raise RuntimeError('Tool or unexpected inference item: ' + str(typ))
                if method == 'item/completed' and typ == 'agentMessage':
                    final = item.get('text')
                    safe_events.append(m)
            if method == 'error':
                safe_events.append(m)
            if method == 'turn/completed' and p['turn']['id'] == turn_id:
                if p['turn']['status'] != 'completed':
                    raise RuntimeError('Codex turn failed: ' + canonical(p['turn']))
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Codex inference timeout')
        if final is None or usage is None:
            raise RuntimeError('Missing final output or usage telemetry')
        # ThreadStartResponse records resolved client model/effort, not an immutable server snapshot.
        proof = {'requested_model': cfg['model'], 'resolved_model': resolved['model'],
                 'resolved_snapshot': None, 'reasoning_effort': resolved['reasoningEffort'],
                 'reasoning_mode': None, 'reasoning_mode_note': 'Not separately exposed by installed Codex protocol',
                 'service_tier': resolved.get('serviceTier'), 'thread_id': tid, 'turn_id': turn_id,
                 'response_id': None, 'input_tokens': usage['inputTokens'],
                 'cached_input_tokens': usage['cachedInputTokens'], 'output_tokens': usage['outputTokens'],
                 'reasoning_tokens': usage['reasoningOutputTokens'], 'total_tokens': usage['totalTokens'],
                 'latency_seconds': time.perf_counter() - started, 'tool_events': 0,
                 'instruction_sources': resolved.get('instructionSources', []),
                 'environments': [], 'raw_structured_response': final, 'telemetry_events': safe_events,
                 'request': {'thread': start_args, 'turn': {k: v for k, v in turn_args.items() if k != 'threadId'}}}
        return proof

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
