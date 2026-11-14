"""Subscription-CLI model calls for the Aya story-scene curation. No API keys and no billed endpoints.

Teacher: gpt-6-luna through the Codex CLI, chosen by bake-off (data/story_scenes/bakeoff). Each call is
one fresh non-interactive CLI session in an empty temp dir, so the model sees only the prompt. Haiku 5.5 is the judge
only, reached through the existing Claude CLI shim on 127.0.0.1:8767, never as a teacher (it is also the SOTOPIA
partner and judge)."""
import json, os, re, shutil, subprocess, tempfile, time, urllib.request
from datetime import datetime, timezone

TEACHERS = {'gpt-6-luna': ('codex', 'gpt-6-luna')}  # name -> (backend, model id)
TIMEOUT = 900
JUDGE_MODEL = 'claude-haiku-5-5'
JUDGE_ENDPOINT = 'http://127.0.0.1:8767/v1/chat/completions'


def utc():
    return datetime.now(timezone.utc).isoformat()


def _codex(prompt, model, cwd):
    out = os.path.join(cwd, 'last.txt')
    cmd = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check', '-s', 'read-only', '-C', cwd, '-m', model,
           '-c', 'daybreak=false', '-c', 'model_reasoning_effort="high"', '--color', 'never', '-o', out, '-']
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=TIMEOUT)
    text = open(out, encoding='utf-8').read() if os.path.exists(out) else ''
    if r.returncode or not text.strip():
        raise RuntimeError(f'codex rc={r.returncode}: {r.stderr[-300:]}')
    return text, {}


BACKENDS = {'codex': _codex}


def teacher(name, prompt, attempts=3):
    """One teacher call -> (text, meta). Every failed attempt is recorded in meta['attempts']."""
    backend, model = TEACHERS[name]
    meta = {'teacher': name, 'backend': backend, 'model': model, 'attempts': []}
    for n in range(attempts):
        cwd = tempfile.mkdtemp(prefix=f'teacher_{backend}_')
        start = time.monotonic()
        try:
            text, extra = BACKENDS[backend](prompt, model, cwd)
            meta.update(extra, seconds=round(time.monotonic() - start, 1), utc=utc())
            return text, meta
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            err = repr(error)[:400]
            meta['attempts'].append({'error': err, 'utc': utc()})
            time.sleep(90 * (n + 1) if re.search(r'quota|rate.?limit|exhaust|429|usage limit', err, re.I) else 15 * (n + 1))
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
    raise RuntimeError(f'{name} failed {attempts} attempts: {meta["attempts"][-1]["error"]}')


def judge(prompt, attempts=3):
    """One Haiku 5.5 judge request through the existing shim -> (text, meta)."""
    payload = json.dumps({'model': JUDGE_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0}).encode()
    meta = {'judge': JUDGE_MODEL, 'attempts': []}
    for n in range(attempts):
        start = time.monotonic()
        try:
            req = urllib.request.Request(JUDGE_ENDPOINT, payload, {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3300) as resp:
                env = json.loads(resp.read().decode())
            if env.get('model') != JUDGE_MODEL:
                raise RuntimeError(f"wrong judge model {env.get('model')!r}")
            text = env['choices'][0]['message']['content'] or ''
            meta.update(seconds=round(time.monotonic() - start, 1), utc=utc())
            return text, meta
        except Exception as error:  # transport errors never count as a content verdict
            meta['attempts'].append({'error': repr(error)[:300], 'utc': utc()})
            time.sleep(20 * (n + 1))
    raise RuntimeError(f'judge failed {attempts} attempts')


def first_json(text):
    """First JSON object or array in the text (whole text, then ``` blocks)."""
    decoder = json.JSONDecoder()
    full = (text or '').strip()
    for s in [full] + [b.strip() for b in re.findall(r'```(?:json)?\s*(.*?)```', full, re.S)]:
        for i, ch in enumerate(s):
            if ch in '{[':
                try:
                    return decoder.raw_decode(s[i:])[0]
                except ValueError:
                    continue
    return None
