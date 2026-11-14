"""Story-grounded SOTOPIA scenes for the Aya Expanse 8B courtier data.

Each accepted story from data/stories is transferred into a present-day two-agent scenario (the courtier spec schema),
then the teacher writes one full episode, which is replayed through the frozen SOTOPIA runner so every training chat
is exactly what a local actor would see at evaluation time. Library module, used by story_full.py.

Honesty line: withholding is fine, falsehood is not; bluffing only inside a consensual game;
deceptive story moves become an honest variant or a con-recognition scene whose deceiver role is never trainable.

Prompts: teacher bake-off and prompt pilot (data/story_scenes/bakeoff, luna_pilot) shaped the two kept
prompts. "scene" lets the teacher choose the transfer; "con_scene" forces a target-side con-recognition scene."""
import hashlib, json, re, sys, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
AYA = HERE.parent
W = AYA.parent / 'tiny_aya_3_35b/courtier_run4_revision_20261020'
sys.path.insert(0, str(W / 'sotopia_eval'))
sys.path.insert(0, str(W / 'social_v4'))
import runner as R  # noqa: E402  (frozen prompts, parser and Episode replay only; no transport used)
import courtier_gen as CG  # noqa: E402  (spec schema helpers: to_case, reveal_tokens, NUM, norm)
import teachers as T  # noqa: E402

STORIES = AYA / 'data/stories/stories.jsonl'
STORY_FILES = [STORIES, AYA / 'data/stories/stories_modern.jsonl']
LAWS = AYA.parent / 'tiny_aya_3_35b/courtier/data/laws.json'
OUT = AYA / 'data/story_scenes'
SEED = 'aya-story-scenes-20261023'
CATEGORIES = {
    **CG.CATEGORIES,
    'framing_and_credit': 'who gets credit and how a proposal is framed decide whether others support it',
    'timing_and_leverage': 'when to act, wait or concede matters more than what is offered',
    'influence_without_authority': 'one side must move a decision it does not control, through incentives and allies',
}
SETTINGS = ['a startup board meeting', 'a family-owned restaurant', 'a university department', 'a hospital administration office',
            'a regional newspaper', 'a farmers market', 'a film production', 'a law firm', 'a tech conference',
            'a neighbourhood association', 'a sports team front office', 'a museum acquisitions committee', 'a theatre company',
            'an online marketplace dispute', 'a wedding planning meeting', 'a used-car lot', 'a rental apartment viewing',
            'a school parents committee', 'a trading floor', 'a hotel sales office', 'a construction site office',
            'a city council committee', 'a software agency client call', 'a record label office', 'a nonprofit fundraising meeting',
            'a coworking space', 'a book publisher', 'a dental clinic partnership', 'a logistics depot', 'a wine bar',
            'a gaming studio', 'a property management office', 'a high-school sports booster club', 'a research lab',
            'a fashion boutique', 'a car repair garage', 'a local radio station', 'a bakery supplier negotiation',
            'a conference organising committee', 'an art gallery opening']
PARTNER_STYLES = ['tough and sceptical; concedes only for concrete reasons',
                  'friendly but self-interested; keeps probing for information',
                  'busy and impatient; pushes for a quick decision',
                  'cautious and status-conscious; worried about how things will look to others']
ACTION_TYPES = ('speak', 'non-verbal communication', 'action', 'leave', 'none')
PER_TEACHER = 2  # default concurrent calls per teacher CLI

SPEC_PROMPT = """You are designing a two-person role-play scenario for training a language model in strategic social skill:
reading incentives and hidden goals, protecting private information, timing, framing, negotiation, countermoves,
alliances, reputation and updating on new facts.

Source story (a documented or first-hand account; use it only as inspiration for the strategic move):
Title: {title}
Summary: {summary}
Core move: {move}
Mapped law {law}: {theme}. {desc}

Transfer the story's core strategic move into a NEW, ordinary present-day situation set in: {setting}.
Keep the strategic logic, not the surface details: no armies, rulers, real people, brands or organisations.

Honesty line (hard rule): every trainable agent must be able to fully reach its goal WITHOUT saying anything false.
Withholding, declining to answer, conditional offers, framing, timing, choosing what to emphasise and letting others draw
their own conclusions are fine. Lies, forgeries, impersonation, bribes and staged events are not.
If the story's move depends on deception, choose exactly one:
  honest_variant: an honest version with a similar strategic effect (real preparation instead of a faked one, a true but
    selectively framed fact instead of a false one); or
  con_recognition: agent 0 (trainable) is the TARGET of an approach modelled on the story and must probe, verify and
    protect itself; agent 1 is the would-be deceiver and is NOT trainable. Keep it at the level of social red flags,
    with no operational fraud detail (no scripts, accounts or payment tricks). Use category con_recognition.
Otherwise use transfer_mode direct.

Other rules:
- Two agents talk directly. Neither knows the other's goal. Each has private facts in extra_info that the other lacks.
- The goals must be in real tension: a skilful agent should do clearly better than a naive one, and the other agent has
  good reasons to resist (not a pushover).
- Give private numbers exactly (e.g. "Your walk-away price is $4,200") and list them in private_numbers; list the exact
  phrases the agent must not reveal in must_not_reveal.
- hint: one sentence of strategy advice reflecting the core move, without naming any law or book.
- New, varied names (not famous); ages 20-70; realistic occupations. Scenario text uses only public facts.
- category: one of {cats}.
- Every number in an agent's private_numbers also appears, written the same way, in that agent's own goal or extra_info.
- Give each agent a limit (walk-away point) and at least one piece of leverage the other side does not know, so
  concessions have to be traded rather than handed over.

Return ONLY one JSON object with exactly these keys:
{{"codename": "snake_case_unique", "category": "...", "setting": "...", "deception_in_story": true,
  "transfer_mode": "direct | honest_variant | con_recognition",
  "strategic_core": "one sentence: the transferable move in general terms",
  "scenario": "1-3 sentences, public facts only, refer to the agents by name",
  "agents": [{{"name": "First Last", "background": "<Name> is a <age>-year-old <female/male/nonbinary> <occupation>. <She/her|He/him|They/them> pronouns. <one hobby sentence>. Personality and values description: <one or two sentences>.", "secret": "a personal secret, one sentence"}},
             {{...same for agent 1...}}],
  "goals": [{{"goal": "one sentence, second person", "extra_info": "private facts for this agent only, second person", "hint": "one sentence", "private_numbers": ["..."], "must_not_reveal": ["..."], "trainable": true}},
            {{...same for agent 1...}}],
  "honesty_note": "one sentence: how each trainable agent can win without lying"}}"""

# The same spec prompt with one extra rule that forces a target-side con-recognition scene.
CON_SPEC_PROMPT = """You are designing a two-person role-play scenario for training a language model in strategic social skill:
reading incentives and hidden goals, protecting private information, timing, framing, negotiation, countermoves,
alliances, reputation and updating on new facts.

Source story (a documented or first-hand account; use it only as inspiration for the strategic move):
Title: {title}
Summary: {summary}
Core move: {move}
Mapped law {law}: {theme}. {desc}

Transfer the story's core strategic move into a NEW, ordinary present-day situation set in: {setting}.
Keep the strategic logic, not the surface details: no armies, rulers, real people, brands or organisations.

Honesty line (hard rule): every trainable agent must be able to fully reach its goal WITHOUT saying anything false.
Withholding, declining to answer, conditional offers, framing, timing, choosing what to emphasise and letting others draw
their own conclusions are fine. Lies, forgeries, impersonation, bribes and staged events are not.
If the story's move depends on deception, choose exactly one:
  honest_variant: an honest version with a similar strategic effect (real preparation instead of a faked one, a true but
    selectively framed fact instead of a false one); or
  con_recognition: agent 0 (trainable) is the TARGET of an approach modelled on the story and must probe, verify and
    protect itself; agent 1 is the would-be deceiver and is NOT trainable. Keep it at the level of social red flags,
    with no operational fraud detail (no scripts, accounts or payment tricks). Use category con_recognition.
Otherwise use transfer_mode direct.

Other rules:
- Two agents talk directly. Neither knows the other's goal. Each has private facts in extra_info that the other lacks.
- The goals must be in real tension: a skilful agent should do clearly better than a naive one, and the other agent has
  good reasons to resist (not a pushover).
- Give private numbers exactly (e.g. "Your walk-away price is $4,200") and list them in private_numbers; list the exact
  phrases the agent must not reveal in must_not_reveal.
- hint: one sentence of strategy advice reflecting the core move, without naming any law or book.
- New, varied names (not famous); ages 20-70; realistic occupations. Scenario text uses only public facts.
- category: one of {cats}.
- Every number in an agent's private_numbers also appears, written the same way, in that agent's own goal or extra_info.
- Give each agent a limit (walk-away point) and at least one piece of leverage the other side does not know, so
  concessions have to be traded rather than handed over.
- THIS scenario must use transfer_mode con_recognition and category con_recognition: agent 0 (trainable) is the target
  of an approach modelled on the story's move and must probe, verify and protect itself without accusing unfairly;
  agent 1 is the would-be deceiver and is not trainable. The target can win by asking for verification, slowing the
  decision, checking independently, setting safe conditions or declining. Keep it to social red flags, with no
  operational fraud detail.

Return ONLY one JSON object with exactly these keys:
{{"codename": "snake_case_unique", "category": "...", "setting": "...", "deception_in_story": true,
  "transfer_mode": "direct | honest_variant | con_recognition",
  "strategic_core": "one sentence: the transferable move in general terms",
  "scenario": "1-3 sentences, public facts only, refer to the agents by name",
  "agents": [{{"name": "First Last", "background": "<Name> is a <age>-year-old <female/male/nonbinary> <occupation>. <She/her|He/him|They/them> pronouns. <one hobby sentence>. Personality and values description: <one or two sentences>.", "secret": "a personal secret, one sentence"}},
             {{...same for agent 1...}}],
  "goals": [{{"goal": "one sentence, second person", "extra_info": "private facts for this agent only, second person", "hint": "one sentence", "private_numbers": ["..."], "must_not_reveal": ["..."], "trainable": true}},
            {{...same for agent 1...}}],
  "honesty_note": "one sentence: how each trainable agent can win without lying"}}"""

EPISODE_PROMPT = """Write one realistic conversation between {n0} and {n1} for the scenario below. It will train a model to play
the trainable role(s): {trainable}.

=== What {n0} can see ===
{obs0}
=== What {n1} can see ===
{obs1}
=== End ===

Strategic core the skilful trainable agent should use naturally (never name it): {core}
Honesty note: {honesty}
Personality of {other}: {style}

Rules:
- Strictly alternate, starting with {n0}: agent 0, 1, 0, 1, ... Between 8 and 12 actions in total.
- Each agent uses ONLY what it can see above plus what has been said aloud. Nobody knows the other's goal, extra
  information, secret or numbers unless it was said in the conversation.
- Trainable agents never state anything false. They may withhold, decline, set conditions, frame and choose timing.
- Trainable agents never reveal: {mnr}.
- Every turn moves things forward: new information, a question, a concession, a condition or a decision. Never restate
  an earlier line or reuse its phrasing; vary sentence openings.
- argument: what is said or done, 15-70 words of natural spoken English.
- action_type is "speak" for most turns; "non-verbal communication" or "action" only when it fits; "leave" only as the
  final action, once the matter is settled or clearly cannot be, with argument "".
- The outcome is realistic: the skilful agent earns its result, and the other agent does not simply give in.
- When negotiating, the skilful trainable agent opens above its limit, never names its limit or calls a number its
  minimum, never concedes its limit in one step, and trades each concession for something concrete.
- The other agent pushes back at least twice with concrete reasons before agreeing, and walks away if the terms stay
  poor for it.
- Trainable agents never claim knowledge they lack, never present a conditional commitment as settled, never invent
  personal history and never assert the other side's private situation as fact.
- Confirm the final agreement once, briefly; do not restate earlier terms turn after turn.
- Each trainable turn makes a point it has not made before; never repeat your own earlier argument in new words.
- Do not settle exactly at your own limit while the other side still shows room to move; test for a better term first.

Return ONLY a JSON object: {{"actions": [{{"agent": 0, "action_type": "speak", "argument": "..."}}, ...]}}"""

PROMPTS = {'scene': (SPEC_PROMPT, EPISODE_PROMPT), 'con_scene': (CON_SPEC_PROMPT, EPISODE_PROMPT)}
LOCKS = {t: threading.Semaphore(PER_TEACHER) for t in T.TEACHERS}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pick(options, key):
    return options[int(digest(f'{SEED}:{key}'), 16) % len(options)]


def load_stories():
    out = {}
    for f in STORY_FILES:
        if f.exists():
            for r in map(json.loads, filter(str.strip, f.read_text(encoding='utf-8').splitlines())):
                assert r['id'] not in out, f"duplicate story id {r['id']}"
                out[r['id']] = r
    return out


def laws():
    return {x['law']: x for x in json.loads(LAWS.read_text())}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


# ---------- validation ----------

def check_spec(spec):
    """Same checks as the v4 courtier generator, plus the story-transfer fields. Raises AssertionError."""
    assert isinstance(spec, dict), 'not an object'
    assert spec.get('category') in CATEGORIES, f"category {spec.get('category')!r}"
    assert spec.get('transfer_mode') in ('direct', 'honest_variant', 'con_recognition'), 'transfer_mode'
    assert isinstance(spec.get('deception_in_story'), bool), 'deception_in_story'
    assert isinstance(spec.get('strategic_core'), str) and spec['strategic_core'].strip(), 'strategic_core'
    assert isinstance(spec.get('codename'), str) and re.fullmatch(r'[a-z0-9_]+', spec['codename']), 'codename'
    a, g = spec.get('agents'), spec.get('goals')
    assert isinstance(a, list) and len(a) == 2 and isinstance(g, list) and len(g) == 2, 'two agents/goals'
    names = [x['name'].strip() for x in a]
    assert len(set(names)) == 2 and all(len(n.split()) >= 2 for n in names), 'names'
    for x in a:
        assert re.match(re.escape(x['name'].strip()) + r' is an? ', x['background']) and 'Personality and values description:' in x['background'], 'background template'
        assert x['secret'].strip(), 'secret'
    for y in g:
        assert all(isinstance(y.get(k), str) and y[k].strip() for k in ('goal', 'extra_info', 'hint')), 'goal text'
        assert isinstance(y.get('trainable'), bool) and isinstance(y.get('private_numbers'), list) and isinstance(y.get('must_not_reveal'), list), 'goal lists'
    if spec['transfer_mode'] == 'con_recognition' or spec['category'] == 'con_recognition':
        assert spec['category'] == 'con_recognition' and spec['transfer_mode'] == 'con_recognition', 'con mode/category mismatch'
        assert g[0]['trainable'] is True and g[1]['trainable'] is False, 'con roles'
    assert any(y['trainable'] for y in g), 'no trainable role'
    for y in g:
        own = CG.norm(' '.join([y['extra_info'], y['goal'], y['hint'], spec['scenario']]))
        for entry in y['private_numbers']:
            for n in CG.NUM.findall(str(entry)):
                m = n.replace(',', '')
                spelled = '.' not in m and int(m) < 1_000_000 and re.search(r'\b' + CG.words(int(m)) + r'\b', own)
                assert re.search(r'(?<![\d.])' + re.escape(m) + r'(?![\d])', own) or spelled, f'private number {m} not in own info'
    spec['setting'] = str(spec.get('setting', ''))
    return spec


def check_episode(obj):
    acts = obj.get('actions') if isinstance(obj, dict) else None
    assert isinstance(acts, list) and 6 <= len(acts) <= R.CONFIG['max_actions'], f'action count {len(acts) if isinstance(acts, list) else None}'
    for i, a in enumerate(acts):
        assert isinstance(a, dict) and a.get('agent') == i % 2, f'turn {i} agent {a.get("agent") if isinstance(a, dict) else a}'
        kind = str(a.get('action_type', '')).strip().lower()
        assert kind in ACTION_TYPES, f'turn {i} action_type {kind!r}'
        arg = a.get('argument') or ''
        assert isinstance(arg, str), f'turn {i} argument'
        if kind in ('speak', 'non-verbal communication', 'action'):
            assert arg.strip(), f'turn {i} empty argument'
            assert len(arg.split()) <= R.CONFIG['argument_word_instruction'], f'turn {i} {len(arg.split())} words'
        assert kind != 'leave' or i == len(acts) - 1, f'turn {i} leave before the end'
    return acts


def case_for(spec, story_id, teacher):
    case = CG.to_case(spec, 1, 'story_scene', 0)
    case.update(id=f'story-{story_id}-{teacher}', environment_id=f"story-{spec['codename']}", origin=f'story_scene:{teacher}',
                split='story_scene', story_id=story_id, transfer_mode=spec['transfer_mode'], strategic_core=spec['strategic_core'])
    return case


def replay(case, acts, teacher):
    """Feed the written actions through the frozen runner Episode; returns it (final chats per role)."""
    ep = R.Episode(case, case['focal_role'])
    for a in acts:
        chat = ep.chat_for_next()
        raw = json.dumps({'action_type': a['action_type'].strip().lower(), 'argument': (a.get('argument') or '').strip(), 'to': []}, ensure_ascii=False)
        ep.commit(chat, raw, teacher, {})
        if ep.status != 'running':
            break
    return ep


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def role_flags(case, ep, role):
    own = [a['action']['argument'] for a in ep.actions if a['role'] == role and a['action']['action_type'] != 'none']
    ws = [words(t) for t in own]
    toks = CG.reveal_tokens(case['must_not_reveal'][role])
    said = ' '.join(own).lower().replace(',', '')
    return {'turns': len(own), 'words': sum(map(len, ws)),
            'self_repeat': sum(overlap(ws[i], ws[i + 1]) > 0.5 for i in range(len(ws) - 1)),
            'leaks': [t for t in toks if t in said]}


def overlap(a, b, n=4):  # same definition as social_v4/report_v4.py
    ga = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    gb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(ga & gb) / max(1, min(len(ga), len(gb))) if ga and gb else 0.0


# ---------- generation ----------

def ask(teacher, prompt, check, path):
    """Teacher call with one corrective retry; every raw answer is kept in the record."""
    rec = json.loads(path.read_text()) if path.exists() else None
    if rec and rec.get('ok'):
        return rec
    if rec and not any('raw' in c for c in rec['calls']):
        rec = None  # transport failures only (quota, timeout): no content attempt was spent, so call again
    if rec:  # never re-call a finished record; only re-check its saved raw answers under the current validator
        for call in rec['calls']:
            if 'raw' not in call:
                continue
            try:
                rec['parsed'] = check(T.first_json(call['raw']))
                rec.update(ok=True, revalidated=True)
                write_json(path, rec)
                break
            except (AssertionError, KeyError, TypeError, AttributeError) as error:
                call['invalid_now'] = str(error)[:200]
        return rec
    rec = {'prompt_sha256': digest(prompt), 'calls': []}
    message = prompt
    for attempt in range(2):
        with LOCKS[teacher]:
            try:
                text, meta = T.teacher(teacher, message)
            except RuntimeError as error:
                rec['calls'].append({'transport_error': str(error)[:400]})
                break
        rec['calls'].append({'raw': text, **meta})
        try:
            rec['parsed'] = check(T.first_json(text))
            rec['ok'] = True
            break
        except (AssertionError, KeyError, TypeError, AttributeError) as error:
            rec['calls'][-1]['invalid'] = str(error)[:200]
            message = prompt + f'\n\nYour previous answer was invalid ({str(error)[:150]}). Return only the corrected JSON.'
    rec.setdefault('ok', False)
    write_json(path, rec)
    return rec


def run_pair(base, teacher, story, law, prompt_name='scene'):
    spec_prompt, episode_prompt = PROMPTS[prompt_name]
    setting = pick(SETTINGS, story['id'])
    prompt = spec_prompt.format(title=story['title'], summary=story['summary'], move=story['move'], law=law['law'],
                                theme=law['theme'], desc=law['desc'], setting=setting, cats=', '.join(CATEGORIES))
    spec = ask(teacher, prompt, check_spec, base / 'specs' / teacher / f"{story['id']}.json")
    print(json.dumps({'teacher': teacher, 'story': story['id'], 'stage': 'spec', 'ok': spec['ok']}), flush=True)
    if not spec['ok']:
        return
    case = case_for(spec['parsed'], story['id'], teacher)
    names = case['agent_names']
    trainable = case['trainable_roles']
    other = 1 if trainable == [0] else 0 if trainable == [1] else 1
    prompt = episode_prompt.format(n0=names[0], n1=names[1], trainable=' and '.join(names[r] for r in trainable),
                                   obs0=case['role_initial_observations'][0].strip(), obs1=case['role_initial_observations'][1].strip(),
                                   core=case['strategic_core'], honesty=case.get('honesty_note') or '', other=names[other],
                                   style=pick(PARTNER_STYLES, story['id'] + ':style'),
                                   mnr='; '.join(f"{names[r]}: {', '.join(map(str, case['must_not_reveal'][r])) or 'nothing listed'}" for r in trainable))
    path = base / 'episodes' / teacher / f"{story['id']}.json"
    ep_rec = ask(teacher, prompt, check_episode, path)
    if ep_rec['ok'] and 'replay' not in ep_rec:
        ep = replay(case, ep_rec['parsed'], teacher)
        ep_rec['case'] = case
        ep_rec['replay'] = {'status': ep.status, 'end_reason': ep.end_reason, 'actions': ep.actions,
                            'final_chats': ep.chats, 'flags': {r: role_flags(case, ep, r) for r in trainable}}
        write_json(path, ep_rec)
    print(json.dumps({'teacher': teacher, 'story': story['id'], 'stage': 'episode', 'ok': ep_rec['ok']}), flush=True)
