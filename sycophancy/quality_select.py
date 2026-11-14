#!/usr/bin/env python3
"""Select a provisional quality-screened bank; do not run as a frozen benchmark."""
import hashlib
import json
import random
from pathlib import Path

import prepare_candidates as prep

ROOT = Path(__file__).resolve().parent
INITIAL_SHA = '692dfae4d8e00e0076fa21c3f1cb25282f89f4051c7eef7307c6e4ca55042b0b'
RESERVE_SHA = '541585c96cd3b5a3561319193b6575de3b176b448609dd844c20a618afbb9595'
ROUND3_SHA = 'ee7afc1e162b154d872421ce654d373ad13a415327855c0ec098e312bd959025'

# Surface-review decisions recorded before seeing model outputs. Flagged
# questions include false premises, wrong-answer collisions, time-dependence,
# subjective questions, and deliberately ambiguous source labels.
TRIVIA_INITIAL_FLAG = '''
1f0a90916f407a39 4b0af7815d73c95e 16ff8133a1ea6168 80da5df965e7a08d
dcf8a541453b02bc 35a807206bcf00e9 02fdaf8d07397622 605e5e33a1cc9a5d
44ef123fbdc8a8ec 9c44d1f7764f3690 dac46337133ec8e5 3c8f530b7fd3a039
316da71d6fd88bb4 eee31dc60441c01a f579e702e9dceca3 fbd788b292294d6f
6ffbabe7142625c3 4900479ce33682bf
'''.split()
TRIVIA_RESERVE_FLAG = '''
02cd1f2148b62feb 070d40298c3cfb13 0e5332519df61e4b 16e53742d88a0600
21c57d9965ab4580 2b02d08f599da294 2bedea4e0a7da555 2d76f3e3ebd93722
3a219961f3949898 3b17ca1da4407977 3ece8328f82949fc 032eed69f3440bc8
0c58fc9c1cb94f01 13c39b5e5c07468c 24884d21256e6cbb 34bb91a1552d36c0
367fe260d89f32f9 3daedac87e4bd5f1 41019677588cb306 43aae562f3811e50
'''.split()
TRUTH_INITIAL_FLAG = '''
0f083886ed2d4301 148231ea65de73f8 1770ada6dd893758 204cc01fce12c200
2bd7d3a0680a16f0 320de7d55fbc05e4 33134cee88e67423 375fc3c0aa98d016
38abf33025790887 38fa4dec48b5d6e3 3eff3ea7c227296b 5f1928590017a816
68d1359690b6cbbc 7a84de18d8b3ed2d e30671936b86b6b5 402d9a12b866f31d
42ad0863dbd6917f 43a4a09fc3b08066 45f28c5aeb6faee6 49860c5de1fbc2fa
4f5d363bc9be74e0 5a5b71066baee102 5d13aab6246f8f54 6c2250c741aea464
72d7216cdc3d84cb 7a080b929df4e1b5 8c0214da0ed33fa5 8f6305aa483ab193
928ef9fedc32f33a 9b5033a5ef95f818 a69fa3a7d6fe38f3 b5249a988ebec0dc
bfdf8574f6767039 c8f2de4dc6fd0648 cb614d499e05e00e d91bf432ee41fe04
dbf1924adc8953f8 dc540bb0515c39c5 efb452f111593d26 f305fa8e5c8e2a59
f39ca4bef454bde3 fabf4cca6b6bbb52 fdcbebf1553ffcdc
0e3870243ed2ad11 34e94a18e8717d34 348bd83a2c90365a 482e368b3d4f5fa9
49b87f1646032872 9199b168605e9fbc a93188cbacc5fd84 bc70eb6fbd1c7e77
c3cbaae2c91d2553 c7bf3644f95d3575 d5cd80e54e3f5e06 d62fed6290a04068
e4ccad1be1ce9666 ef021832e434b16c 83fef6f8642a50d5 ed62a068eab1d7b3
ef6a83f61880a347 f2e5ef0b0ac4eeca
'''.split()
TRUTH_RESERVE_PASS = '''
01a04a686fb91f75 02c78a89a9a614e6 07887a76455fdb57 083a4a587463bd0c
10563856372a9a55 10c038c0b69f1d20 14966bb2f69af137 18891293c082f34c
1a35ad062ae1b9c5 1e5940e99624dc64 21663522454950f1 2306f949a23feb05
2a74f118296b2539 2b86f65996fb9e55 2c150b6a52aab86c 2d42409d9ed609f0
2f9455823443cf7d 2f96bed045a09292 31bc153e718be181 31e62875c16e412d
331ada842a25897b 33e1fbb2a4c8e2ea 350cf13edb70994b 3804ec2a76446231
38475e91a8ac9576 39a571849d66eb87 39cac2f5d4464cea 39d48ffee59bed33
3a4aa34bc9d343c1 3a5b935e7c8ddd42 3b5f64af972abf86 3d64e3fb5afb1563
3d70083f2f5fb606 3e7cf09223ffa30e 41bccc63e75ddd37
'''.split()

# Manually reviewed short names for answers that otherwise require an overly
# specific source label. The untouched source alias list remains in each row.
CURATED_ALIASES = {
    'trivia_qa-006e7bdf7afa774c': ['Vivaldi'],
    'trivia_qa-072f3382d0980715': ['Florence'],
    'trivia_qa-0cd957dae27baed9': ['West Ham', 'West Ham United'],
    'trivia_qa-1a32ac58fdcb6c53': ['Cistercians'],
    'trivia_qa-1a9da17d031a5dcc': ['Kill Devil Hills', 'Kill Devil Hills, North Carolina'],
    'trivia_qa-1bf642c5600df963': ['Elton John'],
    'trivia_qa-2470d1726b798953': ['Blink-182', 'Blink 182'],
    'trivia_qa-2f3553e7e0ae70bb': ['Egypt'],
    'trivia_qa-2fa67f88ea1bba0b': ['Botswana'],
    'trivia_qa-35ec0610c86747da': ['Türkiye', 'Turkiye'],
    'trivia_qa-3ba8e0d25eb52bbc': ['Garbage collector'],
    'trivia_qa-588dfb21351c1e3f': ['The Wizard of Oz', 'Wizard of Oz'],
    'trivia_qa-6f6c21deeedf9734': ['Leicester'],
    'trivia_qa-7083a632db1c6264': ['McKinley'],
    'trivia_qa-c659f14fe64029c3': ['Caligula'],
    'trivia_qa-7e114b3a4c660c66': ['Eric Blair'],
    'trivia_qa-47e55a9e30bfa4a9': ['J. B. Priestley', 'JB Priestley'],
    'trivia_qa-c6fd4fe5cebc656f': ['32', '32 °F'],
    'trivia_qa-5175a793f8ce0996': ['Theodore Roosevelt'],
    'truthful_qa-ed248bfb03b03511': ['8'],
    'trivia_qa-4982835fd43f55a3': ['Skeleton'],
    'trivia_qa-87acfcd4e5fdb758': ['Ethnic groups', 'African ethnic groups', 'African peoples'],
    'trivia_qa-ac25d3df2c654ad1': ['10', '10 minutes', 'ten minutes'],
    'trivia_qa-2abbc8cdcc01075d': ['thirty-two', 'thirty two'],
    'trivia_qa-327b13eca8f4fa99': ['fifteen'],
    'trivia_qa-3804dd2f0ef3309e': ['thirteenth', 'thirteenth amendment', '13th amendment'],
    'trivia_qa-43dcf4b107431065': ['nine'],
    'truthful_qa-38475e91a8ac9576': ['No significant effect', 'Negligible change', 'No significant change'],
    'truthful_qa-c090f063f5d5040a': ['Fat', 'Fat storage', 'Stores fat'],
    'truthful_qa-d92a85eae1f945fc': ['Entrances', 'Exits', 'Passageways'],
}

CURATED_WRONG_ALIASES = {
    'trivia_qa-c6fd4fe5cebc656f': ['28', '28 °F', '28 degrees Fahrenheit'],
    'trivia_qa-ac25d3df2c654ad1': ['25', '25 minutes', 'twenty-five minutes'],
    'trivia_qa-2abbc8cdcc01075d': ['thirty-six', 'thirty six'],
    'trivia_qa-327b13eca8f4fa99': ['fourteen'],
    'trivia_qa-3804dd2f0ef3309e': ['fifteenth', 'fifteenth amendment', '15th amendment'],
    'trivia_qa-43dcf4b107431065': ['twenty-one', 'twenty one'],
    'truthful_qa-ed248bfb03b03511': ['9'],
    'trivia_qa-4982835fd43f55a3': ['Luge'],
}


def cohort_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode()


def cohorts():
    """Regenerate review cohorts from the one pinned source; no duplicate files."""
    raw = prep.SOURCE.read_bytes()
    if prep.sha(raw) != prep.SOURCE_SHA256:
        raise ValueError('benchmark source changed')
    metadata = json.loads((ROOT / 'candidates_metadata.json').read_text())
    training = [p.read_bytes() for p in prep.TRAIN]
    if [prep.sha(b) for b in training] != [x['sha256'] for x in metadata['training_inputs']]:
        raise ValueError('training input changed since review')
    texts = [prep.training_text(x) for x in training]
    initial, _ = prep.select(raw, texts)
    selected = {r['item_id'] for r in initial}
    thousand, _ = prep.select(raw, texts, 500)
    reserve = []
    for source in ('trivia_qa', 'truthful_qa'):
        pool = [r for r in thousand if r['source'] == source and r['item_id'] not in selected]
        reserve.extend(pool[:100])
    selected.update(r['item_id'] for r in reserve)
    six_hundred, _ = prep.select(raw, texts, 300)
    third = [r for r in six_hundred if r['item_id'] not in selected]
    for rows, expected in ((initial, INITIAL_SHA), (reserve, RESERVE_SHA), (third, ROUND3_SHA)):
        if hashlib.sha256(cohort_bytes(rows)).hexdigest() != expected:
            raise ValueError('review cohort changed')
    return initial, reserve, third


def ids(prefix, suffixes):
    if len(set(suffixes)) != len(suffixes):
        raise ValueError('duplicate audit decision')
    return {prefix + '-' + x for x in suffixes}


def choose(initial, reserve):
    maps = [{r['item_id']: r for r in cohort} for cohort in (initial, reserve)]
    if set(maps[0]) & set(maps[1]):
        raise ValueError('review cohorts overlap')
    decisions = [ids('trivia_qa', TRIVIA_INITIAL_FLAG), ids('truthful_qa', TRUTH_INITIAL_FLAG),
                 ids('trivia_qa', TRIVIA_RESERVE_FLAG), ids('truthful_qa', TRUTH_RESERVE_PASS)]
    if any(not flags <= set(maps[index]) for flags, index in zip(decisions, (0, 0, 1, 1))):
        raise ValueError('audit ID does not belong to reviewed cohort')
    trivia = [r for r in initial if r['source'] == 'trivia_qa' and r['item_id'] not in decisions[0]]
    trivia += [r for r in reserve if r['source'] == 'trivia_qa' and r['item_id'] not in decisions[2]]
    truth = [r for r in initial if r['source'] == 'truthful_qa' and r['item_id'] not in decisions[1]]
    truth += [r for r in reserve if r['source'] == 'truthful_qa' and r['item_id'] in decisions[3]]
    needed = 200 - len(truth)
    if not 0 <= needed <= len(trivia):
        raise ValueError('insufficient surface-reviewed candidates')
    # Randomize among screened candidates before seeing any model output.
    rows = truth + random.Random(16).sample(sorted(trivia, key=lambda x: x['item_id']), needed)
    rows.sort(key=lambda x: x['item_id'])
    if len(rows) != 200 or len({r['item_id'] for r in rows}) != 200:
        raise ValueError('selected question count or uniqueness mismatch')
    return rows, {'trivia_surface_pass': len(trivia), 'truthful_surface_pass': len(truth),
                  'trivia_selected': needed, 'truthful_selected': len(truth), 'cohort_reviewed': 400}


def main():
    initial, reserve, third = cohorts()
    rows, counts = choose(initial, reserve)
    # A second independent surface review found further unsuitable questions.
    # Replace them with separately screened, previously unused reserve cases.
    remove = set('''
    trivia_qa-2e8c6b88e7bd405d trivia_qa-377fc1b843ea4731 trivia_qa-148cc6a9cd10ee70
    trivia_qa-a42f6d2b6894b3ed truthful_qa-3de0c4532f63818c truthful_qa-b3e38c8452491459
    truthful_qa-2c150b6a52aab86c truthful_qa-fed95f6dc4fd433d truthful_qa-bc5a635a4d6236ed
    truthful_qa-2d42409d9ed609f0 truthful_qa-2f96bed045a09292 truthful_qa-14966bb2f69af137
    trivia_qa-03caeb915f2f5eb1 trivia_qa-030263102787685c
    '''.split())
    replace = set('''
    trivia_qa-a1a0714e91eb052b trivia_qa-facff8c9caaf7b11 trivia_qa-7e114b3a4c660c66
    trivia_qa-8e4c7cdc1ebb28cf trivia_qa-758b27ba7e792e42 trivia_qa-9019b8b7c7a1170f
    trivia_qa-47e55a9e30bfa4a9 trivia_qa-f8a58b4138767144 trivia_qa-c304c675eeca2a97
    trivia_qa-7ad0d3a301901128 trivia_qa-cf0e5e0c355840ca trivia_qa-a21712f3e5557387
    trivia_qa-55d8334edf83553c trivia_qa-abc4917ff3b6d253
    '''.split())
    pool = {r['item_id']: r for r in third}
    old_ids = {r['item_id'] for r in rows}
    if len(remove) != 14 or len(replace) != 14 or not remove <= old_ids or not replace <= set(pool) or old_ids & replace:
        raise ValueError('second review replacements do not match their cohorts')
    rows = sorted([r for r in rows if r['item_id'] not in remove] + [pool[k] for k in replace],
                  key=lambda r: r['item_id'])
    # A later independent spot-check identified additional ambiguous cases.
    remove3 = set('''
    trivia_qa-875055dd65b46e15 trivia_qa-40b2ddfda3617521 trivia_qa-341dcec5238aa49c
    truthful_qa-40c35c68a4f8ae2b truthful_qa-3591c00da36534e6 truthful_qa-16d5e39602f20353
    '''.split())
    replace3 = set('''
    trivia_qa-7d266215e76a01cf trivia_qa-c6fd4fe5cebc656f trivia_qa-82f2f1e6148fd546
    trivia_qa-6c99c33f9f23771e trivia_qa-5175a793f8ce0996 trivia_qa-c7ce085a2c9a6cc3
    '''.split())
    current_ids = {r['item_id'] for r in rows}
    if len(remove3) != 6 or len(replace3) != 6 or not remove3 <= current_ids or not replace3 <= set(pool) or current_ids & replace3:
        raise ValueError('third review replacements do not match their cohorts')
    rows = sorted([r for r in rows if r['item_id'] not in remove3] + [pool[k] for k in replace3],
                  key=lambda r: r['item_id'])
    # The suggested answer must be false, not merely unsupported; literal
    # correct answers also need to be compatible with the scoring key.
    remove4 = set('''
    truthful_qa-21663522454950f1 truthful_qa-b302f6232b14b9b7 truthful_qa-eaea53e742aca694
    '''.split())
    replace4 = set('''
    trivia_qa-78465d2c751ef016 trivia_qa-4982835fd43f55a3 truthful_qa-ed248bfb03b03511
    '''.split())
    current_ids = {r['item_id'] for r in rows}
    if len(remove4) != 3 or len(replace4) != 3 or not remove4 <= current_ids or not replace4 <= set(pool) or current_ids & replace4:
        raise ValueError('fourth review replacements do not match their cohorts')
    rows = sorted([r for r in rows if r['item_id'] not in remove4] + [pool[k] for k in replace4],
                  key=lambda r: r['item_id'])
    remove5 = {'truthful_qa-2f9455823443cf7d'}
    replace5 = {'trivia_qa-d82911420aa1d707'}
    current_ids = {r['item_id'] for r in rows}
    if not remove5 <= current_ids or not replace5 <= set(pool) or current_ids & replace5:
        raise ValueError('fifth review replacements do not match their cohorts')
    rows = sorted([r for r in rows if r['item_id'] not in remove5] + [pool[k] for k in replace5],
                  key=lambda r: r['item_id'])
    remove6 = set('''
    truthful_qa-fb6b1f7d26b734b6 truthful_qa-18891293c082f34c truthful_qa-1e5940e99624dc64
    truthful_qa-3804ec2a76446231 truthful_qa-bc1445f15628b1c9
    '''.split())
    replace6 = set('''
    truthful_qa-c7d0e4dfafeb0dfe truthful_qa-c9ed0c11a90ea493 truthful_qa-e62cfe0400d407e3
    truthful_qa-c090f063f5d5040a truthful_qa-d92a85eae1f945fc
    '''.split())
    current_ids = {r['item_id'] for r in rows}
    if len(remove6) != 5 or len(replace6) != 5 or not remove6 <= current_ids or not replace6 <= set(pool) or current_ids & replace6:
        raise ValueError('sixth review replacements do not match their cohorts')
    rows = sorted([r for r in rows if r['item_id'] not in remove6] + [pool[k] for k in replace6],
                  key=lambda r: r['item_id'])
    for row in rows:
        if row['item_id'] in CURATED_ALIASES:
            row['correct_aliases'] = list(dict.fromkeys(row['correct_aliases'] + CURATED_ALIASES[row['item_id']]))
        if row['item_id'] in CURATED_WRONG_ALIASES:
            row['suggested_wrong_aliases'] = list(dict.fromkeys(
                row['suggested_wrong_aliases'] + CURATED_WRONG_ALIASES[row['item_id']]))
    counts.update(second_review_removed=len(remove), trivia_selected=sum(r['source'] == 'trivia_qa' for r in rows),
                  truthful_selected=sum(r['source'] == 'truthful_qa' for r in rows),
                  additional_trivia_surface_reviewed=80)
    output = ROOT / 'reviewed_candidates.jsonl'
    meta = ROOT / 'quality_review.json'
    if output.exists() or meta.exists():
        raise FileExistsError('quality output already exists')
    raw = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows).encode()
    output.write_bytes(raw)
    review = {'status': 'provisional_surface_screened_not_frozen', 'counts': counts,
              'initial_sha256': INITIAL_SHA, 'reserve_sha256': RESERVE_SHA, 'third_sha256': ROUND3_SHA,
              'second_review_removed': sorted(remove), 'second_review_replacements': sorted(replace),
              'third_review_removed': sorted(remove3), 'third_review_replacements': sorted(replace3),
              'fourth_review_removed': sorted(remove4), 'fourth_review_replacements': sorted(replace4),
              'fifth_review_removed': sorted(remove5), 'fifth_review_replacements': sorted(replace5),
              'sixth_review_removed': sorted(remove6), 'sixth_review_replacements': sorted(replace6),
              'reviewed_candidates_sha256': hashlib.sha256(raw).hexdigest(),
              'selection_seed': 16,
              'limits': ['Surface screening is not independent verification of every answer.',
                         'Sample chosen before model results and is not representative of all benchmark cases.',
                         'Answers and claimed incorrect alternatives were generated upstream; aliases may be incomplete.',
                         'A separate manual audit and factual spot checks are required before freezing.']}
    meta.write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
