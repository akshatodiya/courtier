"""Draft planner for exactly eight supervised rows over 14 FSDP ranks.

Every rank must run forward/backward for FSDP collectives. Inactive ranks use
an existing row with a zero contribution to the loss. This deliberately uses
extra compute to preserve the 8B run's global batch size; it is not a trainer.
"""
import random

WORLD_SIZE = 14
GLOBAL_BATCH = 8


def epoch_plan(row_count, seed, epoch):
    if row_count <= 0 or epoch < 0:
        raise ValueError('invalid row count or epoch')
    order = list(range(row_count))
    random.Random(seed + epoch).shuffle(order)
    padding = (-len(order)) % GLOBAL_BATCH
    if padding:
        order.extend(order[i % row_count] for i in range(padding))
    for offset in range(0, len(order), GLOBAL_BATCH):
        rows = order[offset:offset + GLOBAL_BATCH]
        first_rank = (offset // GLOBAL_BATCH * GLOBAL_BATCH) % WORLD_SIZE
        assignments = [{'rank': r, 'row': rows[0], 'active': False} for r in range(WORLD_SIZE)]
        for i, row in enumerate(rows):
            rank = (first_rank + i) % WORLD_SIZE
            assignments[rank] = {'rank': rank, 'row': row, 'active': True}
        yield assignments


def token_weighted_rank_scale(world_size, global_supervised_tokens):
    if world_size <= 0 or global_supervised_tokens <= 0:
        raise ValueError('positive world size and token count required')
    return world_size / global_supervised_tokens
