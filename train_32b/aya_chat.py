"""Aya Expanse chat encoding with assistant-only supervision, built from the model's own chat-template tokens.

The Aya Expanse 8B template (revision 5062468b) renders a conversation as
  <BOS_TOKEN> then, per message, <|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|>
with <|CHATBOT_TOKEN|> for assistant turns. Generation stops at <|END_OF_TURN_TOKEN|> (id 255001).
encode() builds the same token sequence segment by segment, so the supervised span of each assistant turn (its
content plus the closing <|END_OF_TURN_TOKEN|>) is exact. check() proves equality with apply_chat_template.

supervise: 'all_assistant' trains every assistant turn (social trajectories); 'last_assistant' only the final one."""
START, END = '<|START_OF_TURN_TOKEN|>', '<|END_OF_TURN_TOKEN|>'
ROLE = {'system': '<|SYSTEM_TOKEN|>', 'user': '<|USER_TOKEN|>', 'assistant': '<|CHATBOT_TOKEN|>'}
IGNORE = -100


def encode(tok, messages, supervise):
    """-> (input_ids, labels); labels are IGNORE outside supervised assistant spans."""
    assert supervise in ('all_assistant', 'last_assistant'), supervise
    assistant = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    assert assistant and messages[-1]['role'] == 'assistant', 'conversation must end with an assistant turn'
    piece = lambda text: tok(text, add_special_tokens=False)['input_ids']
    ids, labels = [tok.bos_token_id], [IGNORE]
    for i, m in enumerate(messages):
        head, body, end = piece(START + ROLE[m['role']]), piece(m['content']), piece(END)
        trained = m['role'] == 'assistant' and (supervise == 'all_assistant' or i == assistant[-1])
        ids += head + body + end
        labels += [IGNORE] * len(head) + (body + end if trained else [IGNORE] * (len(body) + len(end)))
    return ids, labels


def check(tok, rows):
    """Every row must encode exactly as the tokenizer's own chat template; returns summary numbers."""
    longest, supervised = 0, 0
    for r in rows:
        ids, labels = encode(tok, r['messages'], r['supervise'])
        ref = tok(tok.apply_chat_template(r['messages'], tokenize=False), add_special_tokens=False)['input_ids']
        assert ids == ref, f"{r['id']}: segment encoding differs from the chat template"
        longest = max(longest, len(ids))
        supervised += sum(x != IGNORE for x in labels)
    return {'rows': len(rows), 'longest_tokens': longest, 'supervised_tokens': supervised}
