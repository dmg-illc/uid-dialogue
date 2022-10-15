
import torch


def add_context(inputs, context, context_length, tokenizer, device):
    BOS_ID = tokenizer.bos_token_id
    NO_LABEL_ID = -100

    if context is None:
        context = {
            'input_ids': torch.full((1, context_length), BOS_ID, dtype=torch.long),
            'attention_mask': torch.zeros(1, context_length)
        }

    new_input_ids = torch.cat(
        (context['input_ids'][:, -context_length:], inputs['input_ids']), dim=1)
    new_attention_mask = torch.cat(
        (context['attention_mask'][:, -context_length:], inputs['attention_mask']), dim=1)
    labels_w_ctx = torch.cat(
        (torch.full((1, context_length), NO_LABEL_ID, dtype=torch.long), inputs['input_ids']), dim=1)

    inputs_w_ctx = {
        'input_ids': new_input_ids[:, :tokenizer.max_len_single_sentence],
        'attention_mask': new_attention_mask[:, :tokenizer.max_len_single_sentence]
    }
    labels_w_ctx = labels_w_ctx[:, :tokenizer.max_len_single_sentence]

    new_context = {
        'input_ids': new_input_ids[:, -context_length:],
        'attention_mask': new_attention_mask[:, -context_length:]
    }

    inputs_w_ctx['input_ids'] = inputs_w_ctx['input_ids'].to(device)
    inputs_w_ctx['attention_mask'] = inputs_w_ctx['attention_mask'].to(device)
    labels_w_ctx = labels_w_ctx.to(device)

    return inputs_w_ctx, labels_w_ctx, new_context
