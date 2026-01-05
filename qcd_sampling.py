import os
from typing import Optional, Union

import torch
from torch import nn
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    ALL_CACHE_NAMES,
    logger,
)

import transformers
from transformers import GenerationConfig


def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    pad_id_scalar = int(generation_config.pad_token_id) if generation_config.pad_token_id is not None else None
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample

    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    _, cd_cur_len = model_kwargs['input_ids_cd'].shape[:2]
    model_kwargs_cd = dict(model_kwargs)
    model_kwargs_cd = self._get_initial_cache_position(cd_cur_len, input_ids.device, model_kwargs_cd)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if self.config._attn_implementation == "flash_attention_2":
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                    "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    use_cd = ("input_ids_cd" in model_kwargs) and (model_kwargs["input_ids_cd"] is not None)
    cd_full_len = input_ids.shape[1]

    if use_cd:
        cd_prefix = model_kwargs_cd.pop("input_ids_cd").to(input_ids.device)  # [B, L_cd]
        model_kwargs_cd["cd_prefix_ids"] = cd_prefix
        model_kwargs_cd["cd_full_len"] = int(cd_full_len)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(**{**model_kwargs, "input_ids": input_ids})
        outputs = (self if is_prefill else model_forward)(**model_inputs, return_dict=True)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        if use_cd:
            tail = input_ids[:, cd_full_len:]
            cd_input_ids = torch.cat([model_kwargs_cd["cd_prefix_ids"], tail], dim=1)

            attention_mask_cd = self._prepare_attention_mask_for_generation(
                cd_input_ids, generation_config, {"input_ids": cd_input_ids}
            )

            mk_cd = dict(model_kwargs_cd)
            mk_cd["input_ids"] = cd_input_ids
            mk_cd["attention_mask"] = attention_mask_cd
            model_inputs_cd = self.prepare_inputs_for_generation(**mk_cd)
            outputs_cd = (self if is_prefill else model_forward)(**model_inputs_cd, return_dict=True)
            next_token_logits_cd = outputs_cd.logits[:, -1, :].to(dtype=torch.float32, device=input_ids.device)

            cd_alpha = float(model_kwargs.get("cd_alpha", 0.5))

            cd_logits = (1.0 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd

            next_token_scores = logits_processor(input_ids, cd_logits)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if has_eos_stopping_criteria and pad_id_scalar is not None:
            next_tokens = next_tokens * unfinished_sequences + pad_id_scalar * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
        del outputs
        if use_cd:
            del outputs_cd

        if is_prefill:
            is_prefill = False

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        cache = None
        if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
            cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )
    else:
        return input_ids


def evolve_qcd_sampling():
    if hasattr(transformers.generation.utils.GenerationMixin, "_sample"):
        transformers.generation.utils.GenerationMixin._sample = sample
    else:
        transformers.generation.utils.GenerationMixin.sample = sample
