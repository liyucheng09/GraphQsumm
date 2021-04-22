from dataclasses import field, dataclass
from transformers import TrainingArguments, TrainerCallback, Trainer, BertTokenizer, PreTrainedModel
from transformers.file_utils import hf_bucket_url, cached_path
import torch
import json

with open('gen_config.json', encoding='utf-8') as f:
    args=json.load(f)

PROMPT_TEXT="""新浪体育讯　日本自推出“梦想巨奖”及“百万梦想”彩票后，销售火爆，自去年下半年就开始的招募2010年的彩票“幸运女神”活动最终在今年4月份结束并诞生了6名幸运女神。据报道，“幸运女神”从1980年“诞生”以来，已有近30年的历史，本次诞生的名幸运女神已经是第31次向日本全社会公开招募。
　　2010年当选的六名彩票幸运女神分别是(上图从左开始)冈田麻里MM，木村晓奈MM，新村理纱MM，番匠麻衣MM，山田舞子和吉野惠小姐。
　　5月14日，幸运女神之一的番匠麻衣小姐到访相模原市政府，向市长加山俊夫及该市市民推荐宣传两大梦想彩票。"""

@dataclass()
class TrainingConfig(TrainingArguments):

    model_name_or_path: str='roberta-base'
    cached_dir: str=field(
        default='~/project/model_cache'
    )

    type_vocab_size: int=2
    task: str = 'thuc'
    thuc_data_path:str=None
    qsumm_data_path:str=None
    max_length:int=512
    server:bool=False
    thuc_cache_dir:str=None
    do_test=False


def load_cached_hf_parameters(model_name_or_path, cache_dir):
    archive_file = hf_bucket_url(
        model_name_or_path,
        filename='pytorch_model.bin'
    )
    resolved_archive_file = cached_path(
        archive_file,
        cache_dir=cache_dir
    )
    state_dict = torch.load(resolved_archive_file, map_location="cpu")
    return state_dict


class TextGenTrainer(Trainer):

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            generate_samples(model, self.tokenizer, prompt_text=PROMPT_TEXT)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def generate_samples(
    model: PreTrainedModel,
    tokenizer: BertTokenizer,
    prompt_text: str,
    max_length=args['max_length'],
    temperature=args['temperature'],
    top_k=args['k'],
    top_p=args['p'],
    repetition_penalty=args['repetition_penalty'],
    num_return_sequences=args['num_return_sequences'],
    stop_token=args['stop']
    ):

    encoded_prompt=tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt')
    encoded_prompt=encoded_prompt.to(model.device)
    input_ids=encoded_prompt if encoded_prompt.shape[-1]>0 else None

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(stop_token) if args['stop_token'] else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        print(total_sequence)

    return generated_sequences