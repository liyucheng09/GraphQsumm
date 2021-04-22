from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, \
    Trainer, AutoConfig, TrainingArguments, HfArgumentParser
import os
import torch
from model import RobertaSeq2Seq, BertSeq2seq
from utils import TrainingConfig, load_cached_hf_parameters, TextGenTrainer
import sys
import pickle
from GraphModel.copy_summ_multiencoder import CopySummGAT, CopySummParagraph


MODEL_CLASSES = {
    'robert-base': RobertaSeq2Seq,
    'bert-base-chinese': BertSeq2seq,
    "bert-base-chinese-local": BertSeq2seq,
    "zh-wwm-roberta": BertSeq2seq,
    "graphqsumm" : CopySummGAT
}


class DebatePediaDataSet:
    def __init__(self, corpus_type: str, data_path, tokenizer: RobertaTokenizer, max_length=512):
        self.summary_path = os.path.join(data_path, corpus_type + '_summary')
        self.query_path = os.path.join(data_path, corpus_type + '_query')
        self.content_path = os.path.join(data_path, corpus_type + '_content')

        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(self.summary_path, encoding='utf-8') as f:
            self.summary = [i for i in f.read().split('\n') if i]
        with open(self.query_path, encoding='utf-8') as f:
            self.query = [i for i in f.read().split('\n') if i]
        with open(self.content_path, encoding='utf-8') as f:
            self.content = [i for i in f.read().split('\n') if i]

    def __len__(self):
        return len(self.summary)

    def __getitem__(self, item):
        query = self.query[item]
        content = self.content[item]
        summary = self.summary[item]

        encoded = self.tokenizer(query + '[SEP]' + content, summary, max_length=self.max_length, padding=True,
                                 truncation=True)

        return {k: torch.LongTensor(v) for k, v in encoded.items()}


class QsummDataSet:
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.train = DebatePediaDataSet('train', data_path, self.tokenizer, max_length=512)
        self.valid = DebatePediaDataSet('valid', data_path, self.tokenizer, max_length=512)
        self.test = DebatePediaDataSet('test', data_path, self.tokenizer, max_length=512)


def main():
    assert sys.argv[1].endswith(".json"), \
        "Need to specify the config.json file."

    parser = HfArgumentParser(TrainingConfig)
    args = parser.parse_json_file(os.path.abspath(sys.argv[1]))[0]

    if args.server:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cached_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cached_dir)

    # specify configuration
    config.num_labels = tokenizer.vocab_size
    config.type_vocab_size = args.type_vocab_size
    config.eos_token_id = 102
    config.return_dict = True

    model = MODEL_CLASSES[args.model_name_or_path](config)
    state_dict = torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin')) \
        if args.server else load_cached_hf_parameters(args.model_name_or_path, args.cached_dir)

    # if 'roberta' in args.model_name_or_path:
    #     state_dict.pop('roberta.embeddings.token_type_embeddings.weight')
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()

    if args.do_test:

        from utils import generate_samples, PROMPT_TEXT

        generate_samples(
            model,
            tokenizer,
            PROMPT_TEXT,
        )

    if args.task == 'thuc':

        from bert_seq2seq.THUCNews import collate_fn
        import datasets

        if args.thuc_cache_dir and os.path.exists(args.thuc_cache_dir[0]) and os.path.exists(args.thuc_cache_dir[1]):
            tokenized_train = datasets.load_from_disk(args.thuc_cache_dir[0])
            tokenized_test = datasets.load_from_disk(args.thuc_cache_dir[1])
        else:
            dataset = datasets.load_dataset('thuc_datasets.py', data_path=args.thuc_data_path, split='train')
            dataset = dataset.train_test_split(test_size=0.05, seed=42)

            train_dataset = dataset['train']
            test_dataset = dataset['test']

            def tokenize(examples):
                return tokenizer(examples['content'], examples['title'])

            tokenized_train = train_dataset.map(
                tokenize,
                remove_columns=train_dataset.column_names,
                batched=True,
                num_proc=args.dataloader_num_workers,
            )

            tokenized_test = test_dataset.map(
                tokenize,
                remove_columns=test_dataset.column_names,
                batched=True,
                num_proc=args.dataloader_num_workers,
            )

            tokenized_train.save_to_disk('thuc.train.cache')
            tokenized_test.save_to_disk('thuc.test.cache')

        trainer = TextGenTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            data_collator=collate_fn,
            tokenizer=tokenizer,
        )

    elif args.task == 'debatepedia':
        qsumm_dataset = QsummDataSet(args.qsumm_data_path, tokenizer)

        args.evaluation_strategy = 'epoch'
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=qsumm_dataset.train,
            eval_dataset=qsumm_dataset.valid,
            tokenizer=tokenizer,
        )

    if args.do_train:
        trainer.train()
        trainer.save_model()


if __name__ == '__main__':
    main()