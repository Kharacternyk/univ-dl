from sentencepiece import SentencePieceProcessor
import torch


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, in_lang, out_lang, sequence_length):
        super().__init__()
        self.in_processor = SentencePieceProcessor()
        self.out_processor = SentencePieceProcessor()
        self.in_processor.Load(f"{in_lang}.model")
        self.out_processor.Load(f"{out_lang}.model")
        self.in_data = read_lines(in_lang)
        self.out_data = read_lines(out_lang)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, index):
        in_sentence = self.fix_length(
            self.in_processor.EncodeAsIds(self.in_data[index].strip())
        )
        out_sentence_bos = self.fix_length(
            [1] + self.out_processor.EncodeAsIds(self.out_data[index].strip())
        )
        out_sentence_eos = self.fix_length(
            self.out_processor.EncodeAsIds(self.out_data[index].strip()) + [2]
        )

        return in_sentence, out_sentence_bos, out_sentence_eos

    def fix_length(self, tokens):
        if len(tokens) < self.sequence_length:
            tokens += [0] * (self.sequence_length - len(tokens))
        else:
            tokens = tokens[: self.sequence_length]
        return torch.tensor(tokens)


def read_lines(path):
    with open(path) as file:
        return file.readlines()
