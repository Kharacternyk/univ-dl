from sentencepiece import SentencePieceTrainer

for lang in ["de", "en"]:
    SentencePieceTrainer.Train(
        f"--input={lang} --model_prefix={lang} --vocab_size=10000 --model_type=unigram "
        "--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --character_coverage=1.0"
    )
