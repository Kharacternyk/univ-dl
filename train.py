from itertools import count

from fire import Fire
import torch

from dataset import TranslationDataset, read_lines
from transformer import Transformer


def main(
    in_lang="de",
    out_lang="en",
    n_iters=10000,
    checkpoint_period=500,
    batch_size=128,
    num_workers=4,
    lr=0.0001,
    sequence_length=128,
    depth=6,
    breadth=512,
    dropout=0.1,
    n_heads=8,
):
    vocabulary_size = len(read_lines(f"{in_lang}.vocab"))

    model = Transformer(
        in_vocabulary_size=vocabulary_size,
        out_vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        depth=depth,
        breadth=breadth,
        dropout=dropout,
        n_heads=n_heads,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TranslationDataset(in_lang, out_lang, sequence_length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    print(f"Epoch length is {len(dataloader)}")

    for parameter in model.parameters():
        if parameter.dim() > 1:
            torch.nn.init.xavier_uniform_(parameter)

    global_index = 0
    losses = []

    for epoch in count():
        for index, (
            in_sentence,
            out_sentence_bos,
            out_sentence_eos,
        ) in enumerate(dataloader):
            optimizer.zero_grad()

            in_sentence = in_sentence.cuda()
            out_sentence_bos = out_sentence_bos.cuda()
            out_sentence_eos = out_sentence_eos.cuda()

            e_mask = (in_sentence != 0).unsqueeze(1)
            d_mask = (out_sentence_bos != 0).unsqueeze(1)
            d_mask = d_mask & (
                torch.tril(
                    torch.ones(
                        [1, sequence_length, sequence_length],
                        dtype=torch.bool,
                        device="cuda",
                    )
                )
            )

            output = model(in_sentence, out_sentence_bos, e_mask, d_mask)

            loss = torch.nn.functional.nll_loss(
                output.view(-1, vocabulary_size),
                out_sentence_eos.view(
                    out_sentence_eos.size(0) * out_sentence_eos.size(1)
                ),
            )

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, batch {index}, loss {loss.item()}")

            global_index += 1
            losses.append(loss)

            if not global_index % checkpoint_period:
                torch.save(model, f"checkpoint_{global_index}.pt")

            if global_index >= n_iters:
                break

        if global_index >= n_iters:
            break

    torch.save(model, f"checkpoint_{n_iters}.pt")
    torch.save(losses, "losses.pt")


if __name__ == "__main__":
    Fire(main)
