from fire import Fire
import torch

from dataset import TranslationDataset


@torch.inference_mode()
def main(checkpoint, text, in_lang="de", out_lang="en", verbose=False):
    model = torch.load(checkpoint, weights_only=False)
    model.cuda().eval()

    if verbose:
        print(model)
        print(sum(parameter.numel() for parameter in model.parameters()), "parameters")

    dataset = TranslationDataset(in_lang, out_lang, model.sequence_length)
    tokens = (
        dataset.fix_length(dataset.in_processor.EncodeAsIds(text)).unsqueeze(0).cuda()
    )
    e_mask = (tokens != 0).unsqueeze(1)
    e_output = model.encoder(model.pos_embed(model.in_embed(tokens)), e_mask)
    decoded = torch.tensor([0] * model.sequence_length).cuda()
    decoded[0] = 1

    for i in range(model.sequence_length):
        d_mask = (decoded != 0).unsqueeze(0).unsqueeze(0)
        d_mask = d_mask & torch.tril(
            torch.ones(
                [1, model.sequence_length, model.sequence_length],
                dtype=torch.bool,
                device="cuda",
            )
        )

        d_output = model.decoder(
            model.pos_embed(model.out_embed(decoded.unsqueeze(0))),
            e_output,
            e_mask,
            d_mask,
        )
        next_token = torch.argmax(model.last(d_output), -1)[0][i].item()

        if i < model.sequence_length - 1:
            decoded[i + 1] = next_token

        if next_token == 2:
            break

    print(dataset.out_processor.DecodeIds(decoded[1:i].tolist()))


if __name__ == "__main__":
    Fire(main)
