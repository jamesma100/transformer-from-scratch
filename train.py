from transformer import *


if __name__ == "__main__":
    d_model = 128
    ctx_sz = 10
    batch_sz = 100
    h = 8
    tokens_path = "./out/tokens.txt"
    base_path = "./out/base.json"
    vocab_path = "./out/vocab.json"

    lr = 1e-2

    with open(tokens_path, "r") as fp:
        src_tokens = [int(token) for token in fp.read().split(",")]

    data_sz = len(src_tokens)
    split = int(data_sz * 0.7)

    train, test = src_tokens[:split], src_tokens[split:]

    with open(base_path) as fp:
        base = {int(k): v for k, v in json.load(fp).items()}
    with open(vocab_path) as fp:
        vocab = {int(k): v for k, v in json.load(fp).items()}

    num_epochs = 1000
    num_batches = len(train) // batch_sz

    print("[INFO] training data size: ", len(train))

    print("[INFO] total vocab size: ", len(base) + len(vocab))
    print("[INFO] B: ", batch_sz)
    print("[INFO] C: ", ctx_sz)
    print("[INFO] d_model: ", d_model)
    print("[INFO] num epochs: ", num_epochs)
    print("[INFO] num batches: ", num_batches)

    transformer = Transformer(ctx_sz, h, d_model, 6, len(base) + len(vocab))
    adam = torch.optim.AdamW(transformer.parameters(), lr=lr)

    # training loop
    transformer.train()
    for epoch in range(num_epochs):
        losses = []
        for batch in range(num_batches):
            X, Y = get_batch(train, ctx_sz, batch_sz)

            Y_hat = transformer(X, Y)

            B, C, V = Y_hat.size()  # dimensions of logit
            loss = get_loss(Y_hat.view(B * C, V), Y.view(B * C))
            losses.append(loss)
            adam.zero_grad(set_to_none=True)
            loss.backward()
            adam.step()
        epoch_loss = sum(losses) / len(losses)
        print("[INFO] epoch loss: ", epoch_loss)

    tokenizer = Tokenizer("")
    transformer.generate(
        torch.tensor([[0]], dtype=torch.long),
        10000,
        len(vocab) + len(base),
        d_model,
        ctx_sz,
    )
