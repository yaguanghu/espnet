#!/usr/bin/env python3

# Copyright (c)  2021  Mobvoi Inc. (authors: Yaguang Hu)
# Apache 2.0

import torch

from espnet.nets.pytorch_backend.k2_ctc import K2CTCLoss


def test_k2_ctc():
    T = 5  # Input sequence length
    C = 5  # Number of classes (including blank)
    N = 4  # Batch size
    S = 3  # Target sequence length of longest target in batch
    input = torch.randn(T, N, C).to(torch.float32).detach().requires_grad_(True)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)

    target_lengths = torch.randint(low=1, high=S, size=(N,), dtype=torch.int32)
    target = torch.randint(
        low=1, high=C, size=(sum(target_lengths),), dtype=torch.int32
    )

    pytorch_log_probs = input.log_softmax(2)
    pytroch_ctc_loss = torch.nn.CTCLoss(reduction="sum")
    pytorch_loss = pytroch_ctc_loss(
        pytorch_log_probs, target, input_lengths, target_lengths
    )
    pytorch_loss.backward()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        k2_ctc_loss = K2CTCLoss(C)
        k2_input = input.detach().clone().to(device).requires_grad_(True)
        k2_log_probs = k2_input.log_softmax(2)
        k2_loss = k2_ctc_loss(k2_log_probs, target, input_lengths, target_lengths)
        k2_loss.backward()
        print(f"k2 loss: {k2_loss}, pytorch loss: {pytorch_loss}")
        assert torch.allclose(k2_loss, pytorch_loss)
        print(f"k2 grad: {input.grad}, pytorch grad: {input.grad}")
        assert torch.allclose(input.grad.to(device), k2_input.grad)


if __name__ == "__main__":
    test_k2_ctc()
