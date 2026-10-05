import torch
import numpy as np

def position_def(alpha):
    alpha=np.deg2rad(alpha)  #角度转弧度

    r=np.sqrt(np.cos(alpha)**2 + np.sin(alpha)**2)
    a=r*np.cos(alpha)
    b=r*np.sin(alpha)
    position = torch.tensor([[a,b]], dtype=torch.float32)
    return position


def positional_encoding(x, L=8):
    """
    Apply NeRF positional encoding to a 4D tensor.

    Args:
    x (torch.Tensor): Input tensor of shape (N, 4), where N is the batch size.
    L (int): Number of frequency bands for the encoding.

    Returns:
    torch.Tensor: Encoded tensor of shape (N, 4 * (2 * L + 1)).
    """
    # Ensure the input is a 2D tensor with shape (N, 4)
    assert x.ndim == 2 and x.shape[1] == 2, "Input tensor must be of shape (N, 4)"
    # assert x.ndim == 2 and x.shape[1] == 4, "Input tensor must be of shape (N, 4)"

    N, _ = x.shape
    frequencies = 2 ** torch.arange(L, dtype=torch.float32) * np.pi
    frequencies = frequencies.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, L)

    x = x.unsqueeze(-1)  # Shape (N, 4, 1)

    # Compute sin and cos for each frequency
    x_enc = torch.cat([torch.sin(frequencies * x), torch.cos(frequencies * x)], dim=-1)  # Shape (N, 4, 2L)

    # Flatten the last two dimensions
    x_enc = x_enc.view(N, -1)  # Shape (N, 4 * 2L)
    return x_enc

# def positional_encoding(x, L):
#     """
#     Apply NeRF positional encoding to a 4D tensor.
#
#     Args:
#     x (torch.Tensor): Input tensor of shape (N, 2), where N is the batch size.
#     L (int): Number of frequency bands for the encoding.
#
#     Returns:
#     torch.Tensor: Encoded tensor of shape (N, 2 * (2 * L + 1)).
#     """
#     # Ensure the input tensor has the correct shape
#     assert x.shape[1] == 2, "Input tensor must have shape (N, 2)"
#
#     N = x.shape[0]
#     encoded = [x]
#
#     for i in range(L):
#         for func in [torch.sin, torch.cos]:
#             encoded.append(func(2.0 ** i * x))
#
#     encoded_tensor = torch.cat(encoded, dim=1)
#     return encoded_tensor



# Example usage
if __name__ == "__main__":
    # x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    # x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    x = position_def(5)
    print(x.cpu().detach().numpy().shape)
    encoded_x = positional_encoding(x)
    print(encoded_x.cpu().detach().numpy().ndim)
    print(encoded_x.cpu().detach().numpy().shape)