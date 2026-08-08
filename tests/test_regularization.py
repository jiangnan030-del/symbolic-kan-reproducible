import torch

from symbolic_kan.regularization import selection_terms


def test_true_nms_detects_edge_overlap() -> None:
    identical = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    diverse = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert selection_terms([identical]).nms > selection_terms([diverse]).nms
