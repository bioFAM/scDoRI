import torch

from scdori._core.utils import log_nb_positive


def test_log_nb_positive_values():
    """Test with known values and expected output."""
    x = torch.tensor([3.0, 0.0, 5.0])
    mu = torch.tensor([2.0, 1.0, 5.0])
    theta = torch.tensor([1.0, 1.0, 1.0])
    result = log_nb_positive(x, mu, theta)

    assert torch.allclose(result, torch.tensor([-2.3150, -0.6931, -2.7034]), atol=0.1)


def test_log_nb_positive_zero_count():
    """Test with zero counts."""
    x = torch.tensor([0.0, 0.0, 0.0])
    mu = torch.tensor([2.0, 1.0, 5.0])
    theta = torch.tensor([1.0, 1.0, 1.0])

    result = log_nb_positive(x, mu, theta)

    # For zero counts, the log likelihood should be theta * log(theta/(theta+mu))
    expected = theta * (torch.log(theta) - torch.log(theta + mu))
    assert torch.allclose(result, expected)


def test_log_nb_positive_custom_functions():
    """Test with custom log and lgamma functions."""
    x = torch.tensor([3.0, 0.0, 5.0])
    mu = torch.tensor([2.0, 1.0, 5.0])
    theta = torch.tensor([1.0, 1.0, 1.0])

    def custom_log(x):
        return torch.log(x) + 1

    def custom_lgamma(x):
        return torch.lgamma(x) + 2

    standard_result = log_nb_positive(x, mu, theta)
    custom_result = log_nb_positive(x, mu, theta, log_fn=custom_log, lgamma_fn=custom_lgamma)

    # The custom result should differ from the standard result
    assert not torch.allclose(standard_result, custom_result)
