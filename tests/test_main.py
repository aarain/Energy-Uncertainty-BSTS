from energy_uncertainty_bsts.main import get_placeholder


def test_get_placeholder_returns_correct_string():
    assert get_placeholder() == "Placeholder text"
