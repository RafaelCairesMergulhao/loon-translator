from tools.night_run import soak_once


def test_a_failed_phrase_does_not_stop_the_ones_after_it() -> None:
    spoken, errors = soak_once()
    assert spoken == 19
    assert errors
    assert any("trecho curto" in message for message in errors)


def test_the_same_batch_can_run_again() -> None:
    first, _errors = soak_once()
    second, _errors = soak_once()
    assert (first, second) == (19, 19)
