from ui import win_chrome


class _Missing:
    def winfo_id(self):
        raise RuntimeError("sem janela")


def test_other_platforms_do_not_touch_the_window(monkeypatch) -> None:
    monkeypatch.setattr(win_chrome.sys, "platform", "linux")
    assert win_chrome.pin_window_opaque(object()) is False


def test_a_window_without_a_handle_is_left_alone() -> None:
    assert win_chrome.pin_window_opaque(_Missing()) is False


class _Events:
    def __init__(self) -> None:
        self.bound: list[str] = []
        self.afters: list[int] = []

    def bind(self, sequence, callback, add=None) -> None:
        self.bound.append(sequence)

    def after(self, delay, callback) -> None:
        self.afters.append(delay)

    def deiconify(self) -> str:
        return "up"

    def wm_deiconify(self) -> str:
        return "up"

    def iconify(self) -> str:
        return "down"

    def wm_iconify(self) -> str:
        return "down"


def test_restore_and_minimize_pin_the_window_again(monkeypatch) -> None:
    pinned: list[object] = []
    monkeypatch.setattr(win_chrome, "pin_window_opaque", lambda window: pinned.append(window) or True)
    window = _Events()

    win_chrome.bind_opaque(window)

    assert window.bound == ["<Activate>", "<Map>", "<Unmap>"]
    assert window.afters == [250, 900]
    assert window.deiconify() == "up"
    assert pinned == [window]
    assert 80 in window.afters
    window.iconify()
    assert pinned[-1] is window
