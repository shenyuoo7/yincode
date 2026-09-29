import importlib

from rich.cells import cell_len


def test_banner_is_readable_at_narrow_width():
    mod = importlib.import_module("yincode.prompt")
    banner = mod.render_banner("0.1.0", "E:/very/long/project/path", width=24)
    assert "YIN" in banner
    assert "yincode v0.1.0" in banner
    assert "E:/very/long/project/path" in banner.replace("\n", "")
    assert "Enter" in banner


def test_banner_wraps_chinese_paths_by_terminal_cells():
    mod = importlib.import_module("yincode.prompt")
    cwd = "E:/中文项目目录/很长的中文工作目录"
    banner = mod.render_banner("0.1.0", cwd, width=24)
    assert cwd in banner.replace("\n", "")
    assert all(cell_len(line) <= 24 for line in banner.splitlines())
