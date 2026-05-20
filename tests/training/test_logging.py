import numpy as np
import pytest

from vox.training.logging import Logger, StdoutLogger, build_logger


def test_stdout_logger_implements_protocol():
    log = StdoutLogger()
    assert isinstance(log, Logger)


def test_stdout_logger_prints_scalars(capsys):
    log = StdoutLogger(name="t")
    log.log({"loss": 1.234, "lr": 0.001}, step=10)
    captured = capsys.readouterr().out
    assert "loss=" in captured
    assert "lr=" in captured
    assert "step=10" in captured


def test_stdout_logger_audio_and_spec_no_crash():
    log = StdoutLogger()
    log.log_audio("sample", np.zeros(1000), sr=44100, step=0)
    log.log_spectrogram("mel", np.zeros((128, 40)), step=0)
    log.close()


def test_build_logger_dispatch():
    assert isinstance(build_logger("stdout"), StdoutLogger)
    with pytest.raises(ValueError):
        build_logger("nope")
