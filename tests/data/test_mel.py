from vox.data.features.mel import MelExtractor


def test_output_shape(sine_wav, sr):
    ext = MelExtractor(sr=sr)
    mel = ext(sine_wav)
    assert mel.dim() == 2
    assert mel.shape[0] == MelExtractor.N_MELS


def test_output_range(sine_wav, sr):
    ext = MelExtractor(sr=sr)
    mel = ext(sine_wav)
    assert mel.min() >= 0.0
    assert mel.max() <= 1.0


def test_440hz_peak(sine_wav, sr):
    """440 Hz sine should have energy concentrated near 440 Hz mel bin."""
    ext = MelExtractor(sr=sr)
    mel = ext(sine_wav)  # (n_mels, T_mel)

    # Average across time, find peak mel bin
    avg = mel.mean(dim=1)  # (n_mels,)
    peak_bin = int(avg.argmax())

    # mel filterbank center frequencies at 44100 Hz, 128 mels
    import torchaudio.functional as F_audio
    freqs = F_audio.melscale_fbanks(
        n_freqs=MelExtractor.N_FFT // 2 + 1,
        f_min=0.0,
        f_max=sr / 2,
        n_mels=MelExtractor.N_MELS,
        sample_rate=sr,
    )  # (n_freqs, n_mels)
    # center freq per mel bin = argmax along frequency axis
    center_bins = freqs.argmax(dim=0)  # (n_mels,)
    bin_hz = center_bins.float() * (sr / MelExtractor.N_FFT)

    # The peak mel bin should correspond to a frequency within ±1 octave of 440
    peak_hz = float(bin_hz[peak_bin])
    assert 220 <= peak_hz <= 880, f"Peak at {peak_hz:.1f} Hz, expected near 440 Hz"


def test_frames_helper(sine_wav, sr):
    """frames() must match the actual mel output (center=True)."""
    ext = MelExtractor(sr=sr)
    mel = ext(sine_wav)
    assert ext.frames(len(sine_wav)) == mel.shape[-1]
