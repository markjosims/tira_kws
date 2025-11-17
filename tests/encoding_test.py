from encoding import *
import librosa
from transformers import WhisperProcessor
from constants import DEVICE, SAMPLE_RATE

AILN_WAV = "data/ailn.wav"
EXPECTED_CLAP_IPA_SIZE = {
    'tiny': 384,
    'base': 512,
    'small': 512,
}

def load_test_audio():
    audio, _ = librosa.load(AILN_WAV, sr=SAMPLE_RATE)
    audio = np.expand_dims(audio, axis=0)
    return audio

def test_load_clap_speech_encoder():
    for size in ['tiny', 'base', 'small']:
        encoder = load_clap_speech_encoder(size)
        assert isinstance(encoder, SpeechEncoder)
        assert encoder.device == DEVICE

def test_load_clap_speech_processor():
    processor = load_clap_speech_processor()
    assert isinstance(processor, WhisperProcessor)

def test_encode_clap_audio():
    audio = load_test_audio()
    processor = load_clap_speech_processor()
    for size in ['tiny', 'base', 'small']:
        encoder = load_clap_speech_encoder(size)
        embeddings = encode_clap_audio(audio, encoder, processor)
        assert embeddings.shape[0] == audio.shape[0]
        assert embeddings.shape[1] == EXPECTED_CLAP_IPA_SIZE[size]