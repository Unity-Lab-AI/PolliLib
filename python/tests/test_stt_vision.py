import os
import tempfile

from polliLib import PolliClient
from .conftest import FakeResponse, FakeSession


def test_transcribe_audio_tmpfile(tmp_path: tempfile.TemporaryDirectory):
    # Create a tiny fake wav/mp3 file (content is irrelevant for test)
    audio_path = os.path.join(tmp_path, 'sample.wav')
    with open(audio_path, 'wb') as f:
        f.write(b'RIFF....WAVEfmt ')

    fs = FakeSession()
    fs.post = lambda url, **kw: FakeResponse(json_data={"choices": [{"message": {"content": "transcribed"}}]})
    c = PolliClient(session=fs)
    out = c.transcribe_audio(audio_path)
    assert out == 'transcribed'


def test_vision_analyze_url_and_file(tmp_path: tempfile.TemporaryDirectory):
    fs = FakeSession()
    fs.post = lambda url, **kw: FakeResponse(json_data={"choices": [{"message": {"content": "This is a bridge"}}]})
    c = PolliClient(session=fs)
    # URL
    out1 = c.analyze_image_url('http://x/y.jpg')
    assert out1 == 'This is a bridge'
    # File
    img_path = os.path.join(tmp_path, 'img.jpg')
    with open(img_path, 'wb') as f:
        f.write(b'\xff\xd8\xff')
    out2 = c.analyze_image_file(img_path)
    assert out2 == 'This is a bridge'

