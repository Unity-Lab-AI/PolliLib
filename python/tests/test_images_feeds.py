import os
import tempfile

from polliLib import PolliClient
from .conftest import FakeResponse, FakeSession


def test_generate_image_stream_to_file_and_fetch_bytes(tmp_path: tempfile.TemporaryDirectory):
    # Streamed image generation
    chunks = [b'a', b'b', b'c']
    fs = FakeSession()

    def fake_get(url, **kw):
        # For prompt URL with stream=True return chunks; otherwise bytes
        if kw.get('stream'):
            return FakeResponse(content_chunks=chunks)
        return FakeResponse(content=b'XYZ')

    fs.get = fake_get
    c = PolliClient(session=fs)
    out = c.generate_image("test", out_path=os.path.join(tmp_path, "gen.jpg"))
    assert os.path.exists(out)
    with open(out, 'rb') as f:
        assert f.read() == b''.join(chunks)

    # Direct fetch
    data = c.fetch_image("http://image/url.jpg")
    assert data == b'XYZ'


def test_save_image_timestamped(tmp_path: tempfile.TemporaryDirectory):
    fs = FakeSession()
    fs.get = lambda url, **kw: FakeResponse(content=b'JPG')
    c = PolliClient(session=fs)
    path = c.save_image_timestamped("test", images_dir=str(tmp_path))
    assert os.path.exists(path)
    assert path.endswith('.jpeg')


def test_image_feed_stream_parsing_and_bytes_and_data_url():
    # Without include options
    lines = [
        'data: {"prompt":"p1","imageURL":"http://img/1.jpg","model":"flux","seed":123}',
        'data: [DONE]'
    ]
    fs = FakeSession()
    fs.get = lambda url, **kw: FakeResponse(stream_lines=lines)
    c = PolliClient(session=fs)
    events = list(c.image_feed_stream())
    assert events and events[0]['prompt'] == 'p1'

    # include_bytes
    def get_with_image(url, **kw):
        if url.endswith('/feed'):
            return FakeResponse(stream_lines=lines)
        return FakeResponse(content=b'\xff\xd8\xff', headers={'Content-Type': 'image/jpeg'})

    fs = FakeSession()
    fs.get = get_with_image
    c = PolliClient(session=fs)
    events = list(c.image_feed_stream(include_bytes=True))
    assert events and events[0]['image_bytes'] == b'\xff\xd8\xff'

    # include_data_url
    fs = FakeSession()
    fs.get = get_with_image
    c = PolliClient(session=fs)
    events = list(c.image_feed_stream(include_data_url=True))
    assert events and events[0]['image_data_url'].startswith('data:image/jpeg;base64,')


def test_text_feed_stream_parsing():
    lines = [
        'data: {"model":"openai","messages":[{"role":"user","content":"hi"}],"response":"Hello"}',
        'data: [DONE]'
    ]
    fs = FakeSession()
    fs.get = lambda url, **kw: FakeResponse(stream_lines=lines)
    c = PolliClient(session=fs)
    ev = next(iter(c.text_feed_stream()))
    assert ev['model'] == 'openai' and ev['response'] == 'Hello'

