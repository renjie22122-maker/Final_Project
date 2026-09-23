"""Web contract tests; neural inference is deliberately mocked."""
import io
import os
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import app as module

def png():
    data = io.BytesIO()
    Image.new('RGB', (16, 16), (10, 90, 120)).save(data, format='PNG')
    data.seek(0)
    return data

class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=module.BASE_DIR)
        self.patch = patch.multiple(module, UPLOAD_FOLDER=self.temp.name, OUTPUT_FOLDER=self.temp.name)
        self.patch.start()
        module.conversion_history.clear()
        self.client = module.app.test_client()

    def tearDown(self):
        self.patch.stop()
        assert os.path.dirname(os.path.realpath(self.temp.name)) == os.path.realpath(module.BASE_DIR)
        self.temp.cleanup()

    def upload(self):
        response = self.client.post('/upload', data={'file': (png(), '../../image.png')})
        self.assertEqual(response.status_code, 200)
        return response.get_json()['filenames'][0]

    def test_home(self):
        self.assertEqual(self.client.get('/').status_code, 200)

    def test_upload_renames_and_serves_image(self):
        name = self.upload()
        self.assertNotIn('..', name)
        self.assertTrue(os.path.isfile(os.path.join(self.temp.name, name)))
        with self.client.get('/uploads/' + name) as response:
            self.assertEqual(response.status_code, 200)

    def test_invalid_upload_and_empty_batch(self):
        self.assertEqual(self.client.post('/upload').status_code, 400)
        response = self.client.post('/upload', data={'file': (io.BytesIO(b'bad'), 'bad.png')})
        self.assertEqual(response.status_code, 400)

    def test_invalid_request_and_traversal(self):
        self.assertEqual(self.client.post('/apply_style', json=[]).status_code, 400)
        for name in ['../x', r'..\x', 'C:x', '', '..']:
            self.assertEqual(self.client.post('/apply_style', json={'filename': name}).status_code, 400)

    def test_invalid_selection_and_missing_file(self):
        for field, value in [('model', 'bad'), ('direction', 'bad')]:
            self.assertEqual(self.client.post('/apply_style', json={'filename': 'x.png', field: value}).status_code, 400)
        self.assertEqual(self.client.post('/apply_style', json={'filename': 'missing.png'}).status_code, 404)

    def test_each_model_dispatch_and_history(self):
        name = self.upload()
        backends = {'cyclegan': 'cyclegan', 'cut': 'cut', 'unsb': 'unsb', 'cyclegan-turbo': 'turbo'}
        for model, suffix in backends.items():
            with patch.object(module, 'transform_image_' + suffix, return_value='result.png') as transform:
                response = self.client.post('/apply_style', json={'filename': name, 'model': model, 'direction': 'r2m'})
                self.assertEqual(response.status_code, 200)
                transform.assert_called_once_with(os.path.join(self.temp.name, name), 'r2m')
        self.assertEqual(len(self.client.get('/history').get_json()['history']), 4)

if __name__ == '__main__':
    unittest.main()
