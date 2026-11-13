"""Packaging/trace regressions requiring no ONNX weights or Apple hardware."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s2s_edge.hybrid_textnorm import HybridNormalizer
from test_hybrid_format import EXAMPLES


class PortabilityTests(unittest.TestCase):
    def test_bundled_crf_and_training_inputs_match_manifest(self):
        assets = ROOT/'assets/normalization'
        manifest = json.loads((assets/'identifier_training.json').read_text())
        for name, sha in [('identifier.crfsuite', manifest['model_sha256'])]:
            self.assertEqual(hashlib.sha256((assets/name).read_bytes()).hexdigest(), sha)
        self.assertEqual(hashlib.sha256((ROOT/manifest['source']).read_bytes()).hexdigest(), manifest['source_sha256'])

    def test_examples_with_real_bundled_classifier(self):
        h = HybridNormalizer()
        for lang, text, expected in EXAMPLES:
            with self.subTest(lang=lang, text=text): self.assertEqual(h(text, lang)[0], expected)
        self.assertEqual(h.identifier_model_status, 'CRF loaded')

    def test_address_zip_classifier_preserves_house_number(self):
        output, record = HybridNormalizer()('918 Oak Road, Cedar Hill, Texas, 58310.', 'vi')
        self.assertTrue(output.startswith('chín trăm mười tám'))
        self.assertTrue(output.endswith('năm tám ba một không.'))
        self.assertEqual([s['kind'] for s in record['spans']], ['number', 'identifier'])

    def test_isolated_import_does_not_load_experimental_neural_stack(self):
        code = "import sys; sys.path.insert(0, sys.argv[1]); from s2s_edge.tts_format import prepare; y,_=prepare('08:00','vi'); assert y=='tám giờ'; assert not any(x in sys.modules for x in ('mlx','mlx_lm','torch','transformers','s2s_edge.neural_textnorm'))"
        subprocess.run([sys.executable, '-I', '-c', code, str(ROOT)], check=True)


if __name__ == '__main__': unittest.main()
