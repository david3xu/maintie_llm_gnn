import unittest
import os
import json
from src.data_processing.data_loader import MaintIEDataLoader

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        # Create dummy data and config for testing
        self.config = {
            'paths': {
                'gold_corpus': 'test_gold.json',
                'silver_corpus': 'test_silver.json',
                'ontology': 'test_scheme.json'
            }
        }
        # Dummy gold corpus
        with open(self.config['paths']['gold_corpus'], 'w') as f:
            json.dump([{'text': 'sample 1', 'entities': [], 'relations': []}], f)
        # Dummy ontology
        with open(self.config['paths']['ontology'], 'w') as f:
            json.dump({'entities': {}, 'relations': {}}, f)

        self.loader = MaintIEDataLoader(self.config)

    def tearDown(self):
        # Clean up dummy files
        for path in self.config['paths'].values():
            if os.path.exists(path):
                os.remove(path)

    def test_load_gold_corpus(self):
        """Test gold corpus loading"""
        gold_data = self.loader.load_gold_corpus()
        self.assertIsInstance(gold_data, list)
        self.assertEqual(len(gold_data), 1)

    def test_load_silver_corpus_not_found(self):
        """Test silver corpus loading when file does not exist"""
        silver_data = self.loader.load_silver_corpus()
        self.assertEqual(silver_data, [])

    def test_data_validation(self):
        """Test data format validation"""
        valid_sample = {'text': 't', 'entities': [], 'relations': []}
        invalid_sample = {'text': 't'}
        self.assertTrue(self.loader.validate_sample_format(valid_sample))
        self.assertFalse(self.loader.validate_sample_format(invalid_sample))

if __name__ == '__main__':
    unittest.main()
