import unittest
import torch
import yaml
from src.models.llm_gnn_hybrid import create_model

class TestModel(unittest.TestCase):
    def setUp(self):
        # Create a dummy config for testing
        self.config_str = """
model:
  embedding_dim: 32
  hidden_dim: 16
  num_gnn_layers: 2
  dropout: 0.1
complexity_levels:
  FG-0: {entity_classes: 5, relation_classes: 3}
"""
        self.config = yaml.safe_load(self.config_str)
        self.model = create_model(self.config, complexity_level='FG-0')

    def test_model_initialization(self):
        """Test model creates correctly"""
        self.assertIsNotNone(self.model)
        # Check if output heads have correct dimensions
        self.assertEqual(self.model.output_heads['entity'].out_features, 5)
        self.assertEqual(self.model.output_heads['relation'].out_features, 3)

    def test_forward_pass(self):
        """Test model forward pass"""
        # Dummy input batch
        batch = {
            'x': torch.randn(10, self.config['model']['embedding_dim']), # 10 nodes
            'edge_index': torch.randint(0, 10, (2, 20)) # 20 edges
        }
        output = self.model(batch)
        self.assertIn('entity_logits', output)
        self.assertIn('relation_logits', output)

    def test_output_dimensions(self):
        """Test output shapes are correct"""
        num_nodes = 10
        batch = {
            'x': torch.randn(num_nodes, self.config['model']['embedding_dim']),
            'edge_index': torch.randint(0, num_nodes, (2, 20))
        }
        output = self.model(batch)

        entity_logits = output['entity_logits']
        relation_logits = output['relation_logits']

        self.assertEqual(entity_logits.shape, (num_nodes, 5))
        self.assertEqual(relation_logits.shape, (num_nodes, 3))

if __name__ == '__main__':
    unittest.main()
