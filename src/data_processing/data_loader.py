import json
import logging
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaintIEDataLoader:
    """Simple data loader for MaintIE JSON files"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with basic configuration"""
        self.config = config
        self.paths = config['paths']
        logging.info("MaintIEDataLoader initialized.")

    def load_gold_corpus(self) -> List[Dict[str, Any]]:
        """Load gold_release.json - must handle real format"""
        file_path = self.paths['gold_corpus']
        logging.info(f"Loading gold corpus from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Successfully loaded {len(data)} samples from gold corpus.")
            return data
        except FileNotFoundError:
            logging.error(f"Gold corpus file not found at {file_path}.")
            return []
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {file_path}.")
            return []

    def load_silver_corpus(self) -> List[Dict[str, Any]]:
        """Load silver_release.json - must handle real format"""
        file_path = self.paths['silver_corpus']
        logging.info(f"Loading silver corpus from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"Successfully loaded {len(data)} samples from silver corpus.")
            return data
        except FileNotFoundError:
            logging.error(f"Silver corpus file not found at {file_path}.")
            return []
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {file_path}.")
            return []

    def load_ontology(self) -> Dict[str, Any]:
        """Load scheme.json ontology"""
        file_path = self.paths['ontology']
        logging.info(f"Loading ontology from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ontology = json.load(f)
            logging.info("Successfully loaded ontology.")
            return ontology
        except FileNotFoundError:
            logging.error(f"Ontology file not found at {file_path}.")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {file_path}.")
            return {}

    def validate_sample_format(self, sample: Dict[str, Any]) -> bool:
        """Validate single sample against expected format"""
        required_keys = ['text', 'entities', 'relations']
        if not all(key in sample for key in required_keys):
            logging.warning("Sample missing required keys.")
            return False
        # Add more detailed validation logic here if needed
        return True

    def get_corpus_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate corpus statistics"""
        if not data:
            return {'total_samples': 0}

        num_samples = len(data)
        num_entities = sum(len(s.get('entities', [])) for s in data)
        num_relations = sum(len(s.get('relations', [])) for s in data)

        stats = {
            'total_samples': num_samples,
            'total_entities': num_entities,
            'total_relations': num_relations,
            'avg_entities_per_sample': num_entities / num_samples if num_samples > 0 else 0,
            'avg_relations_per_sample': num_relations / num_samples if num_samples > 0 else 0,
        }
        logging.info(f"Corpus statistics: {stats}")
        return stats

def extract_type_names(nodes: List[Dict], key: str) -> List[str]:
    """Recursively extract 'fullname' or 'name' from nested dictionaries."""
    names = []
    for node in nodes:
        if key in node:
            names.append(node[key])
        if 'children' in node and node['children']:
            names.extend(extract_type_names(node['children'], key))
    return names

def create_train_val_split(data: List[Dict], ratio: float = 0.8, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Simple train/validation split"""
    if not data:
        return [], []
    train_data, val_data = train_test_split(data, train_size=ratio, random_state=random_state)
    logging.info(f"Data split into {len(train_data)} training samples and {len(val_data)} validation samples.")
    return train_data, val_data

def load_maintie_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main data loading function"""
    loader = MaintIEDataLoader(config)

    gold_corpus = loader.load_gold_corpus()
    silver_corpus = loader.load_silver_corpus()
    ontology = loader.load_ontology()

    # Create mappings from type name to integer index
    entity_types = extract_type_names(ontology.get('entity', []), 'fullname')
    relation_types = extract_type_names(ontology.get('relation', []), 'name')
    entity_map = {name: i for i, name in enumerate(entity_types)}
    relation_map = {name: i for i, name in enumerate(relation_types)}

    # The actual split will be handled by creating masks in the graph builder

    return {
        'gold_corpus': gold_corpus,
        'silver_corpus': silver_corpus,
        'ontology': ontology,
        'entity_map': entity_map,
        'relation_map': relation_map,
    }

def main():
    """CLI entry point for make load-data"""
    import yaml

    logging.info("Starting data loading process from CLI...")
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data = load_maintie_data(config)

        # Optionally, save the splits or other artifacts
        # For now, just confirming the process runs
        if data['gold_corpus']:
            logging.info("Data loading and mapping creation completed successfully.")
        else:
            logging.warning("Data loading resulted in empty datasets.")

    except FileNotFoundError:
        logging.error("Configuration file 'config/config.yaml' not found.")
    except Exception as e:
        logging.error(f"An error occurred during the data loading process: {e}")

if __name__ == "__main__":
    main()
