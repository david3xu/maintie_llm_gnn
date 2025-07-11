#!/usr/bin/env python3
"""CLI entry point for data processing commands"""
import argparse
from .data_pipeline import MaintIEDataLoader
from .embedding_generator import EmbeddingGenerator
from .graph_builder import MaintenanceGraphBuilder

def main():
    parser = argparse.ArgumentParser(description='MaintIE Data Processing')
    subparsers = parser.add_subparsers(dest='command')

    # load-data command
    load_parser = subparsers.add_parser('load-data')
    load_parser.add_argument('--corpus', choices=['gold', 'silver'], required=True)

    # generate-embeddings command
    embed_parser = subparsers.add_parser('generate-embeddings')
    embed_parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')

    # build-graphs command
    graph_parser = subparsers.add_parser('build-graphs')
    graph_parser.add_argument('--config', default='config/default_config.yaml')

    args = parser.parse_args()

    if args.command == 'load-data':
        loader = MaintIEDataLoader()
        if args.corpus == 'gold':
            data = loader.load_gold_corpus('data/raw/gold_release.json')
        else:
            data = loader.load_silver_corpus('data/raw/silver_release.json')
        print(f"Loaded {len(data)} samples from {args.corpus} corpus.")

    elif args.command == 'generate-embeddings':
        import json
        loader = MaintIEDataLoader()
        data = loader.load_silver_corpus('data/raw/silver_release.json')
        texts = [sample['text'] for sample in data]
        generator = EmbeddingGenerator(args.model)
        embeddings = generator.generate_embeddings(texts)
        generator.save_embeddings(embeddings, 'data/processed/embeddings/embeddings.pkl')
        print(f"Embeddings saved to data/processed/embeddings/embeddings.pkl")

    elif args.command == 'build-graphs':
        import json
        loader = MaintIEDataLoader()
        data = loader.load_silver_corpus('data/raw/silver_release.json')
        with open('data/processed/embeddings/embeddings.pkl', 'rb') as f:
            import pickle
            embeddings = pickle.load(f)
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        graph_builder = MaintenanceGraphBuilder(config)
        texts = [sample['text'] for sample in data]
        graph_data = graph_builder.build_maintenance_graph(
            node_features=embeddings,
            texts=texts
        )
        graph_builder.save_graph(graph_data, 'data/processed/graphs/graph.pt')
        print(f"Graph saved to data/processed/graphs/graph.pt")

if __name__ == '__main__':
    main()
