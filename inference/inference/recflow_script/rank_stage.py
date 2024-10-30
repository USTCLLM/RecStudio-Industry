import sys
sys.path.append('.')
import argparse 
import yaml
from inference.inference.inference_engine import InferenceEngine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_config_path", type=str, required=True, help="Inference config file")  
    args = parser.parse_args()

    with open(args.infer_config_path, 'r') as f:
        config = yaml.safe_load(f)

    rank_inference_engine = InferenceEngine(config)

    outputs = rank_inference_engine.batch_inference()