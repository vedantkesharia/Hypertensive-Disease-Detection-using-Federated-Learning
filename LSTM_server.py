import flwr as fl
from typing import List, Tuple, Dict

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not metrics:
        return {}
    
    total_samples = sum([num_examples for num_examples, _ in metrics])
    weighted_metrics = {}
    
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        weighted_sum = sum([metric[metric_name] * num_examples for num_examples, metric in metrics])
        weighted_metrics[metric_name] = weighted_sum / total_samples
    
    return weighted_metrics

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()