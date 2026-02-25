from pydantic import BaseModel, ConfigDict 

class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    perplexity_dataset: str = "wikitext"
    max_perplexity_increase_pct: float = 5.0    # User's actual quality budget (%)
    benchmark_tasks: list[str] = ["mmlu", "hellaswag", "gsm8k", "arc_easy"]
    kl_threshold_per_layer: float = 0.1
    eval_batch_size: int = 8