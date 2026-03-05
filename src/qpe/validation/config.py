from pydantic import BaseModel, ConfigDict 

class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    perplexity_dataset: str = "wikitext"
    max_perplexity_increase_pct: float = 5.0    # User's actual quality budget (%)
    benchmark_tasks: list[str] = ["mmlu", "hellaswag", "gsm8k", "arc_easy"]
    kl_threshold_per_layer: float = 0.1
    eval_batch_size: int = 8

# General + felxible logging system 
# -> we want to log to cli
# -> but also, depending on the env, we might need to log to either a like a file, or like backend 

# Also, some stuff that helps data laoding + model loading,etc