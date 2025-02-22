 import wandb

wandb.init(project="test_project")
wandb.log({"test_metric": 1})
wandb.finish()
