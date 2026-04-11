
from datasets import load_dataset

# dataset = load_dataset("gamma-lab-umd/MMAU-test-mini")
dataset = load_dataset("/workspace/datasets/MMAU-test-mini")

print(dataset)
# print(dataset["test"][0])
print(dataset["test"].select(range(1)))

# 本地
# from huggingface_hub import snapshot_download

# local_dir = snapshot_download(
#     repo_id="gamma-lab-umd/MMAU-test-mini",
#     repo_type="dataset",
#     local_dir="/data/fengwenhao/datasets/MMAU-test-mini"
# )

# print(local_dir)