from datasets import load_dataset

# Load CUAD-QA dataset
dataset = load_dataset("theatticusproject/cuad-qa")


import json

dataset_dict = {
    "train": dataset["train"].to_list(),
    "test": dataset["test"].to_list()
}

# Save to a JSON file
with open("cuad_qa_dataset.json", "w") as f:
    json.dump(dataset_dict, f, indent=4)

print("Dataset saved as cuad_qa_dataset.json")

