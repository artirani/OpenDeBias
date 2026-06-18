import os
# Restrict CUDA to only see GPU 4
os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,5,6"
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMultipleChoice, TrainingArguments, EvalPrediction, AutoTokenizer, AutoConfig, AutoModelForMultipleChoice
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from adapters import AdapterTrainer, AdapterFusionConfig, Fuse, init
import wandb
from huggingface_hub import login

# Ensure PyTorch uses GPU 4
torch.cuda.empty_cache()
# torch.cuda.set_device(4)
selected_gpus = list(range(torch.cuda.device_count()))
device = torch.device(f"cuda:{selected_gpus[0]}")  # Move model to the first available GPU
print(torch.cuda.current_device())

# Ensure CUDA is available
print('Is CUDA available:', torch.cuda.is_available())

# Initialize Weights & Biases
wandb.login(key="YOUR_WANDB_TOKEN")
wandb.init(project="Fusion_training_deberta_v3_large_race_custom_loss", name="fusion_layer_training_and_testing")

# Hugging Face API Key (use your actual API key)
hf_api_key = "YOUR_HUGGINGFACE_TOKEN"
login(token=hf_api_key)

fusion_save_dir = "/storage/nihar/LLM_biasness/Fusion_training/deberta_v3_large_race_custom_loss/Fusion"
fusion_layer_dir = "/storage/nihar/LLM_biasness/Fusion_training/deberta_v3_large_race_custom_loss/Fusion/Fusion_layer"
os.makedirs(fusion_save_dir, exist_ok=True)
os.makedirs(fusion_layer_dir, exist_ok=True)

# def compute_accuracy(p: EvalPrediction):
#     preds = np.argmax(p.predictions, axis=1)
#     return {"accuracy": (preds == p.label_ids).mean()}

# Function to compute accuracy
def compute_accuracy(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=-1)

    if preds.ndim < 3:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    else:
        label_ids = p.label_ids
        total = 0
        num_correct = 0
        for idx, ex_labels in enumerate(label_ids):
            ex_labels[ex_labels == -100] = 1
            total += 1
            if (ex_labels == preds[idx]).all():
                num_correct += 1
        return {'accuracy': num_correct / total}

import torch.nn.functional as F

# Define a custom AdapterTrainer with a modified loss function
class CustomAdapterTrainer(AdapterTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # pop labels before passing inputs to model

        # check for existance of variable and set if not exists
        try:
            self.curr_batch_start
        except:
            self.curr_batch_start = 0

        curr_batch_start = self.curr_batch_start  # this variable is to keep track of batch start index

        context_conditions = self.train_dataset["context_condition"][curr_batch_start:curr_batch_start + labels.shape[0]] # context_condition of batch
        answer_infos = self.train_dataset["answer_info"][curr_batch_start:curr_batch_start + labels.shape[0]] # answer_info os batch

        self.curr_batch_start = curr_batch_start + labels.shape[0] # increase batch start index for next batch process

        outputs = model(**inputs)  # pass input_ids and attention_masks to model
        logits = outputs.logits  # take logits of output

        # Compute Cross Entropy Loss (default for classification)
        ce_loss = F.cross_entropy(logits, labels, reduction="none")  # by default reduction='mean', which takes mean of sample losses as batch loss
        total_loss = ce_loss
        ce_loss = ce_loss.clone().detach()
        kl_loss = []

        # Log the loss to wandb
        # wandb.log({"ce_loss": torch.mean(ce_loss).item()})  # ce_loss must be a batch loss to be logged on wandb

        for idx, context_condition in enumerate(context_conditions):
            # check for ambiguos data and calculate KL_divergence
            if context_condition == "ambig":
                answer_info = answer_infos[idx]
                curr_logits = logits[idx]

                # identify choice with 'unknown' as an answer
                unknown_option = 0
                for i in range(len(answer_info)):
                    if answer_info[f"ans{i}"][-1] == "unknown":
                        unknown_option = i
                        break
            
                # consider logits of options other than 'unknown' one
                answer_logits = None
                if unknown_option == 0:
                    answer_logits = curr_logits[1:]  # Extract logits for ans1 and ans2
                elif unknown_option == 1:
                    answer_logits = torch.stack((
                        curr_logits[0], # Extract logits for ans0
                        curr_logits[2]  # Extract logits for ans2
                    ))
                else:
                    answer_logits = curr_logits[:2]  # Extract logits for ans0 and ans1
            
                answer_probs = F.softmax(answer_logits, dim=-1)  # Apply Softmax
            
                # Uniform distribution over two choices: [0.5, 0.5]
                uniform_dist = torch.full_like(answer_probs, 0.5)
            
                curr_kl_loss = F.kl_div(
                    answer_probs.log(),  # Take log of softmax output
                    uniform_dist, 
                    reduction="batchmean"
                )

                kl_loss.append(curr_kl_loss.item())
            
                # wandb.log({"kl_loss": kl_loss.item()})
            
                # Final loss: Cross Entropy + (λ * KL-Divergence)
                lambda_kl = 0.1  # You can tune this weight factor
                total_loss[idx] = ce_loss[idx] + lambda_kl * curr_kl_loss

        ce_loss = torch.mean(ce_loss).item()
        if len(kl_loss) > 0:
            kl_loss = sum(kl_loss) / len(kl_loss)
        else:
            kl_loss = 0.0
        total_loss = torch.mean(total_loss)  # take average for converting into batch loss

        # wandb.log({"total_loss": total_loss.item()})

        # check for existance of variable and set if not exists
        try:
            self.train_losses
        except:
            self.train_losses = {
                "ce_loss": [],
                "kl_loss": [],
                "total_loss": []
            }
        try:
            self.val_losses
        except:
            self.val_losses = {
                "ce_loss": [],
                "kl_loss": [],
                "total_loss": []
            }

        # Store loss manually
        if self.model.training:
            self.train_losses["ce_loss"].append(ce_loss)  # Training loss
            self.train_losses["kl_loss"].append(kl_loss)  # Training loss
            self.train_losses["total_loss"].append(total_loss.item())  # Training loss
        else:
            self.val_losses["ce_loss"].append(ce_loss)  # Validation
            self.val_losses["kl_loss"].append(kl_loss)  # Validation
            self.val_losses["total_loss"].append(total_loss.item())  # Validation

        return (total_loss, outputs) if return_outputs else total_loss

# Adapter categories and corresponding names
bias_categories = ["age", "gender_identity", "race_ethnicity", "religion", "disability_status"]
adapter_names = [f"{category}_adapter" for category in bias_categories]

# Split sizes
train_size = 500
val_size = 200

# Load BBQ dataset for each category
datasets = {category: load_dataset("Elfsong/BBQ", split=category) for category in bias_categories}


split_data = {}
for category in bias_categories:
    dataset = datasets[category]
    num_samples = len(dataset)

    # split_data[category] = DatasetDict({
    #     "train": dataset.select(range(min(num_samples, train_size))),
    #     "validation": dataset.select(range(train_size, min(num_samples, train_size + val_size))),
    #     "test": dataset.select(range(train_size + val_size, num_samples)),
    # })

    ambg_data = dataset.filter(lambda sample: sample["context_condition"] == "ambig")
    disambg_neg_data = dataset.filter(lambda sample: sample["context_condition"] == "disambig" and sample["question_polarity"] == "neg")
    disambg_nonneg_data = dataset.filter(lambda sample: sample["context_condition"] == "disambig" and sample["question_polarity"] == "nonneg")

    def get_question_index(sample):
        if sample["context_condition"] == "disambig" and sample["question_polarity"] == "neg":
            return sample

    disambg_neg_data_index = dataset.filter(get_question_index)

    disambg_neg_data_index = disambg_neg_data_index.remove_columns(['category', 'example_id', 'question_polarity', 'context_condition', 'context', 'question', 'ans0', 'ans1', 'ans2', 'answer_info', 'answer_label', 'target_label', 'additional_metadata'])

    disambg_data = None

    for question_index in set(disambg_neg_data_index["question_index"]):
        concatenated_dataset = concatenate_datasets([
            disambg_neg_data.filter(lambda sample: sample["question_index"] == question_index),
            disambg_nonneg_data.filter(lambda sample: sample["question_index"] == question_index)
        ])

        if disambg_data is not None:
            disambg_data = concatenate_datasets([
                disambg_data,
                concatenated_dataset
            ])
        else:
            disambg_data = concatenated_dataset

    train_set = concatenate_datasets([
        ambg_data.select(range(0, train_size // 2)),
        disambg_data.select(range(0, train_size // 2))
    ]).shuffle(seed=42)

    val_set = concatenate_datasets([
        ambg_data.select(range(train_size // 2, (train_size + val_size) // 2)),
        disambg_data.select(range(train_size // 2, (train_size + val_size) // 2))
    ]).shuffle(seed=42)

    test_set = concatenate_datasets([
        ambg_data.select(range((train_size + val_size) // 2, len(ambg_data))),
        disambg_data.select(range((train_size + val_size) // 2, len(disambg_data)))
    ]).shuffle(seed=42)

    split_data[category] = DatasetDict({
        "train": train_set,
        "validation": val_set,
        "test": test_set
    })


# from datasets import Dataset, DatasetDict
# import random

# # Load datasets for all categories
# datasets = {category: load_dataset("Elfsong/BBQ", split=category) for category in bias_categories}

# # Initialize the split_data dictionary
# split_data = {}

# # Iterate through each category
# for category in bias_categories:
#     # Load the dataset for the current bias category
#     data = datasets[category]
#     category_data = data

#     # Step 1: Split the dataset into two halves
#     half_size = len(category_data) // 2

#     # Initialize D1 and D2
#     D1 = []
#     D2 = []

#     # Process the first half: ambig -> D1, disambig -> D2
#     for i in range(0, half_size, 2):
#         item1 = category_data[i]
#         item2 = category_data[i + 1]

#         if item1['question'] == item2['question'] and item1['context_condition'] != item2['context_condition']:
#             D1.append(item1) if item1['context_condition'] == 'ambig' else D2.append(item1)
#             D2.append(item2) if item2['context_condition'] == 'disambig' else D1.append(item2)

#     # Process the second half: disambig -> D1, ambig -> D2
#     for i in range(half_size, len(category_data) - 1, 2):
#         item1 = category_data[i]
#         item2 = category_data[i + 1]

#         if item1['question'] == item2['question'] and item1['context_condition'] != item2['context_condition']:
#             D1.append(item2) if item2['context_condition'] == 'disambig' else D2.append(item2)
#             D2.append(item1) if item1['context_condition'] == 'ambig' else D1.append(item1)

#     # Verify sizes of D1 and D2 for each category
#     print(f"Category: {category}")
#     print(f"Size of D1: {len(D1)}")
#     print(f"Size of D2: {len(D2)}")

#     # Shuffle D2 for randomness
#     random.shuffle(D2)

#     # Step 2: Split D2 into validation and test sets
#     d2_val_size = int(0.1 * len(D2))  # 10% for validation
#     d2_test_size = int(0.4 * len(D2))  # 40% for test

#     d2_val = D2[:d2_val_size]
#     d2_test = D2[d2_val_size:d2_val_size + d2_test_size]

#     # Convert splits to Hugging Face Dataset objects and store in split_data
#     split_data[category] = DatasetDict({
#         "train": Dataset.from_dict({key: [item[key] for item in D1] for key in D1[0]}),  # Convert D1 to Dataset
#         "validation": Dataset.from_dict({key: [item[key] for item in d2_val] for key in d2_val[0]}),
#         "test": Dataset.from_dict({key: [item[key] for item in d2_test] for key in d2_test[0]})
#     })

#     # Verify final sizes of splits for each category
#     print(f"Category: {category}")
#     print(f"Train data size: {len(D1)}")
#     print(f"Validation data size: {len(d2_val)}")
#     print(f"Test data size: {len(d2_test)}\n")

all_categories = [
    "age", "disability_status", "gender_identity", "nationality", "physical_appearance", 
    "race_ethnicity", "race_x_gender", "race_x_ses", "religion", "ses", "sexual_orientation"
]

# For all other categories, use the entire dataset as test
for category in all_categories:
    if category not in split_data:
        dataset = load_dataset("Elfsong/BBQ", split=category)
        split_data[category] = DatasetDict({"test": dataset})

# Function to merge multiple Hugging Face datasets
def merge_datasets(datasets_list):
    merged_data = {key: [] for key in datasets_list[0].column_names}
    for dataset in datasets_list:
        for key in dataset.column_names:
            merged_data[key].extend(dataset[key])
    return Dataset.from_dict(merged_data)

# Merge train and validation datasets
train_datasets = [split_data[cat]["train"] for cat in bias_categories if "train" in split_data[cat]]
train_dataset = merge_datasets(train_datasets)

val_datasets = [split_data[cat]["validation"] for cat in bias_categories if "validation" in split_data[cat]]
val_dataset = merge_datasets(val_datasets)

# Tokenizer and model setup
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("artianand/deberta-v3-large-race")

def encode_batch(batch):
    """Encodes the input data for the model."""
    max_length = 512
    input_ids = []
    attention_masks = []
    
    for context, question, ans0, ans1, ans2 in zip(batch["context"], batch["question"], batch["ans0"], batch["ans1"], batch["ans2"]):
        choices = [
            f"{context} {question}{ans0}",
            f"{context} {question}{ans1}",
            f"{context} {question}{ans2}"
            # f"{question} {ans0} {ans1} {ans2} {context} {ans0}",
            # f"{question} {ans0} {ans1} {ans2} {context} {ans1}",
            # f"{question} {ans0} {ans1} {ans2} {context} {ans2}"

        ]
        encoded = tokenizer(
            choices,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Ensure sequence length is consistent
        assert encoded["input_ids"].shape[1] == max_length, f"Input sequence length mismatch: {encoded['input_ids'].shape[1]} != {max_length}"
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": batch["answer_label"]
    }

# Preprocess datasets
train_dataset = train_dataset.map(encode_batch, batched=True)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

val_dataset = val_dataset.map(encode_batch, batched=True)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model configuration
# hf_username = "Shweta-singh"  # Your Hugging Face username
hf_username = "artianand"

# config = RobertaConfig.from_pretrained("roberta-base")
# model = RobertaForMultipleChoice.from_pretrained("roberta-base", config=config)

config = AutoConfig.from_pretrained("artianand/deberta-v3-large-race")
model = AutoModelForMultipleChoice.from_pretrained("artianand/deberta-v3-large-race", config=config)

model = torch.nn.DataParallel(model, device_ids=selected_gpus, output_device=selected_gpus[0])

init(model.module)

# Load adapters
for adapter_name in adapter_names:
    repo_id = f"{hf_username}/{adapter_name}_deberta_v3_large_race_batch_8_custom_loss"
    model.module.load_adapter(repo_id, load_as=adapter_name)
    model.module.set_active_adapters(adapter_name)

# Define fusion setup
model.module.add_adapter_fusion(adapter_names)
adapter_setup = Fuse(*adapter_names)
model.module.set_active_adapters(adapter_setup)
model.module.train_adapter_fusion(adapter_setup)


# Print adapter summary
print(model.module.adapter_summary())

# Print full model architecture
print("\n### Model Architecture ###")
print(model.module)
print("### End of Model Architecture ###\n")

num_epochs = 7
batch_size = 4
training_args = TrainingArguments(
    learning_rate=0.0003,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    output_dir=fusion_save_dir,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    logging_dir=f"{fusion_save_dir}/logs",
)
# Trainer for fusion
trainer = CustomAdapterTrainer(
    model=model.module,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_accuracy
)

# Train and evaluate fusion
trainer.train()
trainer.save_model()
model.module.save_adapter_fusion(fusion_layer_dir, adapter_setup)
# model.load_adapter_fusion(fusion_layer_dir, set_active=True)

# print(model.adapter_summary())

# import csv

# # Function to calculate and save accuracies for full, ambiguous, and disambiguated parts for all categories
# def evaluate_and_save_results_per_category(split_data, trainer, output_path):
#     results = []

#     for category in all_categories:
#         print(f"Evaluating category: {category}")
#         test_dataset = split_data[category]["test"]
#         print(f"total instance is {category} : {len(test_dataset)}")
#         test_dataset = test_dataset.map(encode_batch, batched=True)
#         test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#         # Evaluate full test dataset
#         full_accuracy = trainer.evaluate(eval_dataset=test_dataset)["eval_accuracy"]

#         # Split into ambiguous and disambiguated
#         ambig_data = test_dataset.filter(lambda x: x["context_condition"] == "ambig")
#         print(f"ambig_data instance is {category} : {len(ambig_data)}")
#         disambig_data = test_dataset.filter(lambda x: x["context_condition"] == "disambig")
#         print(f"disambig_data instance is {category} : {len(disambig_data)}")

#         ambig_accuracy = trainer.evaluate(eval_dataset=ambig_data)["eval_accuracy"] if len(ambig_data) > 0 else None
#         disambig_accuracy = trainer.evaluate(eval_dataset=disambig_data)["eval_accuracy"] if len(disambig_data) > 0 else None

#         # Append results
#         results.append({"Category": category, "Subset": "Full", "Accuracy": full_accuracy})
#         if ambig_accuracy is not None:
#             results.append({"Category": category, "Subset": "Ambiguous", "Accuracy": ambig_accuracy})
#         if disambig_accuracy is not None:
#             results.append({"Category": category, "Subset": "Disambiguated", "Accuracy": disambig_accuracy})

#     # Save results to CSV
#     with open(output_path, "w", newline="") as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=["Category", "Subset", "Accuracy"])
#         writer.writeheader()
#         writer.writerows(results)

#     print(f"Results saved to {output_path}")

# # Path to save the results
# output_file = os.path.join(fusion_save_dir, "subset_wise_accuracy_results_mh_false.csv")

# # Evaluate and save results for all categories
# evaluate_and_save_results_per_category(split_data, trainer, output_file)


wandb.finish()
