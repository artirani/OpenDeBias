import os
# Restrict CUDA to only see GPU 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # Set this BEFORE importing torch
import sys
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
import numpy as np
import pandas as pd
import adapters
from transformers import RobertaTokenizer, RobertaConfig, TrainingArguments, AutoTokenizer, AutoConfig
from adapters import AdapterTrainer, RobertaAdapterModel, SeqBnConfig
from transformers import RobertaForMultipleChoice, EvalPrediction, AutoModelForMultipleChoice
from huggingface_hub import login
import wandb
import torch
import torch.nn as nn
from math import ceil
import random

# Ensure PyTorch uses GPU 4
torch.cuda.empty_cache()
# torch.cuda.set_device(4)
selected_gpus = list(range(torch.cuda.device_count()))
device = torch.device(f"cuda:{selected_gpus[0]}")  # Move model to the first available GPU
print(torch.cuda.current_device())

# Ensure CUDA is available
print('Is CUDA available:', torch.cuda.is_available())

# Hugging Face API key (replace <your_hf_api_key> with your actual key)
hf_api_key = "YOUR_HUGGINGFACE_TOKEN"

# Log in to Hugging Face Hub
login(token=hf_api_key)

# Directories for saving the models and adapters
# output_dir = "/home/nihar.sahoo/test/src/Adapter_training/deberta_v3_large_race/exp1/Adapters_tuning/"
# save_dir = "/home/nihar.sahoo/test/src/Adapter_training/deberta_v3_large_race/exp1/Adapters/"
output_dir = "/storage/nihar/LLM_biasness/Adapter_training/deberta_v3_large_race_custom_loss/Adapters_tuning/"
save_dir = "/storage/nihar/LLM_biasness/Adapter_training/deberta_v3_large_race_custom_loss/Adapters/"

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

# Ensure that the directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# Suppress unnecessary logging
import transformers
transformers.logging.set_verbosity_error()

# Setting parameters
bias_categories = ["age", "gender_identity", "race_ethnicity", "religion", "disability_status"]
adapter_names = [f"{category}_adapter" for category in bias_categories]



##------------------------------------------------------------------------------

# DATA SET 1:::: 250 AMBIG AND 250 DISAMBIG IN TRAIN

##------------------------------------------------------------------------------

# Define fixed sizes for train, validation, and test sets
train_size = 500
val_size = 200

# Load BBQ dataset by category
datasets = {category: load_dataset("Elfsong/BBQ", split=category) for category in bias_categories}

# Split dataset into train, validation, and test sets for each category
split_data = {}
for category in bias_categories:
    dataset = datasets[category]
    
    total_samples = len(dataset)
    
    # Ensure there are enough samples for the fixed splits
    if total_samples < train_size + val_size:
        raise ValueError(f"Not enough samples in category '{category}' to satisfy the fixed train and val sizes.")

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

    # # Create splits
    # train_set = dataset.select(range(0, train_size)).shuffle(seed=42) 
    # val_set = dataset.select(range(train_size, train_size + val_size)).shuffle(seed=42)  
    # test_set = dataset.select(range(train_size + val_size, total_samples)).shuffle(seed=42)  
    
    split_data[category] = DatasetDict({
        "train": train_set,
        "validation": val_set,
        "test": test_set
    })




##------------------------------------------------------------------------------

# DATA SET 2:::: 50% of data train (no same example of ambig and disambig)

##------------------------------------------------------------------------------



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

##------------------------------------------------------------------------------

# DATA SET 3:::: 250 DISAMBIG AND 150 AMBIG IN TRAIN  shuffled

##------------------------------------------------------------------------------


# # Define fixed sizes for train, validation, and test sets
# train_size_disambig = 250
# train_size_ambig = 150
# val_size = 200

# # Load BBQ dataset by category
# datasets = {category: load_dataset("Elfsong/BBQ", split=category) for category in bias_categories}

# # Split dataset into train, validation, and test sets for each category
# split_data = {}
# for category in bias_categories:
#     dataset = datasets[category]
    
#     # Extract the first 500 samples
#     first_500 = dataset.select(range(500))
    
#     # Separate ambiguous and disambiguated examples
#     disambig_examples = [ex for ex in first_500 if ex['context_condition'] == 'disambig']
#     ambig_examples = [ex for ex in first_500 if ex['context_condition'] == 'ambig']
    
#     # Ensure there are enough samples for the desired split
#     if len(disambig_examples) < train_size_disambig or len(ambig_examples) < train_size_ambig:
#         raise ValueError(f"Not enough samples in category '{category}' to satisfy the desired train sizes.")
    
#     # Select the required number of examples
#     selected_disambig = disambig_examples[:train_size_disambig]
#     selected_ambig = ambig_examples[:train_size_ambig]
    
#     # Combine and shuffle the training set
#     train_set = selected_disambig + selected_ambig
#     random.shuffle(train_set)
    
#     # Convert to Dataset format
#     train_set = Dataset.from_dict(train_set)
    
#     # Identify the indices of examples used in the training set
#     used_indices = set(ex['id'] for ex in train_set)  # Assuming each example has a unique 'id'
    
#     # Get the remaining examples from the first 500 not in the training set
#     remaining_100 = [ex for ex in first_500 if ex['id'] not in used_indices]
#     remaining_100 = Dataset.from_dict(remaining_100)
    
#     # Create validation and test sets
#     remaining_data = dataset.select(range(500, len(dataset)))
#     val_set = remaining_data.select(range(val_size)).shuffle(seed=42)
#     test_set = remaining_data.select(range(val_size, len(remaining_data))).shuffle(seed=42)
    
#     # Add the remaining 100 examples to the test set
#     test_set = Dataset.from_dict(test_set + remaining_100)
    
#     split_data[category] = DatasetDict({
#         "train": train_set,
#         "validation": val_set,
#         "test": test_set
#     })


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("artianand/deberta-v3-large-race")

def encode_batch(batch):
    """Encodes the input data for the model."""
    max_length = 512
    input_ids = []
    attention_masks = []
    option_token_start_idx = []
    option_token_end_idx = []
    
    for context, question, ans0, ans1, ans2 in zip(batch["context"], batch["question"], batch["ans0"], batch["ans1"], batch["ans2"]):
        choices = [
            f"{context} {question}{ans0}",
            f"{context} {question}{ans1}",
            f"{context} {question}{ans2}"
            # f"{question}\n(a){ans0}(b){ans1}(c){ans2}\n{context}\n{ans0}",
            # f"{question}\n(a){ans0}(b){ans1}(c){ans2}\n{context}\n{ans1}",
            # f"{question}\n(a){ans0}(b){ans1}(c){ans2}\n{context}\n{ans2}"
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

        # # For generation
        # curr_option_token_end_idx = np.array(encoded["attention_mask"]).sum(-1)
        # # heuristic, because tokenizers can be weird
        # curr_option_token_start_idx = curr_option_token_end_idx - np.array([
        #     len(tokenizer.tokenize(x))
        #     for x in [ans0, ans1, ans2]
        # ])
        # # noinspection PyUnresolvedReferences
        # assert (curr_option_token_start_idx < curr_option_token_end_idx).all()
        # option_token_start_idx.append(torch.tensor(curr_option_token_start_idx))
        # option_token_end_idx.append(torch.tensor(curr_option_token_end_idx))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        # "option_token_start_idx": torch.stack(option_token_start_idx),
        # "option_token_end_idx": torch.stack(option_token_end_idx),
        "labels": batch["answer_label"]
    }

# Preprocess datasets
for category in bias_categories:
    for split in ["train", "validation", "test"]:
        split_data[category][split] = split_data[category][split].map(encode_batch, batched=True)
        split_data[category][split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Model configuration
config = AutoConfig.from_pretrained("artianand/deberta-v3-large-race")
model = AutoModelForMultipleChoice.from_pretrained("artianand/deberta-v3-large-race", config=config)

# Use DataParallel to parallelize the model over available GPUs
# model = nn.DataParallel(model)
model.to(device)

adapters.init(model)

# Adapter configuration using SeqBnConfig
adapter_config = SeqBnConfig(
    mh_adapter=True,
    output_adapter=True,
    reduction_factor=16,
    non_linearity="relu",
    original_ln_before=True,
    original_ln_after=True
)

# Add and activate adapters
for adapter_name in adapter_names:
    model.add_adapter(adapter_name, config=adapter_config)
    model.train_adapter(adapter_name)
    model.set_active_adapters(adapter_name)

# Print adapter summary
print(model.adapter_summary())

# Training arguments
num_epochs = 5
batch_size = 8
training_args = TrainingArguments(
    learning_rate=0.0003,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    output_dir=output_dir,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    logging_dir=f"{output_dir}/logs",
)

from torch.utils.data import DataLoader

def print_model_outputs(test_data, model, tokenizer, batch_size=4):
    """
    Prints the outputs of the model for a specific category on the test dataset.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Create a DataLoader for the test data
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, batch in enumerate(test_dataloader):
            # if batch_idx >= 1:  # Only process the first batch to limit output size
            #     break

            # Move data to the appropriate device (GPU if available)
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Predictions (argmax over logits for each choice)
            logits = outputs.logits  # Shape: (batch_size, num_choices)
            predictions = torch.argmax(logits, dim=1)  # Shape: (batch_size,)

            # Decode the input choices for easier visualization
            decoded_choices = [
                [
                    tokenizer.decode(choice_ids, skip_special_tokens=True)
                    for choice_ids in example_ids
                ]
                for example_ids in input_ids
            ]

            # # Print the results
            # print(f"\nBatch {batch_idx + 1}:")
            # for i, (choices, prediction, label) in enumerate(zip(decoded_choices, predictions, labels)):
            #     print(f"\nExample {i + 1}:")
            #     # print(f"Choices:")
            #     # for idx, choice in enumerate(choices):
            #     #     print(f"  Choice {idx}: {choice}")
            #     print(f"Predicted Answer: Choice {prediction.item()}")
            #     print(f"Correct Answer: Choice {label.item()}")

            with open(f"{output_dir}/evaluation_examples.txt", "a") as file:
                file.writelines(f"\nBatch {batch_idx + 1}:")
                for i, (choices, prediction, label) in enumerate(zip(decoded_choices, predictions, labels)):
                    file.writelines(f"\nExample {i + 1}:")
                    file.writelines(f"Choices:")
                    for idx, choice in enumerate(choices):
                        file.writelines(f"  Choice {idx}: {choice}")
                    file.writelines(f"Predicted Answer: Choice {prediction.item()}")
                    file.writelines(f"Correct Answer: Choice {label.item()}")


# hf_username = "Shweta-singh"
hf_username = "artianand"

wandb.login(key="356e5ae17bdf5cbcb2705020fba90f7c2dfbe6a3")

# Train and evaluate each adapter
for category, adapter_name in zip(bias_categories, adapter_names):
    # Initialize Weights & Biases for each category
    wandb.init(project=f"Adapter_tuning_deberta_v3_large_race_custom_loss_batch_{batch_size}", name=f"{category}_bias_adapter_tuning")
    
    print('loading trainer')
    trainer = CustomAdapterTrainer(  # Use the custom trainer
        model=model,
        args=training_args,
        train_dataset=split_data[category]["train"],
        eval_dataset=split_data[category]["validation"],  # Validation during training
        compute_metrics=compute_accuracy,
    )
    print(f"Training adapter for {category}...")
    trainer.train()

    # losses = {
    #     "Epoch": [],
    #     "Train CE Loss": [],
    #     "Train KL Loss": [],
    #     "Train Total Loss": [],
    #     "Validation CE Loss": [],
    #     "Validation KL Loss": [],
    #     "Validation Total Loss": []
    # }

    # train_start_index = 0
    # val_start_index = 0
    # num_train_losses = ceil(len(split_data[category]["train"]) / batch_size)
    # num_val_losses = ceil(len(split_data[category]["validation"]) / batch_size)
    # for epoch in range(num_epochs):
    #     train_ce_loss = trainer.train_losses["ce_loss"][train_start_index:(train_start_index + num_train_losses)]
    #     train_kl_loss = trainer.train_losses["kl_loss"][train_start_index:(train_start_index + num_train_losses)]
    #     train_total_loss = trainer.train_losses["total_loss"][train_start_index:(train_start_index + num_train_losses)]
    #     val_ce_loss = trainer.val_losses["ce_loss"][val_start_index:(val_start_index + num_train_losses)]
    #     val_kl_loss = trainer.val_losses["kl_loss"][val_start_index:(val_start_index + num_train_losses)]
    #     val_total_loss = trainer.val_losses["total_loss"][val_start_index:(val_start_index + num_train_losses)]

    #     train_start_index = train_start_index + num_train_losses
    #     val_start_index = val_start_index + num_val_losses

    #     losses["Epoch"].append(epoch)
    #     losses["Train CE Loss"].append(sum(train_ce_loss) / len(train_ce_loss)),
    #     losses["Train KL Loss"].append(sum(train_kl_loss) / len(train_kl_loss)),
    #     losses["Train Total Loss"].append(sum(train_total_loss) / len(train_total_loss)),
    #     losses["Validation CE Loss"].append(sum(val_ce_loss) / len(val_ce_loss)),
    #     losses["Validation KL Loss"].append(sum(val_kl_loss) / len(val_kl_loss)),
    #     losses["Validation Total Loss"].append(sum(val_total_loss) / len(val_total_loss))

    # losses = pd.DataFrame(losses)
    # losses = wandb.Table(dataframe=losses)
    # wandb.log({f"{category} Loss Curve": wandb.plot.line(losses, x="Epoch", y=["Train CE Loss", "Train KL Loss", "Train Total Loss", "Validation CE Loss", "Validation KL Loss", "Validation Total Loss"], title=f"{category} Training vs Validation Loss")})

    trainer.save_model()

    test_data = split_data[category]["test"]

    # Example usage for a specific category
    print_model_outputs(test_data, model, tokenizer, batch_size)

    metrics = trainer.evaluate(eval_dataset=test_data)  # Final evaluation on test set
    print(f"Test evaluation metrics for {category} adapter:", metrics)
    model.save_adapter(f"{save_dir}/{adapter_name}", adapter_name)

    # Push the adapter to Hugging Face Hub
    repo_id = f"{hf_username}/{adapter_name}_deberta_v3_large_race_custom_loss_batch_{batch_size}"
    print(f"Pushing {adapter_name} to Hugging Face Hub at {repo_id}...")
    model.push_adapter_to_hub(adapter_name=adapter_name, repo_id=repo_id)
    print(f"{adapter_name} successfully pushed to the Hugging Face Hub repository: {repo_id}")

    # Finish Weights & Biases run for the category
    wandb.finish()