from datasets import load_dataset
from torchvision import transforms
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import numpy as np
from evaluate import load
import torch
import torch.nn as nn
import wandb
import os
from transformers.trainer_callback import TrainerCallback

def freeze_all_but_classifier(model):
    for name, param in model.named_parameters():
        # classifier or head (depends on model implementation)
        if "classifier" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


class FreezeUnfreezeCallback(TrainerCallback):
    def __init__(self, freeze_epochs=5):
        self.freeze_epochs = freeze_epochs

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Freeze all but classifier at the start
        print(f"Freezing all layers except the classification head for {self.freeze_epochs} epochs.")
        freeze_all_but_classifier(model)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        # Unfreeze after freeze_epochs
        if state.epoch == self.freeze_epochs:
            print("Unfreezing all layers for fine-tuning.")
            unfreeze_all(model)


# Initialize wandb
PROJECT_NAME = 'galaxy-classification-experiments'
EXPERIMENT_NAME = 'large_vit_freez'
os.environ["WANDB_PROJECT"] = PROJECT_NAME
wandb.login(key='15bd1ad8362e2b1317c26ab056e2df6ced3d4d21')


# Load dataset
dataset_name = "matthieulel/galaxy10_decals"
galaxy_dataset = load_dataset(dataset_name)

# Define class names based on the dataset card
class_names = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"
]

# Create a dictionary for easy lookup
label2name = {i: name for i, name in enumerate(class_names)}
name2label = {name: i for i, name in enumerate(class_names)}

num_classes = len(class_names)
print(f"\nNumber of classes: {num_classes}")
print("Class names:", class_names)


## Define image processor
model_name_or_path = 'google/vit-large-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)


## Prepare dataset
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

# Define augmentations appropriate for galaxy images
def galaxy_augmentation(example_batch):
    # Create augmentation pipeline
    augmentation = transforms.Compose([
        transforms.RandomRotation(180),  # Galaxies can appear at any orientation
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Slight zoom variation
        transforms.RandomHorizontalFlip(),  # Horizontal flip is valid for galaxies
        transforms.RandomVerticalFlip(),   # Vertical flip is also valid for galaxies
        # Avoid hue/saturation changes as they alter astronomical meaaning
    ])
    
    # Apply augmentations to training images only
    augmented_images = []
    for image in example_batch['image']:
        augmented_images.append(augmentation(image))
    
    # Process with ViT processor
    inputs = processor(augmented_images, return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

# Apply augmentation only to training set
train_ds = galaxy_dataset["train"].with_transform(galaxy_augmentation)
test_ds = galaxy_dataset["test"].with_transform(transform)  # Keep your original transform for test set

# Update your prepared_ds
prepared_ds = {
    "train": train_ds,
    "test": test_ds
}

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


## Define metric
metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


## Load model
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(class_names),
    id2label=label2name,
    label2id=name2label,
    ignore_mismatched_sizes=True
)


## Define helpers
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Custom callback to save the best model after each epoch
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.best_metric = float('inf')
        
    def on_epoch_end(self, args, state, control, **kwargs):
        # Evaluate model
        metrics = self.trainer.evaluate()
        eval_loss = metrics.get("eval_loss", float('inf'))
        
        # Save model if it's the best so far
        if eval_loss < self.best_metric:
            self.best_metric = eval_loss
            self.trainer.save_model(f"{args.output_dir}/best_model_epoch_{state.epoch}")
            wandb.log({"best_model_epoch": state.epoch, "best_eval_loss": eval_loss})
        
        return control


## Set up training hypterparameters
training_args = TrainingArguments(
    output_dir=f'./{EXPERIMENT_NAME}',
    per_device_train_batch_size=8,
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save at the end of each epoch
    num_train_epochs=8,
    fp16=True,
    logging_steps=10,
    learning_rate=1e-5,  # Lower learning rate for fine-tuning large model
    weight_decay=0.01,   # Explicit weight decay for AdamW
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='wandb',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch",  # Explicitly use AdamW optimizer
    warmup_steps=500,     # Add warmup steps
    lr_scheduler_type="cosine",  # Cosine decay works well with AdamW
)

# Initialize wandb run with hyperparameters
wandb.init(
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    config={
        "model": model_name_or_path,
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "optimizer": training_args.optim,
        "lr_scheduler": training_args.lr_scheduler_type,
        "loss": "ce_loss",
    }
)


# Run training loop
trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=processor,
)


# Add the custom callback to save best model after each epoch
save_best_callback = SaveBestModelCallback(trainer)
trainer.add_callback(save_best_callback)

freeze_unfreeze_callback = FreezeUnfreezeCallback(freeze_epochs=5)
trainer.add_callback(freeze_unfreeze_callback)

# Train the model
train_results = trainer.train()

# Save the final model
trainer.save_model(f"./{EXPERIMENT_NAME}/final_model")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Finish wandb run
wandb.finish()