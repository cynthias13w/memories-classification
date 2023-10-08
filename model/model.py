import time
from colorama import Fore, Style

print(Fore.BLUE + "\nLoading Transformers 🤗..." + Style.RESET_ALL)
start = time.perf_counter()

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

end = time.perf_counter()
print(f"\n✅ All modules loaded ({round(end - start, 2)}s)")


def initialize_model(model_name_or_path, num_labels, id2label=None, label2id=None):
    """
    Initialize a pretrained model for sequence classification.

    Args:
        model_name_or_path (str): The name or path of the pretrained model to load.
        num_labels (int): The number of labels for sequence classification.
        id2label (dict, optional): A mapping from label IDs to label names.
        label2id (dict, optional): A mapping from label names to label IDs.

    Returns:
        AutoModelForSequenceClassification: The initialized model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    print("✅ Model initialized")
    return model


def initialize_training_args(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
):
    """
    Initialize TrainingArguments with default or customized values.

    Args:
        output_dir (str): The directory where model checkpoints and outputs will be saved.
        learning_rate (float): The learning rate for training.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        num_train_epochs (int): Number of training epochs.
        weight_decay (float): Weight decay for regularization.
        evaluation_strategy (str): "steps" or "epoch" - when to perform evaluation during training.
        save_strategy (str): "steps" or "epoch" - when to save model checkpoints during training.
        load_best_model_at_end (bool): Whether to load the best model checkpoint at the end of training.
        push_to_hub (bool): Whether to push the trained model to the Hugging Face Model Hub.

    Returns:
        TrainingArguments: Initialized TrainingArguments object.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        push_to_hub=push_to_hub,
    )
    return training_args


def train_model(
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics,
    training_args=None
):
    """
    Train a model using the Trainer class from Hugging Face Transformers.

    Args:
        model (AutoModelForSequenceClassification): The model to be trained.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing text.
        data_collator (DataCollator): Data collator for batch processing.
        compute_metrics (Callable): A function to compute evaluation metrics.
        training_args (TrainingArguments, optional): Training arguments for the Trainer.

    Returns:
        Trainer: The initialized Trainer instance.
    """
    print(Fore.BLUE + "\nChecking Training Arguments... 🔍" + Style.RESET_ALL)
    if training_args is None:
        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

    return trainer
