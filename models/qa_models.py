import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.optim import AdamW, Adam
import pandas as pd
import torch
import numpy as np
import os
import logging
import pickle

logger = logging.getLogger(__name__)


class YesNoDataSet(Dataset):

    def __init__(self, csv_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)
        print(self.df.columns)
        self.max_length = max_length
        self.yes_questions = self.df[self.df.label == 'Yes']
        self.no_questions = self.df[self.df.label == 'No']
        self.questions = self.df.question.values
        self.labels = self.df.label.values

    def __len__(self):
        return len(self.questions)

    def pad(self, sample):
        sample_len = len(sample)
        padding_len = self.max_length - sample_len
        pad_sample = torch.hstack([torch.LongTensor(sample), torch.LongTensor([self.tokenizer.pad_token_id] * padding_len)])
        return pad_sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        question = self.questions[idx]
        encoded_question = self.tokenizer.encode_plus(question, return_tensors="pt", max_length=self.max_length,
                                                      padding='max_length')
        encoded_label = self.tokenizer.encode_plus(self.labels[idx] + " </s>", return_tensors="pt")
        input_ids = encoded_question.input_ids.squeeze()
        attention_mask = encoded_question.attention_mask.squeeze()
        label = encoded_label.input_ids.squeeze()

        input_ids = self.pad(input_ids)
        attention_mask = self.pad(attention_mask)
        sample = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
        return sample


class YesNoQuestionAnswering(pl.LightningModule):
    def __init__(self, model, tokenizer, config, log_name="YesNoQALog.log"):
        super().__init__()
        #logging.basicConfig(filename=log_name, encoding='utf-8', level=logging.DEBUG)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_training_loss = []
        self.model_validation_loss = []

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   labels=labels)
        loss = output[0]
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, attention_mask, labels)
        self.model_training_loss.append(loss.item())
        logging.info(f'train_loss: {loss}')
        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss = self(input_ids, attention_mask, labels)
        self.model_validation_loss.append(loss.item())
        logging.info(f'validation_loss: {loss}')
        tensorboard_logs = {"val_loss": loss}
        self.log("val_loss", loss)
        return {"val_loss": loss, "log": tensorboard_logs}

    def train_dataloader(self):
        dataset = YesNoDataSet(csv_path=self.config.get("train_data"), tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config.get("batch_size"), shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = YesNoDataSet(csv_path=self.config.get("dev_data"), tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config.get("batch_size"), shuffle=True)
        return dataloader

    def configure_optimizers(self):
        return AdamW(params=self.parameters(), lr=self.config.get("lr"))


def train_model(config):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="checkpoint",
                                                       prefix="checkpoint", monitor="val_loss", mode="min",
                                                       save_top_k=1)
    train_params = dict(
        gpus=config.get("gpus"),
        max_epochs=config.get("max_epochs", 1),
        checkpoint_callback=checkpoint_callback
    )
    logging.info(config)
    tokenizer = T5Tokenizer.from_pretrained(config.get("model_name"), cache_dir="../cache/")
    model = T5ForConditionalGeneration.from_pretrained(config.get("model_name"), cache_dir="../cache/")
    model = YesNoQuestionAnswering(tokenizer=tokenizer, model=model, config=config)
    if config.get("checkpoint", None):
        logging.info("Use checkpoint")
        checkpoint = torch.load(config.get("checkpoint"), map_location=torch.device(config.get("device")))
        model.load_state_dict(checkpoint["state_dict"])

    trainer = Trainer(**train_params)
    trainer.fit(model)
    with open("pickle/training_loss.pkl", "wb") as f:
        pickle.dump(model.model_training_loss, f)

    with open("pickle/validation_loss.pkl", "wb") as f:
        pickle.dump(model.model_validation_loss, f)


def test_model(config, output_path):
    print("Test Model")
    print(config.get("test_data"))
    tokenizer = T5Tokenizer.from_pretrained(config.get("model_name"), cache_dir="../cache/")
    model = T5ForConditionalGeneration.from_pretrained(config.get("model_name"), cache_dir="../cache/")
    model = YesNoQuestionAnswering(tokenizer=tokenizer, model=model, config=config)

    if config.get("checkpoint", None):
        checkpoint = torch.load(config.get("checkpoint"), map_location=torch.device(config.get("device")))
        model.load_state_dict(checkpoint["state_dict"])
    print("Load checkpoint")
    model.eval()
    data_set = YesNoDataSet(csv_path=config.get("test_data", "csv/test_questions.csv"), tokenizer=tokenizer)
    data_loader = DataLoader(data_set, batch_size=config.get("batch_size"), shuffle=True)

    accuracy = 0.0
    count = 0.0
    questions = []
    model_answers = []
    true_answers = []
    for batch in data_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = model.model.generate(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              max_length=2)
        for idx, answer in enumerate(output):
            model_answer = tokenizer.decode(answer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            true_answer = tokenizer.decode(labels[idx], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            question = tokenizer.decode(input_ids[idx], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            model_answers.append(model_answer)
            true_answers.append(true_answer)
            questions.append(question)
            print("Question:", tokenizer.decode(input_ids[idx], skip_special_tokens=True, clean_up_tokenization_spaces=True))
            print("Model answer:", model_answer)
            print("True answer:", true_answer)
            print("-" * 20)
            if true_answer == model_answer:
                accuracy += 1
            count += 1
    result_df = pd.DataFrame.from_dict({"question": questions, "model_answer": model_answers, "true_answer": true_answers})
    result_df.to_csv(output_path)
    print("Accuracy:", accuracy / count)


if __name__ == "__main__":
    config = {
        "train": False,
        "model_name": "t5-base",
        "gpus": 1,
        "max_epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 16,
        #"train_data": "csv/trained_merged_no_animals.csv",
        "train_data": "csv/train_questions.csv",
        "test_data": "csv/animals_dont_live_underwater_questions.csv",
        #"dev_data": "csv/conceptnet_dev.csv",
        "dev_data": "csv/val_questions.csv",
        "lr": 1e-4,
        "checkpoint": "checkpoint/checkpoint-epoch=8-step=22994.ckpt",
        "checkpoint/t5_large": None
    }
    print("Start Run")
    print("- Config -")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    if config.get("train", True):
        train_model(config)
    else:
        test_files = ["csv/animals_have_a_beak", "csv/animals_have_horns", "csv/animals_have_fins", "csv/animals_have_scales",
               "csv/animals_have_wings", "csv/animals_have_feathers", "csv/animals_have_fur",
               "csv/animals_have_hair", "csv/animals_live_underwater", "csv/animals_can_fly",
               "csv/animals_dont_have_a_beak", "csv/animals_dont_have_horns", "csv/animals_dont_have_fins",
               "csv/animals_dont_have_scales", "csv/animals_dont_have_wings", "csv/animals_dont_have_feathers",
               "csv/animals_dont_have_fur", "csv/animals_dont_have_hair", "csv/animals_dont_live_underwater",
               "csv/animals_cant_fly", "csv/animals", "csv/train_questions"]

#        test_files = ["csv/train"]
        for f in test_files:
            config["test_data"] = f"{f}_questions.csv"
            test_model(config, config["test_data"].replace(".csv", "_result.csv").replace("csv/", "csv/results/"))
