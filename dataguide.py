import random
import json
import torch
import torch.nn as nn
import transformers
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration, GPTNeoXForCausalLM, AutoModelForCausalLM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import os

from datatuner_dataset import *
from metrics import *
from custom_models import *

import sys
sys.path.append('/content/data2text_nlg')
import ser_calculator



class DataGuide:
    def __init__(self, model_name, save_path, theta={}, model_type="enc-dec", current_dataset = "e2e", use_consistency_loss=False, use_sf_loss=False, use_dcs_loss=False):
        """
        Arguments:
        - model_name : string --> pretrained model from huggingface
        - save_path : save path for models and model predictions and losses
        - theta : dict --> hyperparameters of the model
        - model_type : string --> model architecture {enc-dec, dec}
        - current_dataset : string --> dataset name {viggo, e2e, webnlg}
        """
        self.use_consistency_loss=use_consistency_loss
        self.use_sf_loss=use_sf_loss
        self.use_dcs_loss=use_dcs_loss
        if self.use_sf_loss:
            self.loss_suffix = "_sf"
        elif self.use_dcs_loss:
            self.loss_suffix = "_dcs"
        else:
            self.loss_suffix = ""
        if model_type=="enc-dec":
            self.Model_Extractor = AutoModelForSeq2SeqLM
            self.Tokenizer_Extractor = AutoTokenizer
        elif model_type=="dec":
            if "llama" in model_name:
                self.Model_Extractor = AutoModelForCausalLM
                self.Tokenizer_Extractor = AutoTokenizer
            elif "pythia"in model_name:
                self.Model_Extractor = GPTNeoXForCausalLM
                self.Tokenizer_Extractor = AutoTokenizer
        else:
            self.Model_Extractor = AutoModelForSeq2SeqLM
            self.Tokenizer_Extractor = AutoTokenizer
        self.save_path = save_path
        self.model_name = model_name
        if "/" in self.model_name:
            self.model_name_for_save = self.model_name.split("/")[-1]
        else:
            self.model_name_for_save = self.model_name
        self.base_model = self.Model_Extractor.from_pretrained(model_name)
        self.tokenizer = self.Tokenizer_Extractor.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #self.tokenizer.add_special_tokens({"additional_special_tokens":["<data>", "<text>"]})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.current_dataset = current_dataset
        self.model = DatatunerModel(
                    self.base_model,
                    self.tokenizer,
                    device=self.device,
                    current_dataset=self.current_dataset,
                    model_type=self.model_type,
                    use_consistency_loss=self.use_consistency_loss,
                    use_sf_loss=self.use_sf_loss,
                    use_dcs_loss=self.use_dcs_loss)

        if theta!={}:
            if "lr" in theta:
                self.lr = theta["lr"]
            else:
                self.lr = 1e-4
            if "weight_decay" in theta:
                self.weight_decay = theta["weight_decay"]
            else:
                self.weight_decay = 0
            if "batch_size" in theta:
                self.batch_size = theta["batch_size"]
            else:
                self.batch_size = 8
            if "num_epochs" in theta:
                self.num_epochs = theta["num_epochs"]
            else:
                self.num_epochs = 5
        else:
            self.lr = 1e-4
            self.weight_decay = 0
            self.batch_size = 8
            self.num_epochs = 5
        print(self.model_name)
        print(self.model_name_for_save)

    def load_model(self, epoch=None):
        """
        - epoch : int --> epoch suffix
        """
        if epoch!=None:
            path = f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_{epoch}epoch.pth"
        else:
            path = f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}.pth"
        print(path)
        # with open(path, 'r') as json_file:
        #     model_weights_json = json.load(json_file)
        #     print(model_weights_json)

        # # Convert JSON data back to PyTorch tensors
        # model_state_dict = {key: torch.tensor(value) for key, value in model_weights_json.items()}

        # Load model with the converted state_dict
        self.base_model = self.Model_Extractor.from_pretrained(self.model_name)
        self.base_model.load_state_dict(torch.load(path))
        self.model = DatatunerModel(
                    self.base_model,
                    self.tokenizer,
                    device=self.device,
                    current_dataset=self.current_dataset,
                    model_type=self.model_type,
                    use_consistency_loss=self.use_consistency_loss,
                    use_sf_loss=self.use_sf_loss,
                    use_dcs_loss=self.use_dcs_loss)

    def import_dataset(self, dataset_type="train", split="25"):
        """
        - dataset_type: string --> dataset division {train, test}
        - split: string --> desired split {25, 50, 75}
        """
        # Insert your dataset paths here!
        
        if self.current_dataset == "viggo":
            train_path = "drive/Shareddrives/HLT-2023/viggo-train.csv"
            val_path = "drive/Shareddrives/HLT-2023/viggo-valid.csv"
            test_path = "drive/Shareddrives/HLT-2023/viggo-test.csv"
        elif self.current_dataset == "e2e":
            train_path = "drive/Shareddrives/HLT-2023/e2e_training.csv"
            val_path = "drive/Shareddrives/HLT-2023/e2e_devset.csv"
            test_path = "drive/Shareddrives/HLT-2023/e2e_tests.csv"
        elif self.current_dataset == "webnlg":
            train_path = "drive/Shareddrives/HLT-2023/web_nlg_train.csv"
            val_path = "drive/Shareddrives/HLT-2023/web_nlg_dev.csv"
            test_path = "drive/Shareddrives/HLT-2023/web_nlg_test.csv"

        if dataset_type == "train":
            df = pd.read_csv(train_path)
        elif dataset_type == "val":
            df = pd.read_csv(val_path)
        elif dataset_type=="dev":
            df = concatenated_df = pd.concat([pd.read_csv(train_path), pd.read_csv(val_path)], ignore_index=True)
        elif dataset_type=="test":
            df = pd.read_csv(test_path)

        data = list(df["mr"])
        text = list(df["ref"])
        raw_dataset = [{"data": a, "text": b} for a, b in zip(data, text)]
        if self.model_type == "dec":
            current_dataset = DatatunerDatasetDecOnly(raw_dataset, self.tokenizer, dataset_name=self.current_dataset, dataset_type=dataset_type)
        else:
            current_dataset = DatatunerDatasetEncDec(raw_dataset, self.tokenizer, dataset_name=self.current_dataset)

        return current_dataset

    def get_dataloader(self, current_dataset, batch=None): 
        if batch == None:
            batch = self.batch_size
        dataset_loader = DataLoader(current_dataset, batch_size=batch, shuffle=True)
        return dataset_loader
    


    def training(self, train_loader, val_loader = None, load_training=False, current_epoch=1):
        """
        Arguments:
        - train_loader : data loader object from pytorch --> data loader for training
        - val_loader : data loader object from pytorch (Optional) --> data loader for evaluation
        - load_training : bool --> if True load the model on the current_epoch
        - current_epoch : int --> epoch of the last saved model
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-8, weight_decay= self.weight_decay)
        epoch = 0
        step = 0 # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
        last_avg_bleu = 0
        last_ser = 100
        
        if load_training:
            with open(f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_{current_epoch}epoch_loss.json") as file:
                LOSS = json.load(file)
        # else:
            # with open(f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_{current_epoch}epoch_loss.json") as file:
            #         LOSS = json.load(file)
            epoch = int(len(LOSS)/len(train_loader))
            last_losses = [np.array(LOSS[i * len(LOSS) // epoch: (i + 1) * len(LOSS) // epoch]).mean() for i in range(epoch)]
            # Don't train if already trained
            if epoch>3:
                if np.array(last_losses[:-3]).mean() < last_losses[-1] or round(last_losses[-2], 2) == round(last_losses[-1], 2):
                    print("=== Already Trained: don't need more Training ===")
                    return
            self.load_model(epoch=epoch)
        else: ### Added this condition for working properly
            last_losses = []
            LOSS=[]
        self.model.to(self.device)
        num_train = len(train_loader.dataset)
        while epoch < self.num_epochs:
            loss_avg = 0
            epoch += 1
            # log.info(f">>>> Starting epoch {epoch}")
            self.model.train()
            with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
                for batch_num, batch in enumerate(train_loader):
                    #* forward
                    loss, logits = self.model(batch)
                    #* backward
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    optimizer.step()
                    # scheduler.step()
                    #* log info
                    batch_size = len(batch["source_input_ids"])
                    step += batch_size
                    progress_bar.update(batch_size)
                    loss_val = loss.item() # get the item since loss is a tensor
                    loss_avg+=loss_val
                    LOSS.append(loss_val)
                    progress_bar.set_postfix(epoch=epoch, loss=loss_val)
                    del logits
                loss_avg=loss_avg/len(train_loader)
                last_losses.append(loss_avg)
            # Early stopping
            if epoch>3:
                if np.array(last_losses[:-3]).mean() < loss_avg or round(last_losses[-2], 2) <= round(last_losses[-1], 2):
                    print("=== !Stopping condition! ===")
                    print("=== Saving final Model ===")
                    if not os.path.exists(f"{self.save_path}{self.current_dataset}"):
                        os.makedirs(f"{self.save_path}{self.current_dataset}")
                    else:
                        pass
                    torch.save(self.model.model.state_dict(), f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}.pth")
                    with open(f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_loss.json", 'w') as file:
                        # Utilizza json.dump() per salvare la lista nel file come JSON
                        json.dump(LOSS, file)
                    return
            if not os.path.exists(f"{self.save_path}{self.current_dataset}"):
                os.makedirs(f"{self.save_path}{self.current_dataset}")
            else:
                pass
            torch.save(self.model.model.state_dict(), f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_{epoch}epoch.pth")
            with open(f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_{epoch}epoch_loss.json", 'w') as file:
                # Utilizza json.dump() per salvare la lista nel file come JSON
                json.dump(LOSS, file)
            
        
            
    def compute_loss(self,val_loader):
        """
        - val_loader : data loader object from pytorch --> data loader for evaluation
        """
        loss_avg=0
        num_val = len(val_loader.dataset)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(val_loader):
                #* forward
                loss, logits = self.model(batch)

                #* log info
                batch_size = len(batch["source_input_ids"])
                progress_bar.update(batch_size)
                loss_val = loss.item() # get the item since loss is a tensor
                loss_avg+=loss_val
                progress_bar.set_postfix(loss=loss_val)
                del logits
            loss_avg=loss_avg/len(val_loader)
        print("CELoss", loss_avg)
        return loss_avg
    
    def evaluate(self, val_loader, training=None):
        """
        - val_loader : data loader object from pytorch --> data loader for evaluation
        - training : string --> for saving purpouse, specifies if the val_loader is a training set or not
        """
        if training!=None:
            train_label="_train"
        else:
            train_label=""
        #* evaluate
        # epoch = 0
        # step = 0 # number of total examples we have done (will be epoch * len(data_set) at end of each epoch)
        last_avg_bleu = 0
        last_ser = 100
        intermediate_predictions = []
        num_val = len(val_loader.dataset)
        original_data_inputs = [] # collecting at each iteration even if it is not needed
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad(), tqdm(total=num_val) as progress_bar:
            for batch_num, batch in enumerate(val_loader):
                #* generate for token matches
                generated_ids = self.model.inference(batch)
                #* save for qualitative analysis
                data_inputs = self.tokenizer.batch_decode(batch["source_input_ids"], skip_special_tokens=True)
                text_targets = self.tokenizer.batch_decode(batch["target_input_ids"], skip_special_tokens=True)
                # conservation of the rights predicition thanks to a regex
                # myre = r"(^.+[\.|\!|\?])"
                # data_inputs = []
                # text_targets = []
                # data_inputs = [re.search(myre, i).group(1) for i in data_inputs_raw if re.search(myre, i)]
                # text_targets = [re.search(myre, i).group(1) for i in text_targets_raw if re.search(myre, i)]
                outputs_decoded = np.array(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
                # they are not divided into batch so we reorder them from (batch size * beam size, sequence size) to (batch, beam, sequence)
                output_beams_decoded = outputs_decoded.reshape(-1, 5)
                # outputs_decoded_no_special_tokens = outputs_decoded_no_special_tokens.reshape(-1, 5)
                # if batch_num % 5 == 0:
                #     log.info(f"gen dec: {outputs_decoded.shape}\n{outputs_decoded[0][0]}")
                original_data_inputs.extend(batch["original_data"])
                current_predictions = list(zip(
                    data_inputs, text_targets, output_beams_decoded
                    ))
                intermediate_predictions.extend(current_predictions)
                batch_size = len(batch["source_input_ids"])
                progress_bar.update(batch_size)
        if not os.path.exists(f"{self.save_path}{self.current_dataset}/"):
            os.makedirs(f"{self.save_path}{self.current_dataset}")
            
        else:
            pass
        np.save(f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}{self.loss_suffix}_predictions{train_label}.npy", intermediate_predictions)
        # with open(f"{self.save_path}{self.current_dataset}/{self.model_name_for_save}_predictions.json", 'w') as file:
        #     json.dump(intermediate_predictions, file)
        #* compute the average BLEU score
        #self.compute_metrics(intermediate_predictions)
        #current_corpus_bleu_score = corpus_level_bleu(intermediate_predictions)
        # log.info(f"BLEU at end of epoch {epoch}: {current_corpus_bleu_score:.3f}")
        #* compute SER
        #current_ser_values = corpus_level_ser(original_data_inputs, intermediate_predictions, self.current_dataset)
        #return current_corpus_bleu_score#, current_ser_values

    def compute_metrics(self, intermediate_predictions, dataset=""):
        #* compute the average BLEU score
        current_corpus_bleu_score = corpus_level_bleu(intermediate_predictions)
        # log.info(f"BLEU at end of epoch {epoch}: {current_corpus_bleu_score:.3f}")
        #* compute SER
        #current_ser_values = corpus_level_ser(original_data_inputs, intermediate_predictions, self.current_dataset)
        current_chrf_values = corpus_level_chrf(intermediate_predictions)
        # current_ter_values = corpus_level_ter(intermediate_predictions)
        current_meteor_values = corpus_level_meteor(intermediate_predictions)
        current_rogue_values = corpus_level_rogue(intermediate_predictions)
        grouped_items = group_inputs_and_outputs_by_data(intermediate_predictions)
        references, hypotheses = extract_refs_and_hyps_from_grouped_items_like_datatuner(grouped_items)
        try:
            current_ser_values = corpus_level_ser(references[0], intermediate_predictions, dataset_name=dataset)[2]*100
        except:
            current_ser_values=0
        
        bleu = round(current_corpus_bleu_score,2)
        chrf = round(current_chrf_values, 2)
        # ter = round(current_ter_values, 2)
        meteor = round(current_meteor_values, 2)
        rogue = round(current_rogue_values, 2)
        ser = round(current_ser_values, 2)

        print(" BLEU & chrF & METEOR & ROGUE & SER \\\\", "\n", f"{bleu} & {chrf} & {meteor} & {rogue} & {ser} \\\\")
        
        return bleu, chrf, meteor, rogue, ser
