# DataGuideColab
The objective of this project is to implement the method proposed in the work of Leonardo Calamita for small LLMs with decoder-only architecture in the data-to-text task.

The implementation is based on the classes present in the thesis of Calamita to which the methods relating to the management of decoder-only models were modified and some methods were added to group the demo file of the main repository: train.py and evaluate.py. Finally, the set of methods was grouped into a DataGuide class and some original methods were modified to work in the Colab environment, which is used for accessing computational resources remotely.

The LLM Pythia was used for the task at different sizes (70m, 160m, 410m and 1.4b parameters) to test the permissibility of the task and the approach used. Pythia variants were tested on three data-to-text datasets: Viggo, WebNLG and E2E. Viggo and E2E were chosen because of the considerations in the main repository of Calamita regarding the quality of their annotations. In addition, Viggo was choosen also because of its small size: to test the generalisation capabilities of the model with little data and because of the limited computational effort required.

Each dataset was divided into two parts: a Test set and a Development set. The Development set was initially divided into two further parts: Training set and Validation set, but was then brought together for model training, where it was used as a training dataset. In fact, the model selection phase was not performed on the Validation set because it was decided to keep the hyperparameters used in the original work of Calamita, unchanged due to the computational effort that would have required a complete search for hyperparameters.

## Example

```
data_guide = DataGuide(model_name="EleutherAI/pythia-70m-deduped", theta={"lr": 1e-4, "batch_size": 8, "num_epochs" : 10}, model_type="dec", current_dataset="e2e", save_path="/content/drive/Shareddrives/HLT-2023/models/", use_sf_loss=True)
train_dataset = data_guide.import_dataset(dataset_type = "train")
train_loader = data_guide.get_dataloader(train_dataset)
data_guide.training(train_loader, load_training=False)
```
## Colaborators

This study is the fruit of collaboration between Silvio Calderaro, Vittorio Mussin and Vincent Sarbach-Pulicani, students in the "Digital Humanities" master's programme at the University of Pisa in 2023. This repository is supplemented by a report written as part of Giuseppe Attardi's course entitled "Human Language Technology".

### requirements
- github.com/LeoCal4/data2text-nlg
- github.com/KaijuML/parent
- transformers
- sentencepiece
- sacrebleu
- rouge-metric
- nltk ("punkt")
- nltk ('wordnet')
