import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer,AutoConfig
import random
import numpy as np
import faiss
import time
from datetime import datetime, timedelta
import uuid

# other modules
from timetrack import TimingContext 
from retriever import EvidenceRetriever
from dataset_load import DatasetReader, UnifiedDataset, DATASET_CONFIGS

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# Model and training configurations
ST_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1.5'
# ST_MODEL_NAME = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'

# ST_MODEL_NAME = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
# ST_MODEL_NAME = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
# ST_MODEL_NAME = 'all-MiniLM-L12-v2'
# ST_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
# ST_MODEL_NAME = "NovaSearch/stella_en_1.5B_v5"
# ST_MODEL_NAME = 'Alibaba-NLP/gte-modernbert-base'

CLASSIFIER_MODEL_NAME = 'deberta'
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
EPOCH_SIZE = 1
# EPOCH_SIZE = 5
# DATASET_NAME = 'LIAR-RAW'
DATASET_NAME = 'RAWFC'
USE_QUANTIZED_MODEL = False

# Set random seeds
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


def generate_run_id():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{random_id}"



class EvidenceClassificationPipeline:
    def __init__(self, dataset_name: str, retriever_model: str = ST_MODEL_NAME,
                 classifier_model: str = CLASSIFIER_MODEL_NAME,
                 device: str = None):
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.timings = {}

        print(f"Initializing pipeline for {dataset_name} dataset")
        print(f"Using device: {self.device}")

        with TimingContext("retriever_initialization", self.timings):
            print(f"Loading retriever model: {retriever_model}")
            self.retriever = EvidenceRetriever(retriever_model, USE_QUANTIZED_MODEL)

        with TimingContext("classifier_initialization", self.timings):
            print(f"Loading classifier model: {classifier_model}")
            self._initialize_classifier(classifier_model)

        self.best_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        self.best_epoch = 0

    def _initialize_classifier(self, classifier_model):
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto"
        }

        if classifier_model == 'deberta':
            model_name = "microsoft/deberta-v3-base"
            # model_name = "microsoft/deberta-v3-large"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       add_prefix_space=True,
                                                       use_fast=True)
            special_tokens = {'additional_special_tokens': ['[SEP]']}
            self.tokenizer.add_special_tokens(special_tokens)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.config['num_labels'],
            ).to(self.device)
            self.classifier.resize_token_embeddings(len(self.tokenizer))

    def process_raw_dataset(self, dataset, split: str = 'train'):
        
        with TimingContext(f"{split}_sentence_collection", self.timings):
            claims = []
            all_sentences = []
            labels = []
            
            for item in tqdm(dataset):
                if 'reports' in item:
                    for report in item['reports']:
                        if 'tokenized' in report:
                            for sent_obj in report['tokenized']:
                                if isinstance(sent_obj, dict) and 'sent' in sent_obj:
                                    all_sentences.append(sent_obj['sent'])

        with TimingContext(f"{split}_index_building", self.timings):
            print("Building FAISS index...")
            self.retriever.build_index(all_sentences)

        with TimingContext(f"{split}_evidence_retrieval", self.timings):
            evidence_texts = []
            for item in tqdm(dataset):
                if not isinstance(item, dict):
                    continue

                claim = item.get('claim')
                label = item.get('label')

                if not claim or not label:
                    continue

                claims.append(claim)
                label = self.config['label_map'][item['label'].lower()]
                labels.append(label)

                evidence = self.retriever.retrieve_evidence(claim, k=10)
                evidence_text = ' '.join([e[2] for e in evidence])
                evidence_texts.append(evidence_text)

        for key, value in self.retriever.timings.items():
            self.timings[f"{split}_{key}"] = value

        return claims, evidence_texts, labels

    def train(self, train_dataset, eval_dataset,
              batch_size: int = BATCH_SIZE, num_epochs: int = EPOCH_SIZE,
              learning_rate: float = LEARNING_RATE, save_dir: str = None):
        
        with TimingContext("total_training_time", self.timings):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.best_epoch = 0
            self.best_metrics = None

            with TimingContext("data_processing", self.timings):
                train_claims, train_evidences, train_labels = self.process_raw_dataset(
                    train_dataset, 'train')
                eval_claims, eval_evidences, eval_labels = self.process_raw_dataset(
                    eval_dataset, 'eval')

                train_data = UnifiedDataset(
                    train_claims, train_evidences, train_labels, self.dataset_name, self.tokenizer)
                eval_data = UnifiedDataset(
                    eval_claims, eval_evidences, eval_labels, self.dataset_name, self.tokenizer)

                g = torch.Generator()
                g.manual_seed(SEED)
                train_loader = DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    generator=g
                )

                eval_loader = DataLoader(
                    eval_data,
                    batch_size=batch_size,
                    shuffle=False,
                    generator=g
                )

            optimizer = torch.optim.AdamW(
                self.classifier.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(train_loader) * num_epochs
            )

            # Training loop
            self.best_f1 = 0
            epoch_timings = {}
            
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                with TimingContext(f"epoch_{epoch+1}_training", epoch_timings):
                    self.classifier.train()
                    total_loss = 0
                    train_predictions = []
                    train_labels = []

                    progress_bar = tqdm(
                        train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
                    for batch in progress_bar:
                        optimizer.zero_grad()
                        inputs = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.classifier(**inputs)
                        loss = outputs.loss

                        preds = torch.argmax(outputs.logits, dim=1)
                        train_predictions.extend(preds.cpu().numpy())
                        train_labels.extend(inputs['labels'].cpu().numpy())

                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        total_loss += loss.item()
                        progress_bar.set_postfix(
                            {'loss': f'{total_loss / (progress_bar.n + 1):.4f}'})

                # Calculate training metrics
                with TimingContext(f"epoch_{epoch+1}_metrics", epoch_timings):
                    train_metrics = {
                        'accuracy': accuracy_score(train_labels, train_predictions),
                        'precision': precision_score(train_labels, train_predictions, average='macro'),
                        'recall': recall_score(train_labels, train_predictions, average='macro'),
                        'f1': f1_score(train_labels, train_predictions, average='macro')
                    }

                    print(f"\nEpoch {epoch + 1} Training Metrics:")
                    for key, value in train_metrics.items():
                        print(f"{key}: {value:.4f}")

                # Validation phase
                with TimingContext(f"epoch_{epoch+1}_validation", epoch_timings):
                    val_metrics = self.evaluate(eval_loader)
                    print(f"\nEpoch {epoch + 1} Validation Metrics:")
                    for key, value in val_metrics.items():
                        print(f"{key}: {value:.4f}")

                # Save best model
                if val_metrics['f1'] > self.best_f1:
                    self.best_f1 = val_metrics['f1']
                    self.best_epoch = epoch + 1
                    self.best_metrics = val_metrics.copy()

                    if save_dir:
                        with TimingContext(f"epoch_{epoch+1}_model_saving", epoch_timings):
                            os.makedirs(save_dir, exist_ok=True)
                            model_path = os.path.join(save_dir, f'epoch_{epoch + 1}')
                            self.classifier.save_pretrained(model_path)
                            self.tokenizer.save_pretrained(model_path)

                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch {epoch + 1} completed in: {str(timedelta(seconds=int(epoch_time)))}")

            self.timings.update(epoch_timings)

            print("\nTraining Summary:")
            print(f"Best performance achieved in epoch {self.best_epoch}:")
            self._print_metrics(self.best_metrics)

    def evaluate(self, eval_loader: DataLoader):
        self.classifier.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.classifier(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(inputs['labels'].cpu().numpy())

        return self._calculate_metrics(all_labels, all_preds)

    def _calculate_metrics(self, labels, predictions):
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='macro'),
            'recall': recall_score(labels, predictions, average='macro'),
            'f1': f1_score(labels, predictions, average='macro')
        }

    def _print_metrics(self, metrics):
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

def main():
    # Generate a unique run ID for this training session
    run_id = generate_run_id()
    print(f"\nStarting new training run with ID: {run_id}")
    
    all_timings = {}
    with TimingContext("total_execution", all_timings):
        start_datetime = datetime.now()
        print(f"\nStarting execution at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        dataset_name = DATASET_NAME
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Must be one of {list(DATASET_CONFIGS.keys())}")

        # Set paths with run_id included
        BASE_DIR = os.getcwd()
        dataset_path = os.path.join(BASE_DIR, "dataset", dataset_name)
        
        save_dir = os.path.join(
            BASE_DIR, 
            f"saved_models/{dataset_name.lower()}_classifier/run_{run_id}"
        )
        os.makedirs(save_dir, exist_ok=True)

        with TimingContext("data_loading", all_timings):
            try:
                print("Loading training data...")
                train_data = DatasetReader.read_dataset(dataset_name, dataset_path, "train")
                
                eval_data = DatasetReader.read_dataset(dataset_name, dataset_path, "val")
                
                test_data = DatasetReader.read_dataset(dataset_name, dataset_path, "test")
                
            except Exception as e:
                print(f"Error loading datasets: {str(e)}")
                raise


        with TimingContext("pipeline_initialization", all_timings):
            pipeline = EvidenceClassificationPipeline(dataset_name)

        with TimingContext("model_training", all_timings):
            pipeline.train(
                train_dataset=train_data,
                eval_dataset=eval_data,
                batch_size=BATCH_SIZE,
                num_epochs=EPOCH_SIZE,
                learning_rate=LEARNING_RATE,
                save_dir=save_dir
            )
            # Add pipeline timings to overall timings
            all_timings.update(pipeline.timings)

        # Final evaluation on test set
        best_model_path = os.path.join(save_dir, f'epoch_{pipeline.best_epoch}')
        
        with TimingContext("model_loading", all_timings):
            try:
                if not os.path.exists(best_model_path):
                    raise FileNotFoundError(f"Best model directory not found: {best_model_path}")
                
                config = AutoConfig.from_pretrained(best_model_path, local_files_only=True)
                pipeline.classifier = AutoModelForSequenceClassification.from_pretrained(
                    best_model_path,
                    config=config,
                    local_files_only=True,
                    trust_remote_code=True
                ).to(pipeline.device)
                
                print(f"Successfully loaded best model from {best_model_path}")
                
            except Exception as e:
                print(f"Error loading best model: {str(e)}")
                print("Continuing with current model state...")
        
        with TimingContext("final_evaluation", all_timings):
            test_claims, test_evidences, test_labels = pipeline.process_raw_dataset(
                test_data, 'test')
            test_dataset = UnifiedDataset(
                test_claims, test_evidences, test_labels, dataset_name, pipeline.tokenizer)
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            test_metrics = pipeline.evaluate(test_loader)

        # Save final results with timings
        results = {
            'run_id': run_id,
            'best_epoch': pipeline.best_epoch,
            'test_metrics': test_metrics,
            'training_complete': True,
            'completion_time': datetime.now().isoformat(),
            'timings': all_timings
        }
        
        results_path = os.path.join(save_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print final results and timings
        print("\nFinal Test Results:")
        print(f"Run ID: {run_id}")
        print(f"Best model was from epoch {pipeline.best_epoch}")
        print("\nTest Set Performance:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")

        # uncomment if needed
        # print("\nTiming Information:")
        # for operation, duration in all_timings.items():
        #     print(f"{operation}: {str(timedelta(seconds=int(duration)))}")

        print("\nConfiguration Details:")
        print(f"Run ID: {run_id}")
        print(f"Retriever model: {ST_MODEL_NAME}")
        print(f"Classifier model: {CLASSIFIER_MODEL_NAME}")
        print(f"Batch Size: {BATCH_SIZE}")
        print(f"Learning Rate: {LEARNING_RATE}")
        print(f"Epochs: {EPOCH_SIZE}")
        print(f"Quantized Model: {USE_QUANTIZED_MODEL}")
        print(f"Results saved to: {results_path}")
        print(f"SEED: {SEED}")

        end_datetime = datetime.now()
        duration = (end_datetime - start_datetime).total_seconds()
        print(f"\nExecution started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Execution completed at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {str(timedelta(seconds=int(duration)))}")

if __name__ == "__main__":
    main()