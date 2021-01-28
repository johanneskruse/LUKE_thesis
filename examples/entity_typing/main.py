import json
import logging
import os
from argparse import Namespace

import click
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME
from luke.utils.entity_vocab import MASK_TOKEN

from ..utils import set_seed
from ..utils.trainer import Trainer, trainer_args
from .model import LukeForEntityTyping
from .utils import ENTITY_TOKEN, convert_examples_to_features, DatasetProcessor

import numpy as np
import random

logger = logging.getLogger(__name__)

@click.group(name="entity-typing")
def cli():
    pass


@cli.command()
@click.option("--checkpoint-file", type=click.Path(exists=True))
@click.option("--data-dir", default="data/open_entity", type=click.Path(exists=True))
@click.option("--do-eval/--no-eval", default=True)
@click.option("--do-train/--no-train", default=True)
@click.option("--eval-batch-size", default=32)
@click.option("--num-train-epochs", default=3.0)
@click.option("--seed", default=12)
@click.option("--train-batch-size", default=2)
@click.option("--do-evaluate-prior-train/--no-evaluate-prior-train", default=True)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    task_args.update(common_args)
    args = Namespace(**task_args)

    set_seed(args.seed)

    args.model_config.hidden_dropout_prob = args.hidden_dropout_prob

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})
    args.model_config.vocab_size += 1
    word_emb = args.model_weights["embeddings.word_embeddings.weight"]
    marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    args.model_weights["embeddings.word_embeddings.weight"] = torch.cat([word_emb, marker_emb])
    args.tokenizer.add_special_tokens(dict(additional_special_tokens=[ENTITY_TOKEN]))

    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_config.entity_vocab_size = 2
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    train_dataloader, _, features, label_list = load_examples(args, fold="train")
    num_labels = len(features[0].labels)

    results = {}

    dataset_size = {}
    dataset_size["training_samples"] = len(features)

    if args.do_train:
        model = LukeForEntityTyping(args, num_labels)
        model.load_state_dict(args.model_weights, strict=False)
        model.to(args.device)

        if args.do_evaluate_prior_train:
            dev_results, _, _ = evaluate(args, model, fold="dev")
            results.update({f"dev_{k}_epoch_no_training": v for k, v in dev_results.items()})
        
        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        best_dev_f1 = [-1]
        best_weights = [None]

        def step_callback(model, global_step):
            if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                epoch = int(global_step / num_train_steps_per_epoch - 1)
                dev_results, _, _ = evaluate(args, model, fold="dev")
                args.experiment.log_metrics({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}, epoch=epoch)
                results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})
                tqdm.write("dev: " + str(dev_results))

                if dev_results["f1"] > best_dev_f1[0]:
                    if hasattr(model, "module"):
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                    else:
                        best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                    best_dev_f1[0] = dev_results["f1"]
                    results["best_epoch"] = epoch

                model.train()
                

        trainer = Trainer(
            args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps, step_callback=step_callback
        )
        
        _, global_step, average_loss, training_loss = trainer.train()

    if args.do_train and args.local_rank in (0, -1):
        logger.info("Saving the model checkpoint to %s", args.output_dir)
        torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))

    if args.local_rank not in (0, -1):
        return {}

    model = None
    torch.cuda.empty_cache()

    if args.do_eval:
        
        evaluation_predict_label = {"label_list": label_list, "dev": {}, "test": {}}
        
        model = LukeForEntityTyping(args, num_labels)
        if args.checkpoint_file:
            model.load_state_dict(torch.load(args.checkpoint_file, map_location="cpu"))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
        model.to(args.device)
        
        for eval_set in ("dev", "test"):
            output_file = os.path.join(args.output_dir, f"{eval_set}_predictions.jsonl")
            result_dict, sample_size, evaluation_predict_label[eval_set] = evaluate(args, model, eval_set, output_file)
            results.update({f"{eval_set}_{k}": v for k, v in result_dict.items()})
            dataset_size[f"{eval_set}_samples"] = sample_size

    if args.do_train:
        # Print results: 
        logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
        
        # Removing model bin: 
        if not args.save_model: 
            os.remove(os.path.join(args.output_dir, "pytorch_model.bin"))
            logger.info("Removing pytorch_model.bin")
        
        # Adding training losses and experimental configurations information: 
        results["experimental_configurations"] = {
                                                "log_parameters": {p.name: getattr(args, p.name) for p in run.params}, 
                                                "model_config": vars(args.model_config)
                                                }
        results["experimental_configurations"]["log_parameters"]["global_step"]     = global_step
        results["experimental_configurations"]["log_parameters"]["average_loss"]    = average_loss
        results["experimental_configurations"]["log_parameters"]["output_dir"]      = args.output_dir
        results["experimental_configurations"]["log_parameters"].update(dataset_size)
        if args.train_frac_size > 1.0: 
            results["experimental_configurations"]["log_parameters"]["train_constructed"] = "with_replacement"
        else:
            results["experimental_configurations"]["log_parameters"]["train_constructed"] = "without_replacement"
        results["evaluation_predict_label"] = evaluation_predict_label
        results["training_loss"] = training_loss

        # Save and output final json file with all information: 
        args.experiment.log_metrics(results)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f)
    else: 
        logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
        args.experiment.log_metrics(results)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f)

    return results


def evaluate(args, model, fold="dev", output_file=None, write_all=False):
    dataloader, _, features, label_list = load_examples(args, fold=fold)
    model.eval()

    all_logits = []
    all_labels = []
    for batch in tqdm(dataloader, desc=fold):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            logits = model(**inputs)

        logits = logits.detach().cpu().tolist()
        labels = batch["labels"].to("cpu").tolist()

        all_logits.extend(logits)
        all_labels.extend(labels)

    all_predicted_indexes = []
    all_label_indexes = []
    
    # Idea with all_predicted_prob was to collect logits and probability for prediction. Change in plan I will collect all raw data in results file 
    # all_predicted_prob = []
    # def logit_to_prob(logits):
    #     # Should not be a softmax as here.
    #     for i in logits:
    #         softmax.append(np.exp(i) / sum(np.exp(logits)))
    #     return softmax

    # Logit select all with > 0 (~ 50% chance) 
    for logits, labels in zip(all_logits, all_labels):
        all_predicted_indexes.append([i for i, v in enumerate(logits) if v > 0])
        all_label_indexes.append([i for i, v in enumerate(labels) if v > 0])
        # all_predicted_prob.append([logit_to_prob(logits)])

    if write_all:
        if not os.path.exists(args.output_dir + "/all_files"):
            os.mkdir(args.output_dir + "/all_files")
        with open(os.path.join(args.output_dir, "all_files", 'all_predicted_indexes.txt'), 'w') as outfile:
            json.dump(all_predicted_indexes, outfile)
        with open(os.path.join(args.output_dir, "all_files",'all_label_indexes.txt'), 'w') as outfile:
            json.dump(all_label_indexes, outfile)
        with open(os.path.join(args.output_dir, "all_files", 'all_logits.txt'), 'w') as outfile:
            json.dump(all_logits, outfile)
        with open(os.path.join(args.output_dir, "all_files", 'all_labels.txt'), 'w') as outfile:
            json.dump(all_labels, outfile)
        with open(os.path.join(args.output_dir, "all_files", 'label_list.txt'), 'w') as outfile:
            json.dump(label_list, outfile)
        # with open(os.path.join(args.output_dir, "all_files", 'all_predicted_prob.txt'), 'w') as outfile:
        #     json.dump(all_predicted_prob, outfile)

    if output_file:
        with open(output_file, "w") as f:
            for predicted_indexes, label_indexes in zip(all_predicted_indexes, all_label_indexes):
                data = dict(
                    predictions=[label_list[ind] for ind in predicted_indexes],
                    labels=[label_list[ind] for ind in label_indexes]
                    # prediction_highest=[label_list[pred_prob.index(max(pred_prob))] for pred_prob in prediction_prob],
                    # prediction_probability=[max(pred_prob) for pred_prob in prediction_prob],
                    # prediction_probability_all=[pred_prob for pred_prob in prediction_prob],
                    # prediction_logits_all=[logit for logit in logits_all],
                )
                f.write(json.dumps(data) + "\n")

    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for predicted_indexes, label_indexes in zip(all_predicted_indexes, all_label_indexes):
        num_predicted_labels += len(predicted_indexes)
        num_gold_labels += len(label_indexes)
        num_correct_labels += len(frozenset(predicted_indexes).intersection(frozenset(label_indexes)))

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0

    recall = num_correct_labels / num_gold_labels
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Save raw predicted logits and true labels
    evaluation_predict_label = {"predict_logits": all_logits, 
                            "true_labels": all_labels}

    return dict(precision=precision, recall=recall, f1=f1), len(all_labels), evaluation_predict_label


def load_examples(args, fold="train"):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = DatasetProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length)

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        return dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            labels=torch.tensor([o.labels for o in batch], dtype=torch.long),
        )

    if fold in ("dev", "test"):
        dataloader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(features, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, label_list