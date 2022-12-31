import torch
import shutil
import numpy as np

from argparse import Namespace
from pathlib import Path
from typing import Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)

from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from semantic_retrieval.optim import ContrastiveLoss


def save_checkpoint(
    path_: Path,
    state: dict,
    is_best: bool,
    filename="checkpoint.pt",
):
    if not path_.exists():
        path_.mkdir(parents=True)

    torch.save(
        state,
        path_.joinpath(filename).as_posix(),
    )

    if is_best:
        shutil.copy(
            path_.joinpath(filename).as_posix(),
            path_.joinpath("model_best.pt").as_posix(),
        )


def compute_warmup_steps(
    args: Namespace,
    warmup_persentage: float = 1.5,
) -> Namespace:
    args.steps_per_epoch = int(args.batches_per_epoch /
                               args.gradient_accumulation_steps)
    args.num_warmup_steps = args.steps_per_epoch * warmup_persentage
    args.num_training_steps = int(args.steps_per_epoch * args.epochs)
    return args


class ContrastiveLearningTask(object):

    def __init__(
        self,
        name: str,
        args: Namespace,
    ):
        super(ContrastiveLearningTask, self).__init__()
        self.name = name
        self.args = args
        self.global_steps = 0

    @classmethod
    def get_loss_fn(
        cls,
        type: str = "contrastive_loss",
        reduction: str = "none",
    ):
        if type == "contrastive_loss":
            return ContrastiveLoss(reduction=reduction)
        else:
            raise NotImplementedError(f"loss {type} not yet implemented")

    @staticmethod
    def compute_preds(
        logits_per_image: Tensor,
        logits_per_text: Tensor,
    ) -> Tuple[int, int]:
        with torch.no_grad():
            preds_i = torch.argmax(
                logits_per_image,
                dim=-1,
            )
            preds_t = torch.argmax(
                logits_per_text,
                dim=-1,
            )
            return preds_i, preds_t

    @staticmethod
    def compute_correct(
        logits_per_image: Tensor,
        logits_per_text: Tensor,
        targets: Tensor,
    ) -> Tuple[int, int]:
        preds_i, preds_t = ContrastiveLearningTask.compute_preds(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
        )

        n_correct_i = preds_i.eq(targets).sum().item()
        n_correct_t = preds_t.eq(targets).sum().item()
        return n_correct_i, n_correct_t

    def train(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LambdaLR,
        dataloader: DataLoader,
        device,
        **kwargs,
    ) -> Tuple[float, float, float]:
        # setup
        model = model.train()
        optimizer.zero_grad()
        # got loss
        loss_fn = self.get_loss_fn(type=kwargs.get(
            "loss_type",
            "contrastive_loss",
        ))
        loss_fn = loss_fn.to(device)

        # metrics
        total_loss = 0
        n_pred_total = 0
        n_pred_correct_img = 0
        n_pred_correct_text = 0
        steps = 0

        for batch_idx, batch in enumerate(dataloader):
            img_array = batch["img_array"].to(device)
            description_ids = batch["description_ids"].to(device)
            batch_size = img_array.size(0)
            targets = torch.arange(
                0,
                batch_size,
                dtype=torch.long,
                device=img_array.device,
            )

            with torch.set_grad_enabled(True):
                logits_per_image, logits_per_text = model(
                    images=img_array,
                    text_ids=description_ids,
                )

                loss_t = loss_fn(
                    logits_per_image=logits_per_image,
                    logits_per_text=logits_per_text,
                    targets=targets,
                )  # shape [batch_size]
                loss_t = loss_t.mean()

                if self.args.gradient_accumulation_steps > 1:
                    # scale the loss if gradient accumulation is used
                    loss_t = loss_t / self.args.gradient_accumulation_steps

                loss_t.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.args.max_grad_norm,
                )

                # accumulate the gradients
                if batch_idx % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # update metrics
            steps += 1
            n_correct_img_t, n_correct_text_t = self.compute_correct(
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text,
                targets=targets,
            )
            total_loss += loss_t.item()
            n_pred_total += batch_size
            n_pred_correct_img += n_correct_img_t
            n_pred_correct_text += n_correct_text_t

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                print(f"batch : {batch_idx}")

            if (steps / self.args.gradient_accumulation_steps) == self.args.steps_per_epoch:
                break

        steps /= self.args.gradient_accumulation_steps
        total_loss = total_loss / steps
        accuracy_img = n_pred_correct_img / n_pred_total
        accuracy_text = n_pred_correct_text / n_pred_total
        self.global_steps += int(steps)
        return dict(
            train_loss=total_loss,
            train_accuracy_img=accuracy_img,
            train_accuracy_text=accuracy_text,
        )

    def evaluation(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device,
        **kwargs,
    ):
        model = model.eval()
        # got loss
        loss_fn = self.get_loss_fn()
        loss_fn = loss_fn.to(device)

        total_loss = 0
        steps = 0
        preds_imgs = []
        preds_texts = []
        labels = []

        for batch_idx, batch in enumerate(dataloader):
            
            if batch_idx % 50 == 0:
                print(f"eval batch : {batch_idx}")
            
            img_array = batch["img_array"].to(device)
            description_ids = batch["description_ids"].to(device)
            batch_size = img_array.size(0)
            targets = torch.arange(
                0,
                batch_size,
                dtype=torch.long,
                device=img_array.device,
            )

            with torch.set_grad_enabled(False):
                logits_per_image, logits_per_text = model(
                    images=img_array,
                    text_ids=description_ids,
                )

                loss_t = loss_fn(
                    logits_per_image=logits_per_image,
                    logits_per_text=logits_per_text,
                    targets=targets,
                )  # shape [batch_size]
                loss_t = loss_t.mean()

            preds_img_t, preds_text_t = self.compute_preds(
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_text,
            )
            preds_imgs.append(preds_img_t.detach_().cpu().numpy())
            preds_texts.append(preds_text_t.detach_().cpu().numpy())
            labels.append(targets.detach_().cpu().numpy())

            total_loss += loss_t
            steps = steps + 1
            if steps >= self.args.steps_per_eval_epoch:
                break

        total_loss /= steps

        labels = np.concatenate(labels)
        preds_imgs = np.concatenate(preds_imgs)
        preds_texts = np.concatenate(preds_texts)

        prec_img, rec_img, f_score_img, _ = precision_recall_fscore_support(
            labels,
            preds_imgs,
            average="macro",
        )
        accuracy_img = accuracy_score(
            labels,
            preds_imgs,
        )

        prec_text, rec_text, f_score_text, _ = precision_recall_fscore_support(
            labels,
            preds_texts,
            average="macro",
        )
        accuracy_text = accuracy_score(
            labels,
            preds_texts,
        )

        scores = dict(
            eval_loss=total_loss,
            # imagre
            eval_acc_img=accuracy_img,
            eval_prec_img=prec_img,
            eval_rec_img=rec_img,
            eval_f_score_img=f_score_img,
            # text
            eval_acc_text=accuracy_text,
            eval_prec_text=prec_text,
            eval_rec_text=rec_text,
            eval_f_score_text=f_score_text,
        )

        return scores
