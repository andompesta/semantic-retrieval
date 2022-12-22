import comet_ml
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser

from semantic_retrieval.common import Task
from semantic_retrieval.datareaders import get_farfetch_dataloader
from semantic_retrieval.utils import (
    compute_warmup_steps,
    ContrastiveLearningTask,
    save_checkpoint,
)
from semantic_retrieval.model import CLIP
from semantic_retrieval.model.config import LargeClipConfig
from semantic_retrieval.optim import (
    unfreeze_layer_params,
    get_optimizer,
    get_group_params,
    get_linear_scheduler_with_warmup,
)


class FarFetchModelTraining(Task):

    def parse_args(self):
        parser = ArgumentParser()
        parser.add_argument(
            "--run_name",
            required=True,
            help="run name used for logging",
        )

        parser.add_argument(
            "--dataset_base_path",
            type=str,
            required=True,
            help="basepath of the raw dataset",
        )

        parser.add_argument(
            "--checkpoint_base_path",
            type=str,
            required=True,
            help="basepath of the raw dataset",
        )

        parser.add_argument(
            "--device",
            type=str,
            default="mps",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=50,
        )

        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
        )

        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=1000,
        )

        parser.add_argument(
            "--batches_per_epoch",
            type=int,
            default=10000,
        )

        parser.add_argument("--gradient_accumulation_steps",
                            default=1,
                            type=int)
        parser.add_argument("--freeze_layer", default=3, type=int)
        parser.add_argument("--optim_method", default="adam")
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--n_gpus", default=1, type=int)
        parser.add_argument("--eval_every", default=5, type=int)

        args = compute_warmup_steps(args)
        args, _ = parser.parse_known_args()

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def launch(self):
        np.random.seed(0)
        torch.manual_seed(0)

        dataset_base_path = Path(self.args.dataset_base_path)
        train_dataset_path = dataset_base_path.joinpath("training")
        validation_dataset_path = dataset_base_path.joinpath("validation")
        checkpoint_path = Path(self.args.checkpoint_base_path)

        model_path = checkpoint_path.joinpath(
            "pre-traiend",
            "ViT-L-14@336px.pt",
        )


        device = torch.device(self.args.device)

        state_dict = torch.load(
            "/dbfs" + model_path.as_posix(),
            map_location="cpu",
        )

        model = CLIP(**LargeClipConfig()).load_state_dict(
            state_dict,
            strict=True,
        )

        # setup optimizers
        named_params = list(model.named_parameters())
        group_params = get_group_params(
            named_params,
            self.args.weight_decay,
            no_decay_patterns=["bias", "ln_.+weight", "ln_.+bias"],
        )
        unfreeze_layer_params(
            named_params,
            img_layer=6,
            text_layer=3,
        )
        optim = get_optimizer(
            method=self.args.optim_method,
            params=group_params,
            lr=self.args.lr,
        )
        scheduler = get_linear_scheduler_with_warmup(
            optim,
            self.args.num_warmup_steps,
            self.args.num_training_steps,
        )

        model = model.to(device)

        if torch.cuda.device_count() > 1 and self.args.n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[1, 0])

        task = ContrastiveLearningTask(
            name="contrastive_learning",
            args=self.args,
        )

        
        experimet = comet_ml.Experiment(
            project_name="clip",
            auto_output_logging="simple",
        )

        experimet.set_name(self.args.run_name)
        experimet.add_tag("farfetch-large")
        experimet.log_parameters(vars(self.args))


        best_f1 = 0.0
        train_metrics = []
        eval_metrics = []
        
        try:
            for epoch in range(1, self.args.epochs + 1):
                experimet.set_epoch(epoch)
                with get_farfetch_dataloader(
                        path="file:///dbfs" +
                        train_dataset_path.as_posix()
                ) as train_dl:
                    train_metric = task.train(
                        model=model,
                        optimizer=optim,
                        scheduler=scheduler,
                        dataloader=train_dl,
                        device=device,
                    )

                    experimet.log_metrics(train_metric, epoch=epoch)
                    train_metrics.append(train_metric)

                    print(
                        "epoch:{epoch} \t acc-img:{acc_img} \t acc-text:{acc_text} \t loss:{loss}".format(
                            epoch=epoch,
                            acc_img=train_metric["train_accuracy_img"],
                            acc_text=train_metric["train_accuracy_text"],
                            loss=train_metric["train_loss"],
                        )
                    )

                if epoch % self.args.eval_every == 0 or epoch == 1:

                with get_farfetch_dataloader(
                        path="file:///dbfs" +
                        validation_dataset_path.as_posix()
                ) as eval_dl:
                    is_best = False
                    eval_metric = task.eval(
                        model=model,
                        dataloader=train_dl,
                        device=device,
                    )

                    if eval_metric["eval_f_score_text"] > best_f1:
                        best_f1 = eval_metric["eval_f_score_text"]
                        is_best = True
                    

                    if isinstance(model, torch.nn.DataParallel):
                        state_dict = dict(
                            [(n, p.to("cpu")) for n, p in model.module.state_dict().items()]
                        )
                    else:
                        state_dict = dict(
                            [(n, p.to("cpu")) for n, p in model.state_dict().items()]
                        )

                    save_checkpoint(
                        path_=checkpoint_path.joinpath(
                            "clip",
                            "farfetch",
                        ),
                        state=state_dict,
                        is_best=is_best,
                        filename=f"ckp_{epoch}.pt",
                    )
        finally:
            experimet.end()



if __name__ == '__main__':
    task = FarFetchDatasetPreparation()
    task.launch()
