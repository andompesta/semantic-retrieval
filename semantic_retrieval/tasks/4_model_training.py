import comet_ml
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser

from semantic_retrieval.common import Task
from semantic_retrieval.datareaders import (
    get_farfetch_dataloader,
    get_single_batch_farfetch_dataloader,
)
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
    get_constant_scheduler,
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
            default="cuda",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=50,
        )

        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=15,
        )

        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=50,
        )

        parser.add_argument(
            "--batches_per_epoch",
            type=int,
            # default=2300,
            default=1,
        )

        parser.add_argument(
            "--gradient_accumulation_steps",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--unfreeze_text_layer",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--unfreeze_image_layer",
            default=15,
            type=int,
        )
        parser.add_argument("--optim_method", default="adam")
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--lr", default=5e-5, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--n_gpus", default=1, type=int)
        parser.add_argument("--eval_every", default=1, type=int)
        parser.add_argument("--num_workers", default=5, type=int)
        args, _ = parser.parse_known_args()
        args = compute_warmup_steps(args)

        print("\n\n-----------")
        for k, v in vars(args).items():
            print("{} \t {}".format(k, v))
        print("-----------\n\n")

        return args

    def launch(self):
        np.random.seed(0)
        torch.manual_seed(0)

        checkpoint_path = Path(self.args.checkpoint_base_path)
        dataset_base_path = Path(self.args.dataset_base_path)
        train_dataset_path = dataset_base_path.joinpath("training")
        validation_dataset_path = dataset_base_path.joinpath("validation")
        train_dataset_path = "file:///dbfs" + train_dataset_path.as_posix()
        validation_dataset_path = "file:///dbfs" + validation_dataset_path.as_posix(
        )
        print("train_dataset_path \t {}".format(train_dataset_path))
        print("eval_dataset_path \t {}".format(validation_dataset_path))

        model_path = checkpoint_path.joinpath(
            "pre-trained",
            "ViT-L-14@336px.pt",
        )
        device = torch.device(self.args.device)
        # device = torch.device("cpu")

        state_dict = torch.load(
            model_path.as_posix(),
            map_location="cpu",
        )
        model_config = LargeClipConfig()
        model = CLIP(**model_config.to_dict())
        model.load_state_dict(
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
            img_layer=self.args.unfreeze_image_layer,
            text_layer=self.args.unfreeze_text_layer,
        )
        optim = get_optimizer(
            method=self.args.optim_method,
            params=group_params,
            lr=self.args.lr,
        )
        # scheduler = get_linear_scheduler_with_warmup(
        #     optim,
        #     self.args.num_warmup_steps,
        #     self.args.num_training_steps,
        # )
        scheduler = get_constant_scheduler(optim=optim)

        model = model.to(device)

        if torch.cuda.device_count() > 1 and self.args.n_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=[1, 0])

        task = ContrastiveLearningTask(
            name="contrastive_learning",
            args=self.args,
        )

        experimet = comet_ml.Experiment(
            api_key="rs4hQR3VBTHUBaQE1raJ09NRV",
            project_name="clip",
            auto_output_logging="simple",
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_host=True,
        )

        experimet.set_name(self.args.run_name)
        experimet.add_tag("farfetch-large")
        experimet.log_parameters(vars(self.args))

        best_f1 = 0.0

        with get_single_batch_farfetch_dataloader(
                path=train_dataset_path,
                batch_size=self.args.train_batch_size,
                reader_pool_type="process",
                workers_count=self.args.num_workers,
        ) as train_dl, get_single_batch_farfetch_dataloader(
                path=train_dataset_path,
                batch_size=self.args.train_batch_size,
                reader_pool_type="process",
                workers_count=self.args.num_workers,
        ) as eval_dl:

            try:
                for epoch in range(1, self.args.epochs + 1):
                    experimet.set_epoch(epoch)

                    train_metric = task.train(
                        model=model,
                        optimizer=optim,
                        scheduler=scheduler,
                        dataloader=train_dl,
                        device=device,
                    )

                    experimet.log_metrics(train_metric, epoch=epoch)

                    print(
                        "epoch:{epoch} \t acc-img:{acc_img} \t acc-text:{acc_text} \t loss:{loss}"
                        .format(
                            epoch=epoch,
                            acc_img=train_metric["train_accuracy_img"],
                            acc_text=train_metric["train_accuracy_text"],
                            loss=train_metric["train_loss"],
                        ))

                    if epoch % self.args.eval_every == 0 or epoch == 1:
                        is_best = False
                        eval_metric = task.eval(
                            model=model,
                            dataloader=eval_dl,
                            device=device,
                        )
                        # log eval metrics
                        experimet.log_metrics(eval_metric, epoch=epoch)

                        if eval_metric["eval_f_score_text"] > best_f1:
                            best_f1 = eval_metric["eval_f_score_text"]
                            is_best = True

                        if isinstance(model, torch.nn.DataParallel):
                            state_dict = dict([
                                (n, p.to("cpu"))
                                for n, p in model.module.state_dict().items()
                            ])
                        else:
                            state_dict = dict([
                                (n, p.to("cpu"))
                                for n, p in model.state_dict().items()
                            ])

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
    task = FarFetchModelTraining()
    task.launch()
