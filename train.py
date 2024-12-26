import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tap import Tap
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hifigan.dataset import MelDataset, LogMelSpectrogram
from hifigan.discriminator import (
    HifiganDiscriminator,
    feature_loss,
    discriminator_loss,
    generator_loss,
)
from hifigan.generator import HifiganGenerator
from hifigan.utils import load_checkpoint, save_checkpoint, plot_spectrogram

logger = logging.getLogger(__name__)

BETAS = (0.8, 0.99)


def train_model(rank, world_size, args: 'TrainCommandParser'):
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://localhost:54321",
    )

    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    if args.generator_config is not None:
        with args.generator_config.open("r") as f:
            generator_config = yaml.safe_load(f)
        generator = HifiganGenerator(**generator_config).to(rank)
    else:
        generator = HifiganGenerator().to(rank)
    logger.info("Generator:")
    logger.info(generator)

    discriminator = HifiganDiscriminator().to(rank)

    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    optimizer_generator = optim.AdamW(
        generator.parameters(),
        lr=args.base_learning_rate if not args.finetune else args.finetune_learning_rate,
        betas=BETAS,
        weight_decay=args.weight_decay,
    )
    optimizer_discriminator = optim.AdamW(
        discriminator.parameters(),
        lr=args.base_learning_rate if not args.finetune else args.finetune_learning_rate,
        betas=BETAS,
        weight_decay=args.weight_decay,
    )

    scheduler_generator = optim.lr_scheduler.ExponentialLR(
        optimizer_generator, gamma=args.learning_rate_decay
    )
    scheduler_discriminator = optim.lr_scheduler.ExponentialLR(
        optimizer_discriminator, gamma=args.learning_rate_decay
    )

    train_dataset = MelDataset(
        root=args.dataset_dir,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        train=True,
        finetune=args.finetune,
    )
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    validation_dataset = MelDataset(
        root=args.dataset_dir,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        train=False,
        finetune=args.finetune,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    melspectrogram = LogMelSpectrogram().to(rank)

    if args.resume is not None:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            generator=generator,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            scheduler_generator=scheduler_generator,
            scheduler_discriminator=scheduler_discriminator,
            rank=rank,
            logger=logger,
            finetune=args.finetune,
        )
    else:
        global_step, best_loss = 0, float("inf")

    if args.finetune:
        global_step, best_loss = 0, float("inf")

    n_epochs = args.epochs
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"batch size: {args.batch_size}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"total of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()
        average_loss_mel = average_loss_discriminator = average_loss_generator = 0
        for i, (wavs, mels, tgts) in enumerate(tqdm(train_loader, desc=f"Train [Epoch {epoch}]"), 1):
            wavs, mels, tgts = wavs.to(rank), mels.to(rank), tgts.to(rank)

            # Discriminator
            optimizer_discriminator.zero_grad()

            wavs_ = generator(mels.squeeze(1))
            mels_ = melspectrogram(wavs_)

            scores, _ = discriminator(wavs)
            scores_, _ = discriminator(wavs_.detach())

            loss_discriminator, _, _ = discriminator_loss(scores, scores_)

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Generator
            optimizer_generator.zero_grad()

            scores, features = discriminator(wavs)
            scores_, features_ = discriminator(wavs_)

            loss_mel = F.l1_loss(mels_, tgts)
            loss_features = feature_loss(features, features_)
            loss_generator_adversarial, _ = generator_loss(scores_)
            loss_generator = 45 * loss_mel + loss_features + loss_generator_adversarial

            loss_generator.backward()
            optimizer_generator.step()

            global_step += 1

            average_loss_mel += (loss_mel.item() - average_loss_mel) / i
            average_loss_discriminator += (
                                                  loss_discriminator.item() - average_loss_discriminator
                                          ) / i
            average_loss_generator += (
                                              loss_generator.item() - average_loss_generator
                                      ) / i

            if rank == 0:
                if global_step % args.log_interval == 0:
                    writer.add_scalar(
                        "train/loss_mel",
                        loss_mel.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_generator",
                        loss_generator.item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/loss_discriminator",
                        loss_discriminator.item(),
                        global_step,
                    )

            if global_step % args.validation_interval == 0:
                generator.eval()

                average_validation_loss = 0
                for j, (wavs, mels, tgts) in tqdm(enumerate(validation_loader, 1),
                                                  desc=f"Val [Epoch {epoch}]"):
                    wavs, mels, tgts = wavs.to(rank), mels.to(rank), tgts.to(rank)

                    with torch.no_grad():
                        wavs_ = generator(mels.squeeze(1))
                        mels_ = melspectrogram(wavs_)

                        length = min(mels_.size(-1), tgts.size(-1))

                        loss_mel = F.l1_loss(mels_[..., :length], tgts[..., :length])

                    average_validation_loss += (
                                                       loss_mel.item() - average_validation_loss
                                               ) / j

                    if rank == 0:
                        if j <= args.num_generated_examples:
                            writer.add_audio(
                                f"generated/wav_{j}",
                                wavs_.squeeze(0),
                                global_step,
                                sample_rate=16000,
                            )
                            writer.add_figure(
                                f"generated/mel_{j}",
                                plot_spectrogram(mels_.squeeze().cpu().numpy()),
                                global_step,
                            )

                generator.train()
                discriminator.train()

                if rank == 0:
                    writer.add_scalar(
                        "validation/mel_loss", average_validation_loss, global_step
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, mel loss: {average_validation_loss:.4f}"
                    )

                new_best = best_loss > average_validation_loss
                if new_best or global_step % args.checkpoint_interval == 0:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = average_validation_loss

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.checkpoint_dir,
                            generator=generator,
                            discriminator=discriminator,
                            optimizer_generator=optimizer_generator,
                            optimizer_discriminator=optimizer_discriminator,
                            scheduler_generator=scheduler_generator,
                            scheduler_discriminator=scheduler_discriminator,
                            step=global_step,
                            loss=average_validation_loss,
                            best=new_best,
                            logger=logger,
                        )

        scheduler_discriminator.step()
        scheduler_generator.step()

        logger.info(
            f"train -- epoch: {epoch}, mel loss: {average_loss_mel:.4f}, generator loss: {average_loss_generator:.4f}, discriminator loss: {average_loss_discriminator:.4f}"
        )

    dist.destroy_process_group()


class TrainCommandParser(Tap):
    dataset_dir: Path  # path to the preprocessed data directory
    checkpoint_dir: Path  # path to the checkpoint directory
    resume: Path = None
    finetune: bool = False
    generator_config: Path = None

    batch_size: int = 8
    segment_length: int = 8320
    hop_length: int = 160
    sample_rate: int = 16000
    base_learning_rate: float = 2e-4
    finetune_learning_rate: float = 1e-4
    learning_rate_decay: float = 0.999
    weight_decay: float = 1e-5
    epochs: int = 3100
    log_interval: int = 5
    validation_interval: int = 1000
    num_generated_examples: int = 10
    checkpoint_interval: int = 5000

    def configure(self) -> None:
        self.add_argument("dataset_dir")
        self.add_argument("checkpoint_dir")


if __name__ == "__main__":
    args = TrainCommandParser().parse_args()

    # display training setup info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")

    # clear handlers
    logger.handlers.clear()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
