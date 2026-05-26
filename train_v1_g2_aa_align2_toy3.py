# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

# from models.tts.fastspeech2.fs2_trainer import FastSpeech2Trainer
# from models.tts.vits.vits_trainer import VITSTrainer
# from models.tts.valle.valle_trainer import VALLETrainer
# from models.tts.naturalspeech2.ns2_trainer import NS2Trainer
# from models.tts.gpt_tts.gpt_tts_trainer import NS2Trainer as GPTTTSTrainer
# from models.codec.kmeans.kmeans_trainer import KMeansTrainer
# from models.codec.amphion_codec.codec_trainer import CodecTrainer
# from models.tts.soundstorm.soundstorm_trainer import SoundStormTrainer
# from models.tts.text2semantic.t2s_trainer import T2STrainer
# from models.codec.kmeans.repcodec_trainer import RepCodecTrainer
# from models.codec.amphion_codec.vocoder_trainer import VocoderTrainer
# from models.tts.voicebox.voicebox_trainer import VoiceBoxTrainer

# from models.tts.difft2s.aligner_trainer import AlignerTrainer
# from models.tts.difft2s.difft2s_trainer import DiffT2STrainer
# from models.tts.difft2s.difft2s_distill_trainer import (
#     DiffT2STrainer as DiffT2SDistillTrainer,
# )
# from models.tts.difft2s.difft2s_critic_trainer import (
#     DiffT2STrainer as DiffT2SCriticTrainer,
# )

# from models.tts.difft2a.difft2a_trainer import DiffT2ATrainer
# from models.tts.difft2s.ar_dur_trainer import DurPredictorTrainer
# from models.tts.difft2s.fm_dur_trainer import DurPredictorFMTrainer


# from models.tts.ar_se.llm_trainer import LLMTrainer, DebugLLMTrainer
# from models.tts.ar_se.llm_trainer_dense import LLMTrainer as LLMTrainerDense
# from models.tts.ar_se.post_training.dpo_trainer import SEDPOTrainer

# from models.tts.flow_se.fmt_trainer import FMTTrainer
# from models.tts.flow_se.post_training.dpo_trainer import DPOFMTTrainer

from models.aa_v1_g2_align2_toy3.flow_matching_transformer.fmt_trainer import FMTTrainer as MIDI2AudioFMTTrainer

from utils.util import load_config


def build_trainer(args, cfg):
    supported_trainer = {
        # "FastSpeech2": FastSpeech2Trainer,
        # "VITS": VITSTrainer,
        # "VALLE": VALLETrainer,
        # "NaturalSpeech2": NS2Trainer,
        # "GPTTTS": GPTTTSTrainer,
        # "KMeans": KMeansTrainer,
        # "Codec": CodecTrainer,
        # "SoundStorm": SoundStormTrainer,
        # "T2S": T2STrainer,
        # "RepCodec": RepCodecTrainer,
        # "VoiceBox": VoiceBoxTrainer,
        # "Vocoder": VocoderTrainer,
        # # "Aligner": AlignerTrainer,
        # "DiffT2S": DiffT2STrainer,
        # "DiffT2SDistill": DiffT2SDistillTrainer,
        # "DiffT2SCritic": DiffT2SCriticTrainer,
        # "DurPredictorFM": DurPredictorFMTrainer,
        # "DiffT2A": DiffT2ATrainer,
        # "DurPredictor": DurPredictorTrainer,
        # "ar_se": LLMTrainer,
        # "ar_se_debug": DebugLLMTrainer,
        # "ar_se_dense": LLMTrainerDense,
        # "ar_se_dense_dpo": SEDPOTrainer,


        # "flow_se": FMTTrainer,
        # "flow_se_dpo": DPOFMTTrainer,

        "flow_matching_transformer_anyrir": MIDI2AudioFMTTrainer,
    }

    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def cuda_relevant(deterministic=False):
    torch.cuda.empty_cache()
    # TF32 on Ampere and above
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    # Deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume", action="store_true", help="The model name to restore"
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        default="resume",
        help="Resume training or finetuning.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--dataloader_seed",
        type=int,
        default=1,
        help="Seed for dataloader",
    )

    # VALLETrainer.add_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Data Augmentation
    # if (
    #     type(cfg.preprocess.data_augment) == list
    #     and len(cfg.preprocess.data_augment) > 0
    # ):
    #     new_datasets_list = []
    #     for dataset in cfg.preprocess.data_augment:
    #         new_datasets = [
    #             f"{dataset}_pitch_shift" if cfg.preprocess.use_pitch_shift else None,
    #             (
    #                 f"{dataset}_formant_shift"
    #                 if cfg.preprocess.use_formant_shift
    #                 else None
    #             ),
    #             f"{dataset}_equalizer" if cfg.preprocess.use_equalizer else None,
    #             f"{dataset}_time_stretch" if cfg.preprocess.use_time_stretch else None,
    #         ]
    #         new_datasets_list.extend(filter(None, new_datasets))
    #     cfg.dataset.extend(new_datasets_list)

    # # CUDA settings
    cuda_relevant()

    # Build trainer
    trainer = build_trainer(args, cfg)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    trainer.train_loop()


if __name__ == "__main__":
    main()
