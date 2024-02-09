# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.

import logging, os

class VeryUsefulLoggerFormatter(logging.Formatter):
    """ A very useful logger formatter lets you locate where a printed log is coming from.
        This class is written by Zhenwei (https://github.com/ParadoxZW).
    """
    def format(self, record):
        pathname = record.pathname
        parts = pathname.split(os.sep)
        start_idx = max(0, len(parts) - (self.imp_log_fflevel + 1))
        relevant_path = os.sep.join(parts[start_idx:])
        record.custom_path = relevant_path
        return super().format(record)

    @classmethod
    def init_logger_help_function(cls, name, level=logging.INFO):
        imp_silient_others = bool(os.environ.get("IMP_SILIENT_OTHERS", False))
        is_silent = imp_silient_others and os.environ.get("LOCAL_RANK", None) not in ["0", None]
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR if is_silent else level)
        logger.propagate = False
        # customize log format
        log_format = "[%(asctime)s] [%(levelname)s] [%(custom_path)s:%(lineno)d] %(message)s"
        # log_format = "[%(asctime)s] [logger:%(name)s] [%(levelname)s] [%(custom_path)s:%(lineno)d] %(message)s"
        formatter = cls(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        formatter.imp_log_fflevel = int(os.environ.get("IMP_LOG_FFLEVEL", "3"))
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


logger = VeryUsefulLoggerFormatter.init_logger_help_function(__name__)
VeryUsefulLoggerFormatter.init_logger_help_function("", level=logging.WARNING)
VeryUsefulLoggerFormatter.init_logger_help_function("transformers.generation", level=logging.WARNING)
VeryUsefulLoggerFormatter.init_logger_help_function("transformers.modeling_utils", level=logging.ERROR)
# VeryUsefulLoggerFormatter.init_logger_help_function("deepspeed")

if os.environ.get("LOCAL_RANK", None) in ["0", None]:
    logger.info(
        f"\n\n\033[95m\033[4mWelcome to Imp! We use a custom logger in the Imp project. It is supported to use environment variables to control the logger:\n"
        "  - `export IMP_LOG_FFLEVEL={number}` to set the number of father folders to be printed.\n"
        "  - `export IMP_SILIENT_OTHERS=true` to set multiple processes to be silent except the rank-0 process, which is useful for distributed training.\n"
        "You are free to access the code where this info is came from and modify the log behavior. The Imp team wishes you a good day:)\033[0m\033[24m\n"
    )


from .model import LlavaLlamaForCausalLM
