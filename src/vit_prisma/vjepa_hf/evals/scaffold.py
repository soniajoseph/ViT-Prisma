
import importlib

from vit_prisma.vjepa_hf.src.utils.logging import get_logger

logger = get_logger("Eval runner scaffold")


def main(eval_name, args_eval, resume_preempt=False):
    logger.info(f"Running evaluation: {eval_name}")
    import_path = f'vit_prisma.vjepa_hf.evals.{eval_name}'
    # if eval_name.startswith("app."):
    #     import_path = f"{eval_name}.eval"
    # else:
    #     import_path = f"evals.{eval_name}.eval"
    return importlib.import_module(import_path).main(args_eval=args_eval, resume_preempt=resume_preempt)
