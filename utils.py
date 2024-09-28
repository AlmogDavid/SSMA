import copy
import os
from typing import Optional
import pandas as pd


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


class Singleton(object):
  def __new__(cls, *args, **kwargs):
    if hasattr(cls, 'instance'):
        raise RuntimeError("Singleton already instantiated, use get_instance method")
    else:
      cls.instance = super(Singleton, cls).__new__(cls)
    return cls.instance

  @classmethod
  def get_instance(cls):
      return cls.instance


class ResultsLogger(Singleton):
    """
    Object used for logging results for SSMA aggregation experiments
    """

    def __init__(self,
                 exp_name: str,
                 wandb_project_name: str = "ssma"):
        self._exp_name = exp_name
        self._run_id = -1
        if "USE_WANDB" in os.environ:
            if "WANDB_PROJECT" in os.environ:
                wandb_project_name = os.environ["WANDB_PROJECT"]
            print(f"Using WANDB project: {wandb_project_name}")
        else:
            wandb_project_name = None
        self._wandb_project_name = wandb_project_name
        self._wandb = None
        self._raw_results = []
        self._agg_results = []
        self._internal_step = 0
        self._setup_called = False

    @staticmethod
    def _is_json_serializable(obj):
        import json
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    def _finalize_last_run(self):
        df = pd.DataFrame.from_dict(self._raw_results)
        if len(df) > 0:
            self._agg_results.append(df)

    def mark_new_run(self):
        self._finalize_last_run()
        self._run_id += 1

    def log(self, **kwargs):
        if "step" in kwargs:
            step = kwargs["step"]
            del kwargs["step"]
        else:
            step = self._internal_step
            self._internal_step += 1

        if self._wandb is not None:
            self._wandb.log(kwargs, step=step)
        assert "step" not in kwargs, "Step is a reserved keyword for logging"
        kwargs["step"] = step
        self._raw_results.append(copy.deepcopy(kwargs))

    def setup(self, config: Optional[dict]):
        if self._setup_called:
            raise RuntimeError("Setup called twice")
        self._setup_called = True
        if self._wandb_project_name is not None:
            import wandb
            wandb.init(project=self._wandb_project_name, group=self._exp_name)
            if config is not None:
                wandb.config.update(config)
            self._wandb = wandb
        else:
            self._wandb = None

    def finalize(self):
        self._finalize_last_run()

    def __enter__(self) -> "ResultsLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()
        if len(self._agg_results) == 0:
            print("No results to finalize")
            return

        summary_res = pd.concat([d.tail(1) for d in self._agg_results], axis=0).reset_index().describe()
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.precision', 2)
        print("Summary results:")
        print(summary_res)

        if self._wandb is not None:
            summary_dict = {}
            for k, vd in summary_res.to_dict().items():
                for vk, v in vd.items():
                    if vk in ("min", "max", "mean", "std"):
                        summary_dict[f"{k}_{vk}"] = v
            self._wandb.log(summary_dict)

            print(f"Finishing wandb run: {self._wandb.run.name}")
            self._wandb.finish()
            self._wandb = None
