# Research configurations

These configurations preserve exploratory CGAT work: curriculum stages, fine-tuning,
learned feedback/residual heads, action scaling, and radial-basis schedules.
They are research recipes, not a sequence required to run the project.

Start with `configs/default.yaml` (1–3 links), `configs/cgat_3link.yaml` (three links),
or `configs/smoke.yaml` (workflow check). Use `python -m training.train_cgat --help`
to select a variant and, where required, a compatible `--init-checkpoint`.

Historical configurations may assume locally trained weights or a particular
fine-tuning stage. They do not ship with checkpoints or reproduced scores.
