import sys, evaluate as EV
models = ["twin", "process_da", "persistence", "gbm", "static_mlp", "gru", "transformer", "crn",
          "abl_no_attn", "abl_calendar", "abl_no_ports", "abl_joint", "abl_pf", "abl_pf_no_obsop",
          "abl_no_images", "abl_color", "abl_no_iot", "twin_s01"]
EV.main(models, conformal_models=["twin", "gru", "transformer", "gbm", "crn", "process_da"])
