def save_checkpoint(state, is_best, best_model_name, accelerator):
    if is_best:
        accelerator.save(state, best_model_name)
