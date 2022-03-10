

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    import sklearn
    metrics = {}
    metrics['accuracy'] = sklearn.metrics.accuracy_score(labels, preds)
    metrics['f1'] = sklearn.metrics.f1_score(labels, preds)
    metrics['precision'] = sklearn.metrics.precision_score(labels, preds)
    metrics['recall'] = sklearn.metrics.recall_score(labels, preds)

    return metrics
    pass

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
    import torch
    configuration = RobertaConfig()
    model = RobertaForSequenceClassification(configuration).from_pretrained("roberta-base")

    return model
    pass
