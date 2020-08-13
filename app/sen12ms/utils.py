import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.metrics
import torch
from skimage import exposure
import datetime

import urllib
import zipfile
from tqdm import tqdm

def metrics(y_true, y_pred, labels=None):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred, labels=labels)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0,
                                                         labels=labels)
    jaccard_score_micro = sklearn.metrics.jaccard_score(y_true, y_pred, average="micro", labels=labels)
    jaccard_score_macro = sklearn.metrics.jaccard_score(y_true, y_pred, average="macro", labels=labels)
    jaccard_score_weighted = sklearn.metrics.jaccard_score(y_true, y_pred, average="weighted", labels=labels)
    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        jaccard_score_micro=jaccard_score_micro,
        jaccard_score_macro=jaccard_score_macro,
        jaccard_score_weighted=jaccard_score_weighted
    )


def report(test_predictions, test_targets, test_ids, pattern, classes):
    samples = pd.DataFrame([test_ids, test_targets.numpy(), test_predictions.numpy()],
                           index=["id", "y_true", "y_pred"]).T.set_index("id")
    samples.to_csv(pattern + "_predictions.csv")

    return metrics(test_targets, test_predictions)

def parse_experiment_name(output_folder, model, shots, ways):
    time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%m")
    return os.path.join(output_folder, f"{model}-{shots}shots-{ways}ways", time)

def log_and_snapshot(model, predictions, targets, ids, trainlosses, testlosses, output_folder, classes, index, log):
    def save(model, filename):
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pattern = os.path.join(
        output_folder, f"{index:06d}"
    )
    if len(predictions) > 0:
        metrics = report(torch.cat(predictions), torch.cat(targets), np.hstack(ids), pattern=pattern, classes=classes)
    else:
        metrics = dict()

    modelpath = pattern + "_model.pth"
    save(model, modelpath)
    metrics["model"] = modelpath
    metrics["episode"] = index
    metrics["trainloss"] = torch.stack(trainlosses).mean().numpy()
    metrics["testloss"] = torch.stack(testlosses).mean().numpy()
    log.append(metrics)

    log_df = pd.DataFrame(log).set_index("episode")
    log_df.to_csv(output_folder + ".csv")

    if (log_df.iloc[-1]["testloss"] < log_df.iloc[:-1]["testloss"]).all():
        save(model, modelpath)

    return log


def plot_prototypes(prototypes, test_embeddings, test_targets):
    plt.scatter(prototypes[0, :, 0].detach().numpy(), prototypes[0, :, 1].detach().numpy(), c=np.arange(5), marker="x",
                label="prototypes")
    plt.scatter(test_embeddings[0, :, 0].detach().numpy(), test_embeddings[0, :, 1].detach().numpy(),
                c=test_targets[0].detach().numpy(), label="test embeddings")
    plt.show()


def to_rgb(image):
    image = image[[5, 4, 3], :, :].transpose(0, 2).cpu().numpy()
    image = exposure.rescale_intensity(image)
    return exposure.adjust_gamma(image, gamma=0.8, gain=1)


def tensorboard_batch_figure(batch, summary_writer, classes, targets, predictions, imgsize=4, global_step=None):
    train_img, train_label, train_ids = batch["train"]
    test_img, test_label, test_ids = batch["test"]

    # infer batch_size, shots, ways from the data
    batch_size = train_label.shape[0]
    unique_classes, count_per_class = train_label[0].unique(return_counts=True)
    num_ways = len(unique_classes)
    num_shots = int(count_per_class[0])
    assert num_ways * num_shots == train_label.shape[1]

    for task_id in range(batch_size):
        fig, axs = plt.subplots(2, num_shots * num_ways,
                                figsize=(num_shots * num_ways * imgsize, 2 * imgsize),
                                sharey=True)

        np.array(classes)[targets].reshape(batch_size, -1)

        #### First Row: Support
        axs_row = axs[0]

        rgb_images = np.stack([to_rgb(image) for image in train_img[task_id]])
        id = np.array(train_ids)[:, task_id]
        train_label_row = train_label[task_id]
        train_label_row = np.array(classes)[train_label_row]

        for ax, image, id_, target in zip(axs_row, rgb_images, id, train_label_row):
            ax.imshow(image)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(f"{target}")
        axs_row[0].set_ylabel("support")

        #### Second Row: Query
        axs_row = axs[1]

        targets_row = targets.reshape(batch_size, -1)[task_id]
        predictions_row = predictions.reshape(batch_size, -1)[task_id]

        targets_row = np.array(classes)[targets_row]
        predictions_row = np.array(classes)[predictions_row]

        rgb_images = np.stack([to_rgb(image) for image in test_img[task_id]])
        id = np.array(test_ids)[:, task_id]
        for ax, image, target, prediction in zip(axs_row, rgb_images, targets_row, predictions_row):
            ax.imshow(image)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(f"{target} (pred {prediction})")
        axs_row[0].set_ylabel("query")

        name = f"Region {id[0].split('/')[1]} ({id[0].split('/')[0]})"
        summary_writer.add_figure(f"meta-test: {name}", figure=fig, global_step=global_step)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"file exists in {output_path}. specify overwrite=True if intended")


def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path) as zip:
        for zip_info in zip.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, target_dir)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
